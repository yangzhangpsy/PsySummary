"""Versioned, non-executable PsySummary result snapshots (JSON + numeric NPY in ZIP)."""

from datetime import date, datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np
import pandas as pd

from app.diagnosticCurveCache import (
    CACHE_VERSION, CURVE_VERSION, DiagnosticCurveCache, attach_cache, cache_key,
)

FORMAT = 'PsySummary Results'
VERSION = 1
MAX_EXPANDED_BYTES = 2 * 1024 ** 3
MAX_MANIFEST_BYTES = 64 * 1024 ** 2


class _CheckedStream:
    """Check interruption during bounded NumPy ZIP reads and writes."""

    def __init__(self, stream, check):
        self.stream, self.check = stream, check

    def write(self, value):
        self.check()
        return self.stream.write(value)

    def read(self, size=-1):
        self.check()
        return self.stream.read(size)

    def __getattr__(self, name):
        return getattr(self.stream, name)


class _Codec:
    """Encode only explicitly supported data types; never import or instantiate saved classes."""

    def __init__(self, archive, check):
        self.archive, self.check = archive, check
        self.counter = 0
        self._encoded_arrays = {}
        self._decoded_arrays = {}

    def encode(self, value):
        self.check()
        if value is pd.NA:
            return {'type': 'NA'}
        if value is pd.NaT:
            return {'type': 'NaT'}
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            return value if math.isfinite(value) else {'type': 'float', 'value': str(value)}
        if isinstance(value, pd.DataFrame):
            return {'type': 'frame', 'index': self.encode(value.index), 'columns': self.encode(value.columns),
                    'series': [self.encode(value.iloc[:, i].reset_index(drop=True)) for i in range(value.shape[1])]}
        if isinstance(value, pd.MultiIndex):
            return {'type': 'multiindex', 'levels': self.encode(list(value.levels)),
                    'codes': self.encode(list(value.codes)), 'names': self.encode(list(value.names))}
        if isinstance(value, pd.RangeIndex):
            return {'type': 'range', 'start': value.start, 'stop': value.stop, 'step': value.step,
                    'name': self.encode(value.name)}
        if isinstance(value, pd.Index):
            return {'type': 'index', 'series': self.encode(pd.Series(value)), 'name': self.encode(value.name)}
        if isinstance(value, pd.Series):
            dtype = value.dtype
            if isinstance(dtype, pd.CategoricalDtype):
                return {'type': 'category', 'categories': self.encode(value.cat.categories),
                        'codes': self.encode(value.cat.codes.to_numpy()), 'ordered': dtype.ordered}
            if isinstance(dtype, pd.StringDtype):
                return {'type': 'string', 'nan': dtype.na_value is not pd.NA, 'values': self.encode(value.tolist())}
            return {'type': 'series', 'dtype': str(dtype),
                    'values': self.encode(value.to_numpy() if isinstance(dtype, np.dtype) else value.tolist())}
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return {'type': 'object_array', 'shape': list(value.shape), 'values': self.encode(value.ravel().tolist())}
            if value.dtype.fields:
                raise ValueError('Structured numeric arrays are not supported in result files.')
            previous = self._encoded_arrays.get(id(value))
            if previous is not None:
                return {'type': 'array', 'name': previous[1]}
            name = f'arrays/{self.counter}.npy'
            self.counter += 1
            self._encoded_arrays[id(value)] = (value, name)
            with self.archive.open(name, 'w', force_zip64=True) as stream:
                np.lib.format.write_array(_CheckedStream(stream, self.check), value, allow_pickle=False)
            return {'type': 'array', 'name': name}
        if isinstance(value, np.generic):
            return {'type': 'numpy_scalar', 'array': self.encode(np.asarray(value))}
        if isinstance(value, pd.Timestamp):
            return {'type': 'timestamp', 'value': value.isoformat()}
        if isinstance(value, pd.Timedelta):
            return {'type': 'timedelta', 'value': value.value}
        if isinstance(value, (datetime, date)):
            return {'type': 'datetime' if isinstance(value, datetime) else 'date', 'value': value.isoformat()}
        if isinstance(value, timedelta):
            return {'type': 'python_timedelta', 'days': value.days, 'seconds': value.seconds,
                    'microseconds': value.microseconds}
        if isinstance(value, complex):
            return {'type': 'complex', 'parts': self.encode([value.real, value.imag])}
        if isinstance(value, (list, tuple)):
            return {'type': 'tuple' if isinstance(value, tuple) else 'list',
                    'items': [self.encode(item) for item in value]}
        if isinstance(value, dict):
            return {'type': 'dict', 'items': [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        raise ValueError(f'Unsupported result value type: {type(value).__name__}.')

    def decode(self, node, depth=0):
        self.check()
        if depth > 80:
            raise ValueError('Result data nesting is too deep.')
        if node is None or type(node) in (str, int, bool, float):
            return node
        if not isinstance(node, dict):
            raise ValueError('Invalid typed result data.')
        kind = node.get('type')
        dec = lambda item: self.decode(item, depth + 1)
        if kind == 'NA': return pd.NA
        if kind == 'NaT': return pd.NaT
        if kind == 'float':
            if node['value'] not in ('nan', 'inf', '-inf'): raise ValueError('Invalid float marker.')
            return float(node['value'])
        if kind in ('list', 'tuple'):
            values = [dec(item) for item in node['items']]
            return tuple(values) if kind == 'tuple' else values
        if kind == 'dict':
            pairs = [(dec(k), dec(v)) for k, v in node['items']]
            result = dict(pairs)
            if len(result) != len(pairs): raise ValueError('Duplicate result keys.')
            return result
        if kind == 'array':
            name = node['name']
            if not re.fullmatch(r'arrays/\d+\.npy', name): raise ValueError('Invalid array reference.')
            if name in self._decoded_arrays:
                return self._decoded_arrays[name]
            with self.archive.open(name) as raw:
                stream = _CheckedStream(raw, self.check)
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported numeric array version.')
                size = math.prod(shape) * dtype.itemsize
                if dtype.hasobject or dtype.fields or size > self.archive.getinfo(name).file_size - stream.tell():
                    raise ValueError('Invalid or unsafe numeric array.')
                stream.seek(0)
                result = np.lib.format.read_array(stream, allow_pickle=False)
                self._decoded_arrays[name] = result
                return result
        if kind == 'numpy_scalar': return dec(node['array'])[()]
        if kind == 'object_array':
            values = dec(node['values'])
            if math.prod(node['shape']) != len(values): raise ValueError('Invalid object array shape.')
            array = np.empty(len(values), dtype=object)
            array[:] = values
            return array.reshape(node['shape'])
        if kind == 'complex': return complex(*dec(node['parts']))
        if kind == 'timestamp': return pd.Timestamp(node['value'])
        if kind == 'timedelta': return pd.Timedelta(node['value'], unit='ns')
        if kind == 'datetime': return datetime.fromisoformat(node['value'])
        if kind == 'date': return date.fromisoformat(node['value'])
        if kind == 'python_timedelta':
            return timedelta(days=node['days'], seconds=node['seconds'], microseconds=node['microseconds'])
        if kind == 'category':
            return pd.Series(pd.Categorical.from_codes(dec(node['codes']), dec(node['categories']), ordered=node['ordered']))
        if kind == 'string':
            try:
                dtype = pd.StringDtype(na_value=np.nan if node['nan'] else pd.NA)
            except TypeError:
                dtype = object if node['nan'] else pd.StringDtype()
            return pd.Series(dec(node['values']), dtype=dtype)
        if kind == 'series':
            text = node['dtype']
            # No arbitrary extension dtype constructors from saved files.
            if not re.fullmatch(r'(object|bool|boolean|[Uu]?[Ii]nt(8|16|32|64)|[Ff]loat(16|32|64)|complex(64|128)|datetime64\[[^\]]+\]|timedelta64\[[^\]]+\])', text):
                raise ValueError(f'Unsupported result column dtype: {text}.')
            return pd.Series(dec(node['values']), dtype=text)
        if kind == 'index': return pd.Index(dec(node['series']).array, name=dec(node['name']), tupleize_cols=False)
        if kind == 'range':
            if len(range(node['start'], node['stop'], node['step'])) > 10000000: raise ValueError('Result index is too large.')
            return pd.RangeIndex(node['start'], node['stop'], node['step'], name=dec(node['name']))
        if kind == 'multiindex':
            return pd.MultiIndex(levels=dec(node['levels']), codes=dec(node['codes']), names=dec(node['names']))
        if kind == 'frame':
            rows, columns = dec(node['index']), dec(node['columns'])
            series = [dec(item) for item in node['series']]
            if len(series) != len(columns) or any(len(s) != len(rows) for s in series):
                raise ValueError('Result table shape does not match its axes.')
            frame = pd.concat(series, axis=1) if series else pd.DataFrame(index=range(len(rows)))
            frame.index, frame.columns = rows, columns
            return frame
        raise ValueError(f'Unsupported result data tag: {kind}.')


def validate_snapshot(snapshot):
    """Reject malformed table structure before it can replace an existing GUI result."""
    if not isinstance(snapshot, dict): raise ValueError('Invalid result snapshot.')
    for name in ('rows', 'columns', 'labels', 'rules'):
        if not isinstance(snapshot.get(name), list) or not all(isinstance(s, str) for s in snapshot[name]):
            raise ValueError(f'Invalid result {name}.')
    results = snapshot.get('results')
    if not isinstance(results, list) or not results or len(results) != len(snapshot['labels']):
        raise ValueError('Result tables and labels do not match.')
    for frame in results:
        if snapshot['rows'] or snapshot['columns']:
            if not isinstance(frame, pd.DataFrame): raise ValueError('Grouped results require tables.')
            if snapshot['rows'] and frame.index.nlevels != len(snapshot['rows']): raise ValueError('Invalid row levels.')
            if snapshot['columns'] and frame.columns.nlevels != len(snapshot['columns']): raise ValueError('Invalid column levels.')
            if frame.empty: raise ValueError('A grouped result table is empty.')
        elif isinstance(frame, pd.DataFrame) and frame.shape != (1, 1):
            raise ValueError('An ungrouped result table must contain one value.')
    if type(snapshot.get('decimals')) is not int or not 0 <= snapshot['decimals'] <= 12:
        raise ValueError('Invalid display precision.')
    if not isinstance(snapshot.get('targets'), list) or not isinstance(snapshot.get('fit_records'), list):
        raise ValueError('Invalid analysis metadata.')
    if not isinstance(snapshot.get('metadata', {}), dict): raise ValueError('Invalid result metadata.')
    for record in snapshot['fit_records']:
        if not isinstance(record, dict) or not isinstance(record.get('distribution'), str):
            raise ValueError('Invalid fitted result record.')
        if not isinstance(record.get('result_prefix'), str) or not isinstance(record.get('group_vars'), list):
            raise ValueError('Invalid fit-to-table mapping.')
        if len(record.get('group_values', ())) != len(record['group_vars']):
            raise ValueError('Invalid fitted group values.')
        if record['group_vars'] != snapshot['rows'] + snapshot['columns']:
            raise ValueError('Fitted groups do not match the result axes.')
    return snapshot


def save_result_archive(path, snapshot, check=lambda: None):
    """Atomically save table precision, fit snapshots and available numeric curves together."""
    validate_snapshot(snapshot)
    path = Path(path).absolute()
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError('Cannot replace a directory or symbolic link with results.')
    descriptor, temporary = tempfile.mkstemp(prefix='.psyresult-', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            codec = _Codec(archive, check)
            # Serialize records individually; transient cache objects and paths never enter the archive.
            records, curves = [], []
            for record in snapshot['fit_records']:
                check()
                records.append(codec.encode({k: v for k, v in record.items() if not k.startswith('_diagnostic_')}))
                cache = record.get('_diagnostic_curve_cache')
                views = {}
                for mode in (False, True):
                    payload = cache.get(cache_key(record, mode)) if cache is not None else None
                    if payload is not None and not payload.get('failed'):
                        views[str(int(mode))] = codec.encode(payload)
                curves.append(views)
            data = {k: v for k, v in snapshot.items() if k != 'fit_records'}
            manifest = {'format': FORMAT, 'version': VERSION, 'saved_at': datetime.now(timezone.utc).isoformat(),
                        'curve_version': CURVE_VERSION, 'cache_version': CACHE_VERSION,
                        'snapshot': codec.encode(data), 'records': records, 'curves': curves}
            content = json.dumps(manifest, ensure_ascii=True, allow_nan=False).encode('utf-8')
            if len(content) > MAX_MANIFEST_BYTES: raise ValueError('Result metadata exceeds the 64 MiB limit.')
            archive.writestr('manifest.json', content)
            if sum(info.file_size for info in archive.infolist()) > MAX_EXPANDED_BYTES:
                raise ValueError('Results exceed the 2 GiB uncompressed archive limit.')
        check()
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def load_result_archive(path, check=lambda: None):
    """Validate a self-contained archive without extracting paths or running saved code."""
    cache = DiagnosticCurveCache()
    warnings = []
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > 100000 or len({i.filename for i in infos}) != len(infos):
                raise ValueError('Invalid or excessively large result archive.')
            if sum(i.file_size for i in infos) > MAX_EXPANDED_BYTES:
                raise ValueError('Results exceed the 2 GiB uncompressed archive limit.')
            if any(i.filename != 'manifest.json' and not re.fullmatch(r'arrays/\d+\.npy', i.filename) for i in infos):
                raise ValueError('Unexpected file in result archive.')
            if archive.getinfo('manifest.json').file_size > MAX_MANIFEST_BYTES:
                raise ValueError('Result metadata is too large.')
            check()
            manifest = json.loads(archive.read('manifest.json'))
            if manifest.get('format') != FORMAT or manifest.get('version') != VERSION:
                raise ValueError('Unsupported PsySummary result format/version.')
            codec = _Codec(archive, check)
            snapshot = codec.decode(manifest['snapshot'])
            snapshot['fit_records'] = [codec.decode(item) for item in manifest['records']]
            validate_snapshot(snapshot)
            views = manifest.get('curves', [])
            if not isinstance(views, list):
                views = []
                warnings.append('Saved curve metadata is invalid; View Fit will rebuild curves from the fitted data.')
            compatible = manifest.get('curve_version') == CURVE_VERSION and manifest.get('cache_version') == CACHE_VERSION
            for index, record in enumerate(snapshot['fit_records']):
                check()
                if any(k.startswith('_diagnostic_') for k in record): raise ValueError('Invalid saved cache reference.')
                attach_cache(record, cache)
                if not compatible or index >= len(views) or not isinstance(views[index], dict):
                    warnings.append('Some saved curves are unavailable or use another version; View Fit will rebuild them.')
                    continue
                for mode in (False, True):
                    item = views[index].get(str(int(mode)))
                    if item is None: continue
                    try:
                        payload = codec.decode(item)
                        # Round-trip through the numeric cache validator before GUI publication.
                        key = cache_key(record, mode)
                        cache.put(key, payload)
                        cache.clear_memory()
                        count = 2 if mode else max(1, len(record.get('specification', {}).get('response_values', [0])))
                        if cache.get(key, count) is None: raise ValueError('Invalid cached plot data.')
                    except InterruptedError:
                        raise
                    except Exception as error:
                        cache.clear_memory()
                        warnings.append(f'Saved diagnostic curves could not be loaded: {error}. View Fit will rebuild them.')
            snapshot['saved_at'] = manifest.get('saved_at', '')
        check()
        return snapshot, list(dict.fromkeys(warnings))
    except BaseException:
        cache.close()
        raise
