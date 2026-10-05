"""Result-owned, disposable numeric curve cache; never part of analysis scripts or setups."""

from collections import OrderedDict
import json
import os
from pathlib import Path
import tempfile
from threading import RLock
import uuid
import weakref

import numpy as np

from app.cognitiveModelSpec import COGNITIVE_MODEL_NAMES, RATCLIFF_MODEL
from app.rtDiagnosticCurves import _CurveBuilder


CACHE_VERSION = 1
CURVE_VERSION = 1  # Bump when numerical preparation or its plotting contract changes.
_CACHE = '_diagnostic_curve_cache'
_IDENTITY = '_diagnostic_curve_identity'


class DiagnosticCurveCache:
    """Bound RAM per result set and atomically store non-pickle numeric archives on disk."""

    def __init__(self, max_bytes=32 * 1024 * 1024, directory_parent=None):
        self.max_bytes = max_bytes
        self._directory_parent = directory_parent
        self._memory = OrderedDict()
        self._bytes = 0
        self._directory = None
        self._cleanup = None
        self._lock = RLock()
        self._io_lock = RLock()

    def _remember(self, key, payload):
        """Keep recently used arrays within the configured byte budget."""
        with self._lock:
            self._remember_locked(key, payload)

    def _remember_locked(self, key, payload):
        """Update the LRU while the short-lived memory lock is held."""
        size = len(payload['status'].encode('utf-8')) + sum(
            np.asarray(value).nbytes for _, _, args, _ in payload['commands'] for value in args)
        previous = self._memory.pop(key, None)
        if previous is not None:
            self._bytes -= previous[1]
        if size > self.max_bytes:
            return
        while self._memory and self._bytes + size > self.max_bytes:
            _, (_, count) = self._memory.popitem(last=False)
            self._bytes -= count
        self._memory[key] = (payload, size)
        self._bytes += size

    def peek(self, key):
        """Read RAM only; safe for the GUI's immediate rendering fast path."""
        with self._lock:
            item = self._memory.get(key)
            if item is not None:
                self._memory.move_to_end(key)
                return item[0]
        return None

    def get(self, key, color_count=None):
        """Load evicted curves off-thread; invalid or missing archives are cache misses."""
        with self._io_lock:
            cached = self.peek(key)
            if cached is not None:
                return cached
            if self._directory is None:
                return None
            try:
                with np.load(Path(self._directory.name) / (key + '.npz'), allow_pickle=False) as archive:
                    metadata = json.loads(archive['metadata'].tobytes().decode('utf-8'))
                    if (metadata['version'] != CACHE_VERSION or metadata['curve_version'] != CURVE_VERSION
                            or metadata['key'] != key or not isinstance(metadata['status'], str)):
                        return None
                    commands = []
                    for axis, method, names, options in metadata['commands']:
                        if axis not in (0, 1) or method not in ('plot', 'step', 'stairs'):
                            return None
                        allowed = {'color', 'edgecolor', 'label', 'linewidth', 'linestyle',
                                   'where', 'alpha', 'fill'}
                        if not isinstance(options, dict) or set(options) - allowed:
                            return None
                        for name in ('color', 'edgecolor'):
                            if name in options and (type(options[name]) is not int or options[name] < 0
                                                    or (color_count is not None and options[name] >= color_count)):
                                return None
                        if not isinstance(options.get('label', ''), str):
                            return None
                        if options.get('where', 'post') not in ('pre', 'post', 'mid'):
                            return None
                        if options.get('linestyle', '-') not in ('-', '--', '-.', ':'):
                            return None
                        if 'fill' in options and type(options['fill']) is not bool:
                            return None
                        for name in ('linewidth', 'alpha'):
                            if name in options and (not isinstance(options[name], (int, float))
                                                    or not np.isfinite(options[name]) or options[name] < 0):
                                return None
                        if options.get('alpha', 1) > 1:
                            return None
                        args = tuple(archive[name] for name in names)
                        if len(args) != 2 or any(a.ndim != 1 or a.dtype.kind not in 'fiu' for a in args):
                            return None
                        if len(args[1]) != len(args[0]) + (1 if method == 'stairs' else 0):
                            return None
                        commands.append((axis, method, args, options))
                    payload = {'commands': commands, 'status': metadata['status'], 'failed': False}
            except Exception:
                return None
            self._remember(key, payload)
            return payload

    def put(self, key, payload, cancel_check=lambda: False):
        """Publish a complete archive before exposing it as a reusable disk entry."""
        with self._io_lock:
            if cancel_check():
                raise InterruptedError('Curve preparation cancelled.')
            self._remember(key, payload)
            if self._directory is None:
                self._directory = tempfile.TemporaryDirectory(prefix='psysummary-curves-', dir=self._directory_parent)
                # Keep the directory alive until its result store is released, without a reference cycle.
                self._cleanup = weakref.finalize(self, self._directory.cleanup)
            arrays, commands = {}, []
            for index, (axis, method, args, options) in enumerate(payload['commands']):
                names = []
                for position, value in enumerate(args):
                    name = f'array_{index}_{position}'
                    arrays[name] = np.asarray(value)
                    names.append(name)
                commands.append((axis, method, names, options))
            metadata = {'version': CACHE_VERSION, 'curve_version': CURVE_VERSION,
                        'key': key, 'status': payload['status'], 'commands': commands}
            arrays['metadata'] = np.frombuffer(json.dumps(metadata).encode('utf-8'), dtype=np.uint8)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self._directory.name, suffix='.npz', delete=False) as stream:
                    temporary = Path(stream.name)
                    np.savez(stream, **arrays)
                if cancel_check():
                    raise InterruptedError('Curve preparation cancelled.')
                os.replace(temporary, Path(self._directory.name) / (key + '.npz'))
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def clear_memory(self):
        """Release RAM while retaining the on-disk fallback."""
        with self._lock:
            self._memory.clear()
            self._bytes = 0

    def close(self):
        """Remove only this cache's owned temporary directory after its tasks finish."""
        with self._io_lock:
            self.clear_memory()
            if self._directory is not None:
                self._cleanup()
                self._directory = None
                self._cleanup = None


def attach_cache(record, cache):
    """Give an immutable fit snapshot a unique identity, independent of model/group names."""
    if _CACHE not in record:
        record[_CACHE] = cache
        record[_IDENTITY] = uuid.uuid4().hex


def cache_key(record, mode):
    """Address one view of exactly one fitted result snapshot."""
    return record[_IDENTITY] + ('-accuracy' if mode else '-response')


def peek_curves(record, mode):
    """Return an in-memory hit without opening files or calculating curves."""
    return record[_CACHE].peek(cache_key(record, mode)) if _CACHE in record else None


def prepared_curves(record, mode, cancel_check):
    """Read or rebuild curves in a worker; cache failures never invalidate fitted parameters."""
    if cancel_check():
        raise InterruptedError('Curve preparation cancelled.')
    cache = record[_CACHE]
    key = cache_key(record, mode)
    color_count = (2 if mode else max(1, len(record['specification']['response_values']))) \
        if record.get('model') in COGNITIVE_MODEL_NAMES else 1
    payload = cache.get(key, color_count)
    if payload is None:
        payload = _CurveBuilder(mode).prepare(record, cancel_check)
        if not payload.get('failed'):
            try:
                cache.put(key, payload, cancel_check)
            except InterruptedError:
                raise
            except Exception as error:
                payload = dict(payload, cache_warning=f'Diagnostic disk cache unavailable: {error}')
    if cancel_check():
        raise InterruptedError('Curve preparation cancelled.')
    return payload


def prepare_record_curves(record, cache, cancel_check, warning):
    """Warm all supported views after a group fit, without changing its success status."""
    attach_cache(record, cache)
    modes = (False, True) if (record.get('model') == RATCLIFF_MODEL
                             and record.get('accuracy') is not None
                             and record.get('specification', {}).get('accuracy_variable')) else (False,)
    for mode in modes:
        try:
            payload = prepared_curves(record, mode, cancel_check)
            if payload.get('failed'):
                warning('Diagnostic curves could not be fully prepared; fitted parameters are retained. '
                        'Opening RT Fit Diagnostics will retry. ' + payload['status'])
            elif payload.get('cache_warning'):
                warning(payload['cache_warning'] + '; fitted parameters are retained.')
        except InterruptedError:
            raise
        except Exception as error:
            warning(f'Diagnostic curves unavailable: {error}. Fitted parameters are retained; '
                    'opening RT Fit Diagnostics will retry.')
