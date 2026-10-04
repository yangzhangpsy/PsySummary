"""Shared, GUI-independent preparation for summaries and saved filter rules."""

import ast
import csv
import json
import os
import tempfile

import numpy as np
import pandas as pd


PSYDATA_SCHEMA_PREFIX = '# PsySummary typed data v1 '


def _psydata_scalar(value):
    """Encode scalar objects without pickle, eval, or type inference."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f'Cannot preserve an object value of type {type(value).__name__} in psydata.')


def write_psydata_stream(frame, stream, chunk_rows=65536, check_cancelled=None):
    """Write embedded column dtypes and bounded CSV chunks into one psydata stream."""
    if not frame.columns.is_unique or not all(isinstance(name, str) for name in frame.columns):
        raise ValueError('psydata requires unique textual column names.')
    if chunk_rows < 1 or (not len(frame.columns) and len(frame)):
        raise ValueError('psydata requires positive chunk sizes and columns for nonempty data.')
    columns = []
    for name in frame.columns:
        dtype = frame[name].dtype
        kind = ('category' if isinstance(dtype, pd.CategoricalDtype) else
                'json' if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype) else
                'boolean' if pd.api.types.is_bool_dtype(dtype) else
                'numeric' if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype) else
                'datetime' if pd.api.types.is_datetime64_any_dtype(dtype) else None)
        if kind is None:
            raise ValueError(f'Cannot preserve psydata column {name!r} with dtype {dtype}.')
        entry = {'name': name, 'dtype': str(dtype), 'kind': kind}
        if isinstance(dtype, pd.StringDtype):
            # Keep v1 files readable by older PsySummary readers, which accept
            # 'string' but not pandas 3's inferred 'str' alias.
            entry['dtype'] = 'string'
            if getattr(dtype, 'na_value', pd.NA) is not pd.NA:
                entry['string_na'] = 'nan'
        if kind == 'category':
            entry.update(categories=[_psydata_scalar(value) for value in dtype.categories], ordered=dtype.ordered)
        columns.append(entry)
    stream.write(PSYDATA_SCHEMA_PREFIX + json.dumps({'columns': columns}, ensure_ascii=True) + '\n')
    starts = range(0, len(frame), chunk_rows) if len(frame) else [0]
    for start in starts:
        if check_cancelled:
            check_cancelled()
        chunk = frame.iloc[start:start + chunk_rows].copy()
        for entry in columns:
            if entry['kind'] in ('json', 'category'):
                name = entry['name']
                text_column = entry['kind'] == 'json' and entry['dtype'] == 'string'
                chunk[name] = [json.dumps(
                    None if text_column and pd.isna(value) else _psydata_scalar(value),
                    ensure_ascii=True) for value in chunk[name]]
        chunk.to_csv(stream, sep='|', quoting=csv.QUOTE_NONNUMERIC,
                     index=False, header=start == 0, na_rep='')


def write_psydata(frame, path):
    """Atomically save typed psydata for GUI-equivalent script replay."""
    path = os.path.abspath(path)
    if os.path.lexists(path) and (os.path.islink(path) or not os.path.isfile(path)):
        raise ValueError('Cannot replace a directory or symbolic link with psydata.')
    descriptor, temporary = tempfile.mkstemp(prefix='.psysummary-data-', suffix='.tmp', dir=os.path.dirname(path))
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as stream:
            write_psydata_stream(frame, stream, max(1, min(65536, 100000 // max(1, len(frame.columns)))))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_psydata(path):
    """Restore typed psydata exactly, retaining the legacy reader for older files."""
    with open(path, 'r', encoding='utf-8-sig', newline='') as stream:
        first = stream.readline()
        if not first.startswith(PSYDATA_SCHEMA_PREFIX):
            if first.startswith('# PsySummary typed data '):
                raise ValueError('Unsupported psydata format version.')
            return pd.read_csv(path, sep='|', quoting=csv.QUOTE_NONNUMERIC, index_col=False)
        columns = json.loads(first[len(PSYDATA_SCHEMA_PREFIX):])['columns']
        if not isinstance(columns, list):
            raise ValueError('Invalid psydata column schema.')
        names = [entry['name'] for entry in columns]
        if len(set(names)) != len(names) or not all(isinstance(name, str) for name in names):
            raise ValueError('Invalid psydata column names.')
        if not columns:
            return pd.DataFrame()
        if next(csv.reader(stream, delimiter='|'), None) != names:
            raise ValueError('psydata headers do not match the saved column schema.')
        try:
            frame = pd.read_csv(stream, sep='|', names=names, header=None,
                                dtype=str, na_filter=False, index_col=False)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame(columns=names)
        for entry in columns:
            name, kind, dtype = entry['name'], entry['kind'], entry['dtype']
            values = frame[name]
            if kind in ('json', 'category'):
                values = [json.loads(value) for value in values]
                if kind == 'category':
                    category = pd.CategoricalDtype(entry['categories'], ordered=entry['ordered'])
                    if not pd.Series(values, dtype=object).dropna().isin(category.categories).all():
                        raise ValueError(f'Invalid category value in {name!r}.')
                    frame[name] = pd.Series(values, dtype=category)
                else:
                    if dtype not in ('object', 'string', 'str'):
                        raise ValueError(f'Invalid textual dtype {dtype}.')
                    if dtype != 'object':
                        missing_kind = entry.get('string_na', 'nan' if dtype == 'str' else 'NA')
                        if missing_kind not in ('NA', 'nan'):
                            raise ValueError(f'Invalid Text missing-value type in {name!r}.')
                        # Older pandas-3 writers used JSON NaN rather than null.
                        def is_text_missing(value):
                            return value is None or (missing_kind == 'nan' and
                                                     isinstance(value, float) and np.isnan(value))
                        if not all(isinstance(value, str) or is_text_missing(value) for value in values):
                            raise ValueError(f'Invalid Text data in {name!r}.')
                        if missing_kind == 'nan':
                            values = [np.nan if is_text_missing(value) else value for value in values]
                            try:
                                dtype = pd.StringDtype(na_value=np.nan)
                            except TypeError:
                                # pandas < 2.3 cannot represent NaN-backed strings.
                                dtype = object
                        else:
                            dtype = pd.StringDtype()
                    frame[name] = pd.Series(values, dtype=dtype)
            elif kind in ('numeric', 'boolean'):
                parsed_dtype = pd.api.types.pandas_dtype(dtype)
                if kind == 'boolean':
                    if not pd.api.types.is_bool_dtype(parsed_dtype) or not values.isin(['True', 'False', '']).all():
                        raise ValueError(f'Invalid Boolean data in {name!r}.')
                    converted = [None if value == '' else value == 'True' for value in values]
                else:
                    if not pd.api.types.is_numeric_dtype(parsed_dtype) or pd.api.types.is_bool_dtype(parsed_dtype) or pd.api.types.is_complex_dtype(parsed_dtype):
                        raise ValueError(f'Invalid numeric dtype {dtype}.')
                    number = int if pd.api.types.is_integer_dtype(parsed_dtype) else float
                    converted = [None if value == '' else number(value) for value in values]
                frame[name] = pd.Series(converted, dtype=parsed_dtype)
            elif kind == 'datetime':
                parsed_dtype = pd.api.types.pandas_dtype(dtype)
                if not pd.api.types.is_datetime64_any_dtype(parsed_dtype):
                    raise ValueError(f'Invalid datetime dtype {dtype}.')
                frame[name] = pd.to_datetime(values.replace('', None), utc=isinstance(parsed_dtype, pd.DatetimeTZDtype)).astype(parsed_dtype)
            else:
                raise ValueError(f'Unsupported psydata column kind {kind}.')
        return frame


def grouped_filter_estimates(frame, group_variables, variable, mad=False, count_needed=False):
    """Broadcast group estimates by row position without scanning the frame per group."""
    # A private RangeIndex avoids alignment changes with duplicated input indices.
    values = frame[variable].reset_index(drop=True)
    keys = [frame[name].reset_index(drop=True) for name in dict.fromkeys(group_variables)]
    grouped = values.groupby(keys, sort=False, observed=True, dropna=True)
    if mad:
        center = grouped.transform('median')
        deviations = (values - center).abs()
        scale = 1.4826 * deviations.groupby(
            keys, sort=False, observed=True, dropna=True).transform('median')
    else:
        center = grouped.transform('mean')
        scale = grouped.transform('std', ddof=1)
    count = grouped.transform('count').to_numpy() if count_needed else None
    return center.to_numpy().reshape(-1, 1), scale.to_numpy().reshape(-1, 1), count


def prepare_summary_frame(frame, variable, operation):
    """Keep raw values for Count/Mode and convert only a numeric target's working copy."""
    if operation in {'Count', 'Mode'} or pd.api.types.is_numeric_dtype(frame[variable]):
        return frame
    prepared = frame.copy(deep=False)
    prepared[variable] = pd.to_numeric(frame[variable], errors='coerce')
    return prepared


def safe_mode(values):
    """Return the first mode, or a missing result when no non-missing values exist."""
    modes = values.mode(dropna=True)
    return np.nan if modes.empty else modes.iloc[0]


def split_filter_rule(rule):
    """Split a saved rule once so colons in its selected values remain untouched."""
    variable, separator, expression = rule.partition(':')
    if not separator or not variable.strip():
        raise ValueError(f'Invalid filter rule: {rule!r}.')
    return variable.strip(), expression.strip()


def is_range_expression(expression):
    """Recognize comparison syntax only at the beginning of a range expression."""
    return expression.lstrip().startswith(('<', '>'))


def format_checklist_rule(variable, values, numeric=False):
    """Write a literal-list checklist without ambiguous separators or quote escaping."""
    selected = [
        (value == 'True' if value in ('True', 'False') else float(value))
        if numeric else str(value) for value in values
    ]
    return f'{variable}:Checklist {selected!r}'


def parse_checklist_values(expression):
    """Read new literal-list checklists and legacy equals-delimited saved rules."""
    expression = expression.strip()
    if expression.startswith('Checklist '):
        try:
            node = ast.parse(expression[len('Checklist '):], mode='eval').body
            if not isinstance(node, ast.List):
                raise ValueError('Checklist values must be a list.')
            values = []
            for item in node.elts:
                if isinstance(item, ast.Name) and item.id in {'inf', 'nan'}:
                    value = float(item.id)
                elif (isinstance(item, ast.UnaryOp) and isinstance(item.op, ast.USub)
                      and isinstance(item.operand, ast.Name) and item.operand.id == 'inf'):
                    value = -np.inf
                else:
                    value = ast.literal_eval(item)
                values.append(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError('Invalid Checklist value list.') from error
        if not isinstance(values, list) or any(
                not isinstance(value, (str, int, float, bool)) for value in values):
            raise ValueError('Checklist values must be a list of text or numbers.')
        return values

    # Legacy text values were enclosed, not Python-escaped. Preserve literal
    # backslashes and apostrophes from older Setup files rather than evaluating them.
    values = []
    remaining = expression
    while remaining:
        if not remaining.startswith('='):
            raise ValueError(f'Invalid Checklist expression: {expression!r}.')
        remaining = remaining[1:].lstrip()
        if not remaining:
            break
        if remaining[0] in "'\"":
            quote = remaining[0]
            end = 1
            while True:
                end = remaining.find(quote, end)
                if end < 0:
                    raise ValueError('Unclosed quote in a legacy Checklist rule.')
                tail = remaining[end + 1:].lstrip()
                if not tail or tail.startswith('='):
                    values.append(remaining[1:end])
                    remaining = tail
                    break
                end += 1
        else:
            token, separator, tail = remaining.partition('=')
            try:
                values.append(float(token.strip()))
            except ValueError as error:
                raise ValueError(f'Invalid numeric Checklist value: {token!r}.') from error
            remaining = ('=' + tail).lstrip() if separator else ''
    return values
