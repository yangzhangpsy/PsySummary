"""Shared, restricted computed-variable expressions for GUI and script replay."""

import ast
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import boxcox


def runBoxcox(values):
    """Return Box-Cox transformed values without the estimated lambda."""
    transformed, _lambda = boxcox(values)
    return transformed


class _DataReferences(ast.NodeTransformer):
    """Normalize data references without changing column names or literals."""

    def visit_Attribute(self, node):
        if (isinstance(node.value, ast.Name)
                and (node.value.id, node.attr) in {
                    ('self', 'dataFrame'), ('self', 'data'), ('aggData', 'data')}):
            return ast.copy_location(
                ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                              attr='data', ctx=ast.Load()), node)
        return self.generic_visit(node)


def _attribute_chain(node):
    """Return a dotted name for simple attribute expressions."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return '.'.join(reversed(parts + [node.id]))
    return None


class _ExpressionValidator(ast.NodeVisitor):
    """Allow numeric expressions and read-only calls, not arbitrary Python."""

    # Positional counts also prevent supplying a NumPy ufunc's out positionally.
    calls = {
        'runBoxcox': (1, 1, frozenset()),
        'abs': (1, 1, frozenset()),
        'np.log': (1, 1, frozenset()),
        'np.exp': (1, 1, frozenset()),
        'np.sqrt': (1, 1, frozenset()),
        'np.abs': (1, 1, frozenset()),
        'np.logical_and': (2, 2, frozenset()),
        'np.logical_or': (2, 2, frozenset()),
        'np.where': (3, 3, frozenset()),
        'np.clip': (1, 3, frozenset({'a_min', 'a_max'})),
    }
    binary_operators = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.BitAnd, ast.BitOr)
    unary_operators = (ast.UAdd, ast.USub, ast.Invert)
    comparison_operators = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

    def generic_visit(self, node):
        raise ValueError(f'Unsupported expression element: {type(node).__name__}')

    def visit_Expression(self, node):
        self.visit(node.body)

    def visit_Name(self, node):
        if node.id not in {'self', 'np', 'runBoxcox', 'abs'}:
            raise ValueError(f'Unsupported name in expression: {node.id}')

    def visit_Attribute(self, node):
        chain = _attribute_chain(node)
        if chain != 'self.data' and chain not in self.calls:
            raise ValueError(f'Unsupported attribute access in expression: {chain}')

    def visit_Constant(self, node):
        return None

    def visit_List(self, node):
        for element in node.elts:
            self.visit(element)

    visit_Tuple = visit_List

    def visit_Subscript(self, node):
        self.visit(node.value)
        self.visit(node.slice)

    def visit_Slice(self, node):
        for value in (node.lower, node.upper, node.step):
            if value is not None:
                self.visit(value)

    def visit_BinOp(self, node):
        if not isinstance(node.op, self.binary_operators):
            raise ValueError(f'Unsupported operator: {type(node.op).__name__}')
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node):
        if not isinstance(node.op, self.unary_operators):
            raise ValueError(f'Unsupported operator: {type(node.op).__name__}')
        self.visit(node.operand)

    def visit_Compare(self, node):
        self.visit(node.left)
        for operator in node.ops:
            if not isinstance(operator, self.comparison_operators):
                raise ValueError(f'Unsupported comparison: {type(operator).__name__}')
        for comparator in node.comparators:
            self.visit(comparator)

    def visit_Call(self, node):
        name = node.func.id if isinstance(node.func, ast.Name) else _attribute_chain(node.func)
        if name not in self.calls:
            raise ValueError(f'Unsupported function in expression: {name}')
        minimum, maximum, keywords = self.calls[name]
        if not minimum <= len(node.args) <= maximum:
            raise ValueError(f'{name} accepts {minimum}–{maximum} positional arguments; output arguments are not supported.')
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            if keyword.arg not in keywords:
                raise ValueError(f'Unsupported keyword argument for {name}: {keyword.arg}. Output arguments are not supported.')
            self.visit(keyword.value)


EXPRESSION_HELP = ('Supported functions: ' + ', '.join(_ExpressionValidator.calls)
                   + '. Output arguments (out=) are not allowed.')


def _parse_expression(expression):
    """Parse, normalize and validate one computed-variable expression."""
    parsed = _DataReferences().visit(ast.parse(expression, mode='eval'))
    ast.fix_missing_locations(parsed)
    _ExpressionValidator().visit(parsed)
    return parsed


def to_aggregate_expression(expression):
    """Produce replayable source with only data-object references rewritten."""
    return ast.unparse(_parse_expression(expression))


def evaluate_expression(expression, data_frame):
    """Evaluate an approved expression without exposing the GUI object."""
    parsed = _parse_expression(expression)
    return eval(compile(parsed, '<computed-variable-expression>', 'eval'),
                {'__builtins__': {}},
                {'self': SimpleNamespace(data=data_frame), 'np': np,
                 'runBoxcox': runBoxcox, 'abs': np.abs})


def validate_variable_name(name, data_frame):
    """Normalize a new column name and reject empty, control or duplicate names."""
    if not isinstance(name, str):
        raise ValueError('The target variable name must be text.')
    name = name.strip()
    if not name or any(ord(character) < 32 for character in name):
        raise ValueError('Enter a non-empty target variable name without control characters.')
    if '@' in name or ':' in name:
        raise ValueError('The target variable name cannot contain "@" or ":"; '
                         'these characters are reserved for aggregation and filter rules.')
    if any(str(column).strip() == name for column in data_frame.columns):
        raise ValueError(f'The target variable {name!r} already exists. Choose a different name.')
    return name


def validate_result(result, data_frame):
    """Return one aligned scalar-valued column without implicit missing-row fills."""
    if isinstance(result, pd.Series):
        if len(result) != len(data_frame) or not result.index.equals(data_frame.index):
            raise ValueError('The result must cover every data row in the same index order.')
        column = result.copy(deep=False)
    elif isinstance(result, np.ndarray) and result.ndim == 0:
        column = pd.Series(result.item(), index=data_frame.index)
    elif pd.api.types.is_scalar(result):
        column = pd.Series(result, index=data_frame.index)
    elif isinstance(result, (list, tuple, np.ndarray)):
        array = np.asarray(result)
        if array.ndim != 1 or len(array) != len(data_frame):
            raise ValueError('The result must be a one-dimensional value for every data row.')
        column = pd.Series(array, index=data_frame.index)
    else:
        raise ValueError('The result must be a scalar or one-dimensional column, not a module, function or table.')
    if column.dtype == object and any(not pd.api.types.is_scalar(value) for value in column):
        raise ValueError('Each result cell must contain a scalar value, not a module, function or nested object.')
    return column


def _input_missing_mask(node, data_frame):
    """Track source missingness in row order, selecting only active where branches."""
    empty = np.zeros(len(data_frame), dtype=bool)
    if isinstance(node, ast.Subscript):
        root = node
        while isinstance(root, ast.Subscript):
            root = root.value
        if _attribute_chain(root) == 'self.data':
            # Read only original data here, including scalar selections and slices.
            source = evaluate_expression(ast.unparse(node), data_frame)
            missing = np.asarray(pd.isna(source), dtype=bool)
            try:
                return np.broadcast_to(missing, empty.shape)
            except ValueError:
                # Uncertain alignment must not suppress a missing-result warning.
                return empty
        # A slice of a computed result can reorder rows; do not assume alignment.
        return empty
    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) not in (1, len(data_frame)):
            return empty
        missing = np.array([_input_missing_mask(child, data_frame)[index]
                            for index, child in enumerate(node.elts)], dtype=bool)
        return np.broadcast_to(missing, empty.shape)
    if isinstance(node, ast.Call) and _attribute_chain(node.func) == 'np.where':
        condition = evaluate_expression(ast.unparse(node.args[0]), data_frame)
        try:
            condition = np.broadcast_to(np.asarray(condition, dtype=bool), empty.shape)
        except ValueError:
            return empty
        return np.where(condition, _input_missing_mask(node.args[1], data_frame),
                        _input_missing_mask(node.args[2], data_frame))
    for child in ast.iter_child_nodes(node):
        empty |= _input_missing_mask(child, data_frame)
    return empty


def result_warning(column, expression, data_frame):
    """Describe newly missing values and infinities without warning for missing-input propagation."""
    missing = column.isna().to_numpy(dtype=bool)
    input_missing = (_input_missing_mask(_parse_expression(expression).body, data_frame)
                     if missing.any() else np.zeros(len(data_frame), dtype=bool))
    new_missing = int(np.count_nonzero(missing & ~input_missing))
    if pd.api.types.is_integer_dtype(column.dtype) or pd.api.types.is_bool_dtype(column.dtype):
        infinite = 0
    elif pd.api.types.is_numeric_dtype(column.dtype):
        # Bound temporary storage; real columns never need a whole complex copy.
        dtype = complex if pd.api.types.is_complex_dtype(column.dtype) else float
        infinite = 0
        for start in range(0, len(column), 65536):
            chunk = column.iloc[start:start + 65536]
            # Preserve an infinite component even when the other component is NaN.
            values = (chunk.to_numpy(dtype=dtype) if dtype is complex
                      else chunk.to_numpy(dtype=dtype, na_value=np.nan))
            infinite += int(np.count_nonzero(np.isinf(values)))
    else:
        infinite = sum(isinstance(value, (float, complex, np.floating, np.complexfloating))
                       and np.isinf(value) for value in column)
    if not new_missing and not infinite:
        return ''
    return (f'The result contains {new_missing} newly missing value(s) and '
            f'{infinite} infinite value(s). Missing values propagated from selected inputs '
            f'are not counted as newly missing. Check the input values and expression.')


def prepare_variable(name, expression, data_frame):
    """Validate a calculation before changing data or recording a script."""
    name = validate_variable_name(name, data_frame)
    normalized = to_aggregate_expression(expression)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        column = validate_result(evaluate_expression(normalized, data_frame), data_frame)
        warning = result_warning(column, normalized, data_frame)
    return name, column, normalized, warning
