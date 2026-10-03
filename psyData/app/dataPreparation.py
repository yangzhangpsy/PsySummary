"""Shared, GUI-independent preparation for summaries and saved filter rules."""

import ast

import numpy as np
import pandas as pd


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
