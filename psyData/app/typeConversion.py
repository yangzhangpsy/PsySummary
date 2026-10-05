"""Plan value-dependent setting updates without touching Qt or source data."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.cognitiveModelSpec import response_value_token, split_target
from app.dataPreparation import (
    is_range_expression, parse_checklist_values, split_filter_rule,
)
from app.expression import convert_variable_type


@dataclass
class TypeConversionRequest:
    """Carry a read-only source column and a GUI-thread settings snapshot to a worker."""

    series: pd.Series
    settings: dict


@dataclass
class TypeConversionPlan:
    """Keep migrated settings and confirmation warnings separate from the source column."""

    filters: list
    targets: list
    warnings: list


def _literal(value):
    """Produce setup/script-compatible scalars, never pandas missing-value expressions."""
    if pd.isna(value):
        return float('nan')
    return value.item() if hasattr(value, 'item') else value


def _convert_saved_value(value, original, target):
    """Preserve the source's numeric formatting when converting an unobserved saved level."""
    values = pd.Series([value])
    if pd.api.types.is_numeric_dtype(original.dtype) and pd.api.types.is_number(value):
        values = values.astype(original.dtype)
    return _literal(convert_variable_type(values, target).iloc[0])


def _check_numeric_meaning(original, converted, description):
    """Reject changed numeric inputs while allowing equivalent numeric text and Boolean values."""
    before = pd.to_numeric(original, errors='coerce')
    after = pd.to_numeric(converted, errors='coerce')
    equivalent = before.eq(after).fillna(False) | (before.isna() & after.isna())
    if not equivalent.all():
        raise ValueError(
            f'Type conversion would change the numeric values used by {description}. '
            'The existing setting cannot be preserved. No data were changed.')


def _migrate_checklist(original, converted, values, target, rule):
    """Use actual source values to preserve selections, including numeric-equivalent literals."""
    selected = original.isin(values).to_numpy(dtype=bool)
    migrated = [_literal(value) for value in pd.unique(converted.iloc[np.flatnonzero(selected)])]
    # Saved levels absent from the source remain in the setup rather than being silently discarded.
    absent = pd.Series(values, dtype=object)
    absent = absent.loc[~absent.isin(pd.unique(original))]
    migrated.extend(_convert_saved_value(value, original, target) for value in absent)
    expression = f'Checklist {migrated!r}'
    parsed = parse_checklist_values(expression)
    if not np.array_equal(selected, converted.isin(parsed).to_numpy(dtype=bool)):
        raise ValueError(
            f'Type conversion cannot preserve the selected rows of {rule!r}. '
            'Selected and unselected values may become indistinguishable, or a missing-value '
            'selection may no longer be representable. No data were changed.')
    return expression


def _migrate_response_mapping(original, converted, specification, target):
    """Keep each configured response's boundary/accumulator assignment, never infer a new one."""
    values = specification.get('response_values', [])
    mapping = specification.get('response_mapping', {})
    tokens = original.map(response_value_token)
    new_values, new_mapping, seen = [], {}, set()
    for value in values:
        mask = tokens.isin([response_value_token(value)]).to_numpy(dtype=bool)
        observed = pd.unique(converted.iloc[np.flatnonzero(mask)])
        if len(observed) > 1:
            raise ValueError(
                f'Type conversion splits response {value!r} into multiple values. '
                'Its saved response mapping cannot be preserved. No data were changed.')
        converted_value = (_literal(observed[0]) if len(observed)
                           else _convert_saved_value(value, original, target))
        token = response_value_token(converted_value)
        if token in seen or str(converted_value) in new_mapping:
            raise ValueError(
                'Type conversion merges distinct configured response values. '
                'Their boundary/accumulator assignments cannot be preserved. No data were changed.')
        seen.add(token)
        new_values.append(converted_value)
        if str(value) in mapping:
            new_mapping[str(converted_value)] = mapping[str(value)]
    specification['response_values'] = new_values
    specification['response_mapping'] = new_mapping


def prepare_type_conversion(original, converted, target, settings):
    """Migrate affected settings and reject semantic changes before GUI confirmation/commit.

    :param original: Complete source column, retained read-only by the owned worker.
    :param converted: Validated converted column with identical row order and missing locations.
    :param target: Numeric, Text, or Boolean conversion target.
    :param settings: Snapshot containing rows, columns, filters, and analysis targets.
    :return: Independent setting updates and warnings for the confirmation dialog.
    """
    name = original.name
    filters, warnings = [], []
    before, after = original, converted
    literal_target = target
    for rule in settings['filters']:
        variable, expression = split_filter_rule(rule)
        if variable != name:
            filters.append(rule)
            continue
        if is_range_expression(expression):
            _check_numeric_meaning(before, after, f'filter {rule!r}')
            # Range filtering coerces this working column before subsequent checklist/model use.
            before = pd.to_numeric(before, errors='coerce')
            after = pd.to_numeric(after, errors='coerce')
            literal_target = 'Numeric'
        elif expression == 'Pooling CDF':
            _check_numeric_meaning(before, after, f'filter {rule!r}')
            if not pd.api.types.is_numeric_dtype(after.dtype):
                raise ValueError(
                    f'Filter {rule!r} requires numeric storage. This conversion would make '
                    'the existing Pooling CDF setting incompatible. No data were changed.')
        else:
            expression = _migrate_checklist(
                before, after, parse_checklist_values(expression), literal_target, rule)
            rule = f'{variable}:{expression}'
        filters.append(rule)

    targets = deepcopy(settings['targets'])
    for index, entry in enumerate(targets):
        variable, operation, specification = split_target(entry)
        if specification is not None:
            if specification.get('response_variable') == name:
                _migrate_response_mapping(before, after, specification, literal_target)
            if name in (specification.get('rt_variable'), specification.get('accuracy_variable')):
                _check_numeric_meaning(original, converted, f'{variable}@{operation}')
            targets[index] = {'model_specification': specification}
        elif variable == name and operation not in ('Count', 'Mode'):
            _check_numeric_meaning(original, converted, f'{variable}@{operation}')

    if name in settings['rows'] + settings['columns']:
        old_count, new_count = original.nunique(dropna=True), converted.nunique(dropna=True)
        old_codes, _ = pd.factorize(original)
        new_codes, _ = pd.factorize(converted)
        present = old_codes >= 0
        pair_count = pd.MultiIndex.from_arrays(
            [old_codes[present], new_codes[present]]).nunique()
        merge, split = pair_count > new_count, pair_count > old_count
        if merge or split:
            change = 'merge and split' if merge and split else 'merge' if merge else 'split'
            warnings.append(
                f'Grouping variable {name!r}: {old_count} distinct values become {new_count}. '
                f'This will {change} grouping levels and may change group-based filtering '
                'and subsequent results. Existing results will not be changed.')
    return TypeConversionPlan(filters, targets, warnings)
