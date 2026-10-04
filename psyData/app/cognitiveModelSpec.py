"""Shared model specifications for PsySummary cognitive RT models."""

from copy import deepcopy
import math
import time
from decimal import Decimal
from numbers import Real

import numpy as np
import pandas as pd


RATCLIFF_MODEL = 'Ratcliff Diffusion Model'
LBA_MODEL = 'Linear Ballistic Accumulator'
RDM_MODEL = 'Racing Diffusion Model'

COGNITIVE_MODEL_NAMES = (RATCLIFF_MODEL, LBA_MODEL, RDM_MODEL)
MODEL_SPEC_SCHEMA_VERSION = 3
RTDISTS_REFERENCE_VERSION = '0.12-0'
ACCURACY_CODING = 'accuracy'
RESPONSE_CODING = 'response'
BOUNDARY_CODING_LABELS = {
    ACCURACY_CODING: 'Accuracy Coding (Correct / Error)',
    RESPONSE_CODING: 'Response Coding (Choice / Direction)',
}

MODEL_REFERENCES = {
    RATCLIFF_MODEL: (
        'Ratcliff, R., & McKoon, G. (2008). The diffusion decision model: '
        'Theory and data for two-choice decision tasks. Neural Computation, 20(4), 873-922. '
        'https://doi.org/10.1162/neco.2008.12-06-420'
    ),
    LBA_MODEL: (
        'Brown, S. D., & Heathcote, A. (2008). The simplest complete model of choice '
        'response time: Linear ballistic accumulation. Cognitive Psychology, 57(3), 153-178. '
        'https://doi.org/10.1016/j.cogpsych.2007.12.002'
    ),
    RDM_MODEL: (
        'Tillman, G., Van Zandt, T., & Logan, G. D. (2020). Sequential sampling models '
        'without random between-trial variability: The racing diffusion model of speeded '
        'decision making. Psychonomic Bulletin & Review, 27(5), 911-936. '
        'https://doi.org/10.3758/s13423-020-01719-6'
    ),
}

RTDISTS_REFERENCE = (
    'Singmann, H., Brown, S., Gretton, M., Heathcote, A., & Fernandez, K. '
    f'rtdists: Response Time Distributions. R package version {RTDISTS_REFERENCE_VERSION}. '
    'https://github.com/rtdists/rtdists/'
)


def cognitive_model_reference_text(model):
    """Return publication and implementation references for one fitted model."""
    return (
        f'Model reference: {MODEL_REFERENCES[model]}\n'
        f'Numerical reference: {RTDISTS_REFERENCE}\n'
        'Implementation: independent Python implementation benchmarked against rtdists; '
        'R and rtdists are not invoked at runtime. Optimizer: constrained SLSQP.'
    )


def parameter_tooltip(model, parameter_name):
    """Return a precise description for one cognitive-model parameter."""
    shared_accumulator = {
        'A': ('Start-point range. On each trial, starting evidence is sampled uniformly '
              'from 0 to A. A must be non-negative and smaller than b.'),
        'b': ('Response threshold. An accumulator wins when its evidence reaches b. '
              'The threshold must be greater than A.'),
        't0': ('Lower bound of non-decision time in seconds, covering encoding and response '
               'execution. When st0 > 0, mean non-decision time is t0 + st0/2.'),
        'st0': ('Across-trial range of non-decision time in seconds. Non-decision time is '
                'uniformly distributed from t0 to t0 + st0.'),
    }
    if model in (LBA_MODEL, RDM_MODEL) and parameter_name in shared_accumulator:
        return shared_accumulator[parameter_name]
    if model == RATCLIFF_MODEL:
        return {
            'a': ('Threshold separation. It is the distance between the lower and upper '
                  'decision boundaries; larger values indicate more cautious responding.'),
            'v': ('Mean evidence drift toward the correct response. During fitting, its direction '
                  'is oriented from Accuracy and the configured physical response boundaries; '
                  'positive values favor the correct response.'),
            't0': ('Lower bound of non-decision time in seconds, covering stimulus encoding and '
                   'response execution. The mean before response-side d offsets is t0 + st0/2.'),
            'zr': ('Relative starting position of decision evidence: zr = z/a, strictly between '
                   '0 and 1. zr = 0.5 starts halfway between the boundaries; smaller values '
                   'favor the configured lower boundary and larger values favor the upper. '
                   'Fixed zr = 0.5 keeps the starting position centered even when a is Free: '
                   'the absolute position z = a * zr changes with a. Free zr estimates starting '
                   'bias. Centered starting evidence alone does not imply equal response '
                   'probabilities. Accuracy Coding mirrors zr to 1-zr when the correct physical '
                   'response is lower. This evidence starting position is distinct from the '
                   'optimizer Value/Start, which initializes parameter estimation.'),
            'd': ('Physical lower/upper response-execution-time difference in seconds. Positive '
                  'values make the configured upper response faster; Accuracy Coding reverses its '
                  'sign when the correct physical response is lower.'),
            'sz': ('Across-trial range of starting-point variability. Trial starting points '
                   'are uniformly distributed around z = a * zr with total absolute width sz. '
                   'Unlike zr, sz is not relative to a; a * zr +/- sz/2 must stay between 0 and a.'),
            'sv': ('Across-trial standard deviation of drift rate. Trial drift rates follow '
                   'a normal distribution with mean v and standard deviation sv.'),
            'st0': ('Across-trial range of non-decision time. Trial values are uniformly '
                    'distributed from t0 to t0 + st0.'),
            's': ('Within-trial diffusion-noise standard deviation. This sets the model scale '
                  'and must remain Fixed for parameter identifiability.'),
        }.get(parameter_name, '')
    if model == LBA_MODEL:
        if parameter_name.startswith('mean_v['):
            accumulator = parameter_name.removeprefix('mean_v[').removesuffix(']')
            return (f'Mean drift rate of accumulator {accumulator}. Drift rates are sampled '
                    'from the positive-truncated normal distribution on each trial.')
        if parameter_name.startswith('sd_v['):
            accumulator = parameter_name.removeprefix('sd_v[').removesuffix(']')
            return (f'Across-trial drift-rate standard deviation of accumulator {accumulator}. '
                    'At least one sd_v parameter must remain Fixed to set the LBA scale.')
    if model == RDM_MODEL:
        if parameter_name == 's':
            return ('Within-trial diffusion-noise standard deviation shared by the racing '
                    'accumulators. It must remain Fixed for parameter identifiability.')
        if parameter_name.startswith('v['):
            accumulator = parameter_name.removeprefix('v[').removesuffix(']')
            return (f'Mean evidence-accumulation rate of racing diffusion accumulator '
                    f'{accumulator}. RDM drift rates must be non-negative.')
    return ''


def _serializable_response_value(value):
    """Normalize a response category for presets and exported Python scripts."""
    if hasattr(value, 'item'):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def is_cognitive_model(operation):
    """Return whether an operation names a supported cognitive RT model."""
    return operation in COGNITIVE_MODEL_NAMES


def split_target(target):
    """Return the RT variable, operation, and optional structured specification."""
    if isinstance(target, dict):
        specification = deepcopy(target.get('model_specification') or target)
        rt_variable = specification.get('rt_variable', '')
        operation = specification.get('model', '')
        return rt_variable, operation, specification
    rt_variable, operation = str(target).split('@', 1)
    return rt_variable, operation, None


def target_display_text(target):
    """Return the compact text displayed for a target entry."""
    rt_variable, operation, _specification = split_target(target)
    return f'{rt_variable}@{operation}'


def resolve_analysis_seeds(targets):
    """Freeze time-based seeds on analysis copies, leaving saved GUI drafts unchanged."""
    resolved = []
    for target in targets:
        if isinstance(target, dict):
            target = deepcopy(target)
            specification = target.get('model_specification') or target
            optimizer = specification.setdefault('optimizer', {})
            if optimizer.get('seed') is None:
                optimizer['seed'] = int(time.time_ns() % (2 ** 32))
        resolved.append(target)
    return resolved


def parameter_spec(name, mode, value, lower, upper):
    """Build one serializable model-parameter specification."""
    return {
        'name': name,
        'mode': mode,
        'value': float(value),
        'lower': float(lower),
        'upper': float(upper),
    }


def default_parameters(model, response_values, minimum_rt=0.2):
    """Return identifiable defaults for a model and its response alternatives."""
    minimum_rt = float(minimum_rt)
    minimum_rt = minimum_rt if math.isfinite(minimum_rt) and minimum_rt > 0 else 0.2
    # A display fallback only; automatic starts are regenerated for each fit group.
    t0_start = min(0.1, minimum_rt / 2.0)
    t0_upper = 0.8
    if model == RATCLIFF_MODEL:
        parameters = [
            parameter_spec('a', 'free', 1.0, 0.05, 5.0),
            parameter_spec('v', 'free', 1.0, -10.0, 10.0),
            parameter_spec('t0', 'free', t0_start, 0.0, t0_upper),
            parameter_spec('zr', 'fixed', 0.5, 0.001, 0.999),
            parameter_spec('d', 'fixed', 0.0, -0.5, 0.5),
            parameter_spec('sz', 'fixed', 0.0, 0.0, 4.999),
            parameter_spec('sv', 'fixed', 0.0, 0.0, 5.0),
            parameter_spec('st0', 'fixed', 0.0, 0.0, 0.6),
            parameter_spec('s', 'fixed', 1.0, 1e-3, 10.0),
        ]
        return _mark_automatic_parameters(parameters)

    parameters = [
        parameter_spec('A', 'free', 0.5, 0.0, 5.0),
        parameter_spec('b', 'free', 1.0, 0.01, 10.0),
        parameter_spec('t0', 'free', t0_start, 0.0, t0_upper),
        parameter_spec('st0', 'fixed', 0.0, 0.0, 0.6),
    ]
    if model == LBA_MODEL:
        for index, _response in enumerate(response_values):
            parameters.append(parameter_spec(f'mean_v[{index + 1}]', 'free', 1.5 - 0.2 * index, 0.01, 10.0))
            parameters.append(parameter_spec(
                f'sd_v[{index + 1}]', 'fixed' if index == 0 else 'free', 1.0, 0.01, 5.0))
        return _mark_automatic_parameters(parameters)

    parameters.append(parameter_spec('s', 'fixed', 1.0, 1e-3, 10.0))
    for index, _response in enumerate(response_values):
        parameters.append(parameter_spec(f'v[{index + 1}]', 'free', 2.0 - 0.2 * index, 0.001, 10.0))
    return _mark_automatic_parameters(parameters)


def _mark_automatic_parameters(parameters):
    """Mark new defaults without assigning automatic ownership to legacy settings."""
    for parameter in parameters:
        parameter['start_source'] = 'auto'
        parameter['auto_value'] = parameter['value']
        if parameter['name'] in ('t0', 'st0'):
            parameter['upper_source'] = 'auto'
    return parameters


def is_automatic_start(parameter):
    """Recognize automatic values while protecting programmatic edits of saved rows."""
    return (parameter.get('start_source') == 'auto'
            and parameter.get('auto_value', parameter.get('value')) == parameter.get('value'))


def update_default_time_bounds(parameters):
    """Update only automatically owned time bounds after a fixed/free change."""
    by_name = {parameter['name']: parameter for parameter in parameters}
    both_free = all(by_name.get(name, {}).get('mode') == 'free' for name in ('t0', 'st0'))
    for name, upper in (('t0', 0.5 if both_free else 0.8), ('st0', 0.6)):
        parameter = by_name.get(name)
        if parameter is not None and parameter.get('upper_source') == 'auto':
            parameter['upper'] = upper


def random_model_start(model, parameters, minimum_rt, rng, randomize_manual=False):
    """Draw rtdists-inspired starts inside user bounds without evaluating likelihoods.

    :param model: Cognitive model name.
    :param parameters: Saved parameter rows; missing ownership means a manual start.
    :param minimum_rt: Smallest positive fitted RT in seconds.
    :param rng: Run- or dialog-owned NumPy random generator.
    :param randomize_manual: Whether additional starts may also vary manual free values.
    :return: Parameter values keyed by name.
    """
    values = {row['name']: float(row['value']) for row in parameters}
    by_name = {row['name']: row for row in parameters}

    def draw(name, low, high, normal_sd=None):
        row = by_name[name]
        if row['mode'] != 'free' or (not randomize_manual and not is_automatic_start(row)):
            return values[name]
        lower, upper = float(row['lower']), float(row['upper'])
        left, right = max(lower, low), min(upper, high)
        if left >= right:
            # Explicit bounds can intentionally lie outside the reference start interval.
            left, right = lower, upper
        if normal_sd is not None:
            for _ in range(32):
                sampled = float(rng.normal(0.0, normal_sd))
                if left <= sampled <= right:
                    values[name] = sampled
                    return sampled
        values[name] = float(rng.uniform(left, right))
        return values[name]

    if model == RATCLIFF_MODEL:
        draw('a', 0.5, 3.0)
        draw('v', -np.inf, np.inf, normal_sd=1.0)
        draw('zr', 0.4, 0.6)
        draw('sz', 0.0, 0.5)
        draw('sv', 0.0, 0.5)
        draw('d', -np.inf, np.inf, normal_sd=0.05)
        draw('t0', 0.0, min(0.5, minimum_rt))
    else:
        start = draw('A', 0.0, 1.0)
        draw('b', start, start + 1.0)
        draw('t0', 0.0, minimum_rt)
        for name in by_name:
            if name.startswith(('mean_v[', 'v[', 'sd_v[')):
                draw(name, 0.0, 1.0)
    # LBA/RDM examples do not fit st0. This extension uses the tutorial's
    # uniform range, shortened for very fast observations to aid time support.
    draw('st0', 0.0, 0.5 if model == RATCLIFF_MODEL else min(0.5, minimum_rt))
    return values


def initial_parameters_are_valid(model, values, minimum_rt, minimum_rt_by_response=None):
    """Check cheap joint constraints and time support, without calculating densities."""
    if not all(math.isfinite(value) for value in values.values()):
        return False
    if values['t0'] < 0 or values['st0'] < 0:
        return False
    if model == RATCLIFF_MODEL:
        if (values['a'] <= 0 or values['s'] <= 0 or values['sz'] < 0 or values['sv'] < 0
                or values['a'] * values['zr'] - values['sz'] / 2 <= 0
                or values['a'] * values['zr'] + values['sz'] / 2 >= values['a']
                or values['t0'] < abs(values['d']) / 2):
            return False
        if minimum_rt_by_response:
            support = min(rt - (values['d'] / 2 if response == 'lower' else -values['d'] / 2)
                          for response, rt in minimum_rt_by_response.items())
        else:
            support = minimum_rt - abs(values['d']) / 2
        points = 9
    else:
        if values['A'] < 0 or values['b'] <= values['A']:
            return False
        if model == LBA_MODEL and any(
                value <= 0 for name, value in values.items() if name.startswith('sd_v[')):
            return False
        if model == RDM_MODEL and (values['s'] <= 0 or any(
                value < 0 for name, value in values.items() if name.startswith('v['))):
            return False
        support, points = minimum_rt, 7
    # Match the earliest time node used by the existing fixed-node quadrature.
    # Large st0 can otherwise give zero numerical density even when continuous support exists.
    earliest = 0.5 * (np.polynomial.legendre.leggauss(points)[0][0] + 1.0)
    return values['t0'] + earliest * values['st0'] < support


def short_rt_summary(specification, dataframe, group_vars=()):
    """Describe fitted RTs below 50 ms without changing data or rejecting settings."""
    rt_name = specification.get('rt_variable')
    response_name = specification.get('response_variable')
    if rt_name not in dataframe.columns or response_name not in dataframe.columns:
        return ''
    rt = pd.to_numeric(dataframe[rt_name], errors='coerce').to_numpy(dtype=float)
    if specification.get('rt_unit') == 'milliseconds':
        rt = rt / 1000.0
    configured = model_response_mapping(specification)
    mapped = dataframe[response_name].map(
        lambda value: configured.get(response_value_token(value))).notna().to_numpy()
    short = np.isfinite(rt) & (rt > 0) & (rt < 0.05) & mapped
    if not np.any(short):
        return ''
    description = (f"{specification.get('model')}, {rt_name}: {int(short.sum())} RT(s) below "
                   f"50 ms; minimum {float(rt[short].min()) * 1000:.6g} ms.")
    if group_vars:
        groups = dataframe.loc[short, list(group_vars)].drop_duplicates()
        labels = ['; '.join(f'{name}={value}' for name, value in zip(group_vars, row))
                  for row in groups.head(8).itertuples(index=False, name=None)]
        description += '\nGroups: ' + ' | '.join(labels)
        if len(groups) > 8:
            description += f' | and {len(groups) - 8} more'
    return description


def make_model_specification(model, rt_variable, response_variable, response_values,
                             minimum_rt=0.2, accuracy_variable=None, rt_unit='seconds',
                             boundary_coding=ACCURACY_CODING):
    """Build a complete default specification for one cognitive RT model."""
    values = [_serializable_response_value(value) for value in response_values]
    return {
        'schema_version': MODEL_SPEC_SCHEMA_VERSION,
        'model': model,
        'rt_variable': rt_variable,
        'response_variable': response_variable,
        'accuracy_variable': accuracy_variable or None,
        'boundary_coding': (
            boundary_coding if model == RATCLIFF_MODEL else RESPONSE_CODING),
        'rt_unit': rt_unit,
        'response_values': values,
        'response_mapping': {
            str(value): ('lower' if model == RATCLIFF_MODEL and index == 0 else
                         'upper' if model == RATCLIFF_MODEL else index + 1)
            for index, value in enumerate(values)
        },
        'parameters': default_parameters(model, values, minimum_rt),
        'optimizer': {
            'method': 'SLSQP',
            'starts': 3,
            'seed': None,
            'max_iterations': 1000,
        },
    }


def validate_model_specification(specification, available_variables=None):
    """Validate a cognitive-model specification and return it unchanged."""
    model = specification.get('model')
    if model not in COGNITIVE_MODEL_NAMES:
        raise ValueError(f'Unsupported cognitive RT model: {model}.')
    rt_variable = specification.get('rt_variable')
    response_variable = specification.get('response_variable')
    if not rt_variable or not response_variable:
        raise ValueError('RT Variable and Response Variable are required.')
    if rt_variable == response_variable:
        raise ValueError('RT Variable and Response Variable must be different variables.')
    accuracy_variable = specification.get('accuracy_variable')
    boundary_coding = specification.get('boundary_coding', RESPONSE_CODING)
    if model == RATCLIFF_MODEL and boundary_coding not in BOUNDARY_CODING_LABELS:
        raise ValueError(f'Unsupported Ratcliff boundary coding: {boundary_coding}.')
    if model == RATCLIFF_MODEL and not accuracy_variable:
        raise ValueError('Ratcliff Diffusion Model requires an Accuracy Variable containing 0 or 1.')
    if accuracy_variable and accuracy_variable in (rt_variable, response_variable):
        raise ValueError('Accuracy Variable must differ from RT and Response variables.')
    if available_variables is not None:
        missing = [name for name in (rt_variable, response_variable,
                                     specification.get('accuracy_variable'))
                   if name and name not in available_variables]
        if missing:
            raise ValueError(f"Model variable(s) are unavailable: {', '.join(missing)}.")
    response_values = specification.get('response_values') or []
    if model == RATCLIFF_MODEL and len(response_values) != 2:
        raise ValueError(
            f"Ratcliff Diffusion Model requires exactly two response values (2 distinct values). "
            f"Response Variable '{response_variable}' has {len(response_values)} configured values: "
            f"{response_values}. Use Define Filters to retain the intended two responses, "
            'then reopen Model Settings and check their lower/upper mapping.')
    if model in (LBA_MODEL, RDM_MODEL) and len(response_values) < 2:
        raise ValueError(f'{model} requires at least two response values.')
    response_mapping = specification.get('response_mapping') or {}
    if len({response_value_token(value) for value in response_values}) != len(response_values):
        raise ValueError('Configured response values must be distinct.')
    if len(response_mapping) != len(response_values) or set(response_mapping) != {str(value) for value in response_values}:
        raise ValueError('Every observed response value must have a model-response mapping.')
    mapped_values = list(response_mapping.values())
    if model == RATCLIFF_MODEL and sorted(mapped_values) != ['lower', 'upper']:
        raise ValueError('Ratcliff responses must map once each to lower and upper.')
    if model in (LBA_MODEL, RDM_MODEL) and sorted(mapped_values) != list(range(1, len(response_values) + 1)):
        raise ValueError('Each accumulator index must be mapped exactly once.')

    parameters = specification.get('parameters') or []
    if not parameters:
        raise ValueError('At least one model parameter is required.')
    names = set()
    for parameter in parameters:
        name = parameter.get('name')
        if not name or name in names:
            raise ValueError(f'Invalid or duplicate model parameter: {name}.')
        names.add(name)
        mode = parameter.get('mode')
        if mode not in ('fixed', 'free'):
            raise ValueError(f"Parameter '{name}' must be Fixed or Free.")
        try:
            lower = float(parameter.get('lower'))
            upper = float(parameter.get('upper'))
            value = float(parameter.get('value'))
        except (TypeError, ValueError):
            raise ValueError(f"Parameter '{name}': Value/Start, Lower and Upper must be finite numbers.") from None
        if not all(math.isfinite(number) for number in (lower, upper, value)):
            raise ValueError(f"Parameter '{name}': Value/Start, Lower and Upper must be finite numbers.")
        if lower >= upper or not lower <= value <= upper:
            raise ValueError(f"Parameter '{name}' must satisfy Lower <= Value/Start <= Upper.")

    by_name = {parameter['name']: parameter for parameter in parameters}
    if model == RATCLIFF_MODEL and ('zr' not in by_name or 'z' in by_name):
        raise ValueError('Ratcliff settings require relative starting position zr; recreate Model Settings.')
    if model in (RATCLIFF_MODEL, RDM_MODEL) and by_name.get('s', {}).get('mode') != 'fixed':
        raise ValueError("The diffusion scale parameter 's' must be Fixed for identifiability.")
    if model == LBA_MODEL:
        scale_parameters = [parameter for parameter in parameters if parameter['name'].startswith('sd_v[')]
        if not any(parameter['mode'] == 'fixed' for parameter in scale_parameters):
            raise ValueError('At least one LBA sd_v parameter must be Fixed for identifiability.')
    values = {name: float(parameter['value']) for name, parameter in by_name.items()}
    if model == RATCLIFF_MODEL:
        if not (values['a'] > 0 and values['s'] > 0 and values['sz'] >= 0
                and values['sv'] >= 0 and values['st0'] >= 0):
            raise ValueError('Diffusion scale and variability parameters are outside their valid ranges.')
        if (values['a'] * values['zr'] - values['sz'] / 2.0 <= 0
                or values['a'] * values['zr'] + values['sz'] / 2.0 >= values['a']):
            raise ValueError('a × zr ± sz/2 must remain strictly between 0 and a (0 < zr < 1).')
        if values['t0'] - abs(values['d']) / 2.0 < 0:
            raise ValueError('t0 must be at least abs(d)/2.')
    else:
        if values['A'] < 0 or values['b'] <= values['A']:
            raise ValueError('Accumulator threshold b must be greater than start range A >= 0.')
        if values['t0'] < 0 or values['st0'] < 0:
            raise ValueError('t0 and st0 cannot be negative.')
        if model == LBA_MODEL and any(
                value <= 0 for name, value in values.items() if name.startswith('sd_v[')):
            raise ValueError('LBA drift-rate standard deviations must be greater than zero.')
        if model == RDM_MODEL and values['s'] <= 0:
            raise ValueError('RDM diffusion scale s must be greater than zero.')
        if model == RDM_MODEL and any(
                value < 0 for name, value in values.items() if name.startswith('v[')):
            raise ValueError('RDM drift rates cannot be negative.')
    return specification


def response_value_token(value):
    """Compare numeric responses by value without converting text responses to numbers."""
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
        return ('number', Decimal(str(value)))
    return ('text', str(value)) if isinstance(value, str) else ('other', str(value))


def model_response_mapping(specification):
    """Resolve saved string keys through their typed, configured response values."""
    mapping = specification.get('response_mapping', {})
    return {response_value_token(value): mapping.get(str(value))
            for value in specification.get('response_values', [])}


def validate_model_data(specification, dataframe, group_vars=()):
    """Validate a saved model against current filtered data before any fitting starts."""
    model = specification.get('model')
    variable = specification.get('response_variable')
    if variable in dataframe.columns:
        observed = list(pd.unique(dataframe[variable].dropna()))
        if model == RATCLIFF_MODEL and len(observed) != 2:
            raise ValueError(
                f"{model} requires exactly 2 distinct response values. Response Variable "
                f"'{variable}' contains {len(observed)} values after the current filters: {observed}. "
                'Use Define Filters to retain the intended two responses, then reopen Model Settings '
                'and map one to lower and the other to upper.')
    validate_model_specification(specification, dataframe.columns)
    configured = {response_value_token(value) for value in specification['response_values']}
    actual = {response_value_token(value) for value in observed}
    if actual != configured:
        raise ValueError(
            f"Response Variable '{variable}' no longer matches the saved response mapping. "
            f"Current filtered values: {observed}; saved values: {specification['response_values']}. "
            'Check Define Filters and reopen Model Settings to update the mapping.')
    if dataframe.empty:
        raise ValueError('No rows remain after applying the current filters.')
    groups = dataframe.groupby(list(group_vars), dropna=False, sort=False, observed=True) if group_vars else [((), dataframe)]
    for key, frame in groups:
        keys = key if isinstance(key, tuple) else (key,)
        label = ', '.join(f'{name}={value}' for name, value in zip(group_vars, keys)) or 'Overall'
        rt = pd.to_numeric(frame[specification['rt_variable']], errors='coerce')
        valid = np.isfinite(rt) & frame[variable].notna()
        present = {response_value_token(value) for value in frame.loc[valid, variable]}
        if not present:
            raise ValueError(
                f"Group [{label}]: no valid RT/response pairs remain after filtering. "
                'Check filters, grouping and Model Settings.')
        accuracy_variable = specification.get('accuracy_variable')
        if accuracy_variable:
            accuracy = pd.to_numeric(frame.loc[valid, accuracy_variable], errors='coerce')
            if accuracy.empty or not np.isfinite(accuracy).all() or not accuracy.isin([0, 1]).all():
                raise ValueError(
                    f"Group [{label}]: Accuracy Variable '{accuracy_variable}' must contain "
                    'only 0 (error) and 1 (correct), with no missing values, for every retained '
                    'RT/response pair. Recode or exclude invalid rows in Filter Data before running.')


class ValidatedModelData:
    """Keep a run-local validation receipt for one read-only frame and model configuration."""

    def __init__(self, specification, dataframe, group_vars=()):
        validate_model_data(specification, dataframe, group_vars)
        self._dataframe = dataframe
        self._shape = dataframe.shape
        self._columns = tuple(dataframe.columns)
        self._specification = deepcopy(specification)
        self._group_vars = tuple(group_vars)

    def matches(self, specification, dataframe, group_vars=()):
        """Reuse validation only for the exact prepared frame and unchanged configuration."""
        return (dataframe is self._dataframe and dataframe.shape == self._shape
                and tuple(dataframe.columns) == self._columns
                and tuple(group_vars) == self._group_vars
                and specification == self._specification)


def model_result_parameters(specification):
    """Return the one shared parameter set reported for either boundary coding."""
    return specification['parameters']
