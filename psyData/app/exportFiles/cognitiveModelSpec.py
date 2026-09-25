"""Shared model specifications for PsySummary cognitive RT models."""

from copy import deepcopy


RATCLIFF_MODEL = 'Ratcliff Diffusion Model'
LBA_MODEL = 'Linear Ballistic Accumulator'
RDM_MODEL = 'Racing Diffusion Model'

COGNITIVE_MODEL_NAMES = (RATCLIFF_MODEL, LBA_MODEL, RDM_MODEL)
MODEL_SPEC_SCHEMA_VERSION = 1
RTDISTS_REFERENCE_VERSION = '0.12-0'

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
        't0': ('Non-decision time in seconds. It represents encoding and response execution '
               'outside the evidence-accumulation process.'),
        'st0': ('Across-trial range of non-decision time in seconds. Non-decision time is '
                'uniformly distributed from t0 to t0 + st0.'),
    }
    if model in (LBA_MODEL, RDM_MODEL) and parameter_name in shared_accumulator:
        return shared_accumulator[parameter_name]
    if model == RATCLIFF_MODEL:
        return {
            'a': ('Threshold separation. It is the distance between the lower and upper '
                  'decision boundaries; larger values indicate more cautious responding.'),
            'v': ('Mean drift rate. Positive values favor the upper response boundary and '
                  'negative values favor the lower response boundary.'),
            't0': ('Lower bound of non-decision time in seconds, covering processes such as '
                   'stimulus encoding and response execution.'),
            'z': ('Absolute starting point between 0 and a. Values away from a / 2 represent '
                  'an initial bias toward one response boundary.'),
            'd': ('Difference in response-execution time between boundaries, in seconds. '
                  'Positive values make upper-boundary execution faster than lower-boundary execution.'),
            'sz': ('Across-trial range of starting-point variability. Trial starting points '
                   'are uniformly distributed around z with total width sz.'),
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
    minimum_rt = max(float(minimum_rt), 1e-3)
    t0_start = min(0.3, minimum_rt * 0.8)
    t0_upper = max(minimum_rt * 0.999, t0_start + 1e-3)
    if model == RATCLIFF_MODEL:
        return [
            parameter_spec('a', 'free', 1.0, 0.05, 5.0),
            parameter_spec('v', 'free', 1.0, -10.0, 10.0),
            parameter_spec('t0', 'free', t0_start, 0.0, t0_upper),
            parameter_spec('z', 'fixed', 0.5, 0.001, 4.999),
            parameter_spec('d', 'fixed', 0.0, -0.5, 0.5),
            parameter_spec('sz', 'fixed', 0.0, 0.0, 4.999),
            parameter_spec('sv', 'fixed', 0.0, 0.0, 5.0),
            parameter_spec('st0', 'fixed', 0.0, 0.0, t0_upper),
            parameter_spec('s', 'fixed', 1.0, 1e-3, 10.0),
        ]

    parameters = [
        parameter_spec('A', 'free', 0.5, 0.0, 5.0),
        parameter_spec('b', 'free', 1.0, 0.01, 10.0),
        parameter_spec('t0', 'free', t0_start, 0.0, t0_upper),
        parameter_spec('st0', 'fixed', 0.0, 0.0, t0_upper),
    ]
    if model == LBA_MODEL:
        for index, _response in enumerate(response_values):
            parameters.append(parameter_spec(f'mean_v[{index + 1}]', 'free', 1.5 - 0.2 * index, 0.01, 10.0))
            parameters.append(parameter_spec(
                f'sd_v[{index + 1}]', 'fixed' if index == 0 else 'free', 1.0, 0.01, 5.0))
        return parameters

    parameters.append(parameter_spec('s', 'fixed', 1.0, 1e-3, 10.0))
    for index, _response in enumerate(response_values):
        parameters.append(parameter_spec(f'v[{index + 1}]', 'free', 2.0 - 0.2 * index, 0.001, 10.0))
    return parameters


def make_model_specification(model, rt_variable, response_variable, response_values,
                             minimum_rt=0.2, accuracy_variable=None, rt_unit='seconds'):
    """Build a complete default specification for one cognitive RT model."""
    values = [_serializable_response_value(value) for value in response_values]
    return {
        'schema_version': MODEL_SPEC_SCHEMA_VERSION,
        'model': model,
        'rt_variable': rt_variable,
        'response_variable': response_variable,
        'accuracy_variable': accuracy_variable or None,
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
    if accuracy_variable and accuracy_variable in (rt_variable, response_variable):
        raise ValueError('Accuracy/Correct Variable must differ from RT and Response variables.')
    if available_variables is not None:
        missing = [name for name in (rt_variable, response_variable,
                                     specification.get('accuracy_variable'))
                   if name and name not in available_variables]
        if missing:
            raise ValueError(f"Model variable(s) are unavailable: {', '.join(missing)}.")
    response_values = specification.get('response_values') or []
    if model == RATCLIFF_MODEL and len(response_values) != 2:
        raise ValueError('Ratcliff Diffusion Model requires exactly two response values.')
    if model in (LBA_MODEL, RDM_MODEL) and len(response_values) < 2:
        raise ValueError(f'{model} requires at least two response values.')
    response_mapping = specification.get('response_mapping') or {}
    if set(response_mapping) != {str(value) for value in response_values}:
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
        lower = float(parameter.get('lower'))
        upper = float(parameter.get('upper'))
        value = float(parameter.get('value'))
        if lower >= upper or not lower <= value <= upper:
            raise ValueError(f"Parameter '{name}' must satisfy Lower <= Value/Start <= Upper.")

    by_name = {parameter['name']: parameter for parameter in parameters}
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
        if values['z'] - values['sz'] / 2.0 <= 0 or values['z'] + values['sz'] / 2.0 >= values['a']:
            raise ValueError('z ± sz/2 must remain strictly between 0 and a.')
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
