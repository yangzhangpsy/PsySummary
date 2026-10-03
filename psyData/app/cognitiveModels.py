"""Numerical distributions and fitting for cognitive response-time models."""

import math
import time

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import OptimizeResult, brentq, minimize
from scipy.special import ndtr

try:
    from app.fitCancellation import raise_if_fit_cancelled
except ImportError:  # Standalone PsySummary analysis-script export.
    def raise_if_fit_cancelled(cancel_check):
        """Honor an optional callback without requiring the GUI cancellation module."""
        if cancel_check is not None and cancel_check():
            raise RuntimeError('Model fitting was cancelled by the caller.')

try:
    from app.cognitiveModelSpec import (
        ACCURACY_CODING, LBA_MODEL, RATCLIFF_MODEL, RESPONSE_CODING, RDM_MODEL,
        model_response_mapping, response_value_token, validate_model_data,
        random_model_start, initial_parameters_are_valid,
        is_automatic_start,
    )
except ImportError:  # Standalone PsySummary analysis-script export.
    from cognitiveModelSpec import (
        ACCURACY_CODING, LBA_MODEL, RATCLIFF_MODEL, RESPONSE_CODING, RDM_MODEL,
        model_response_mapping, response_value_token, validate_model_data,
        random_model_start, initial_parameters_are_valid,
        is_automatic_start,
    )


_TINY = np.finfo(float).tiny
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _normal_pdf(value):
    """Return the standard-normal density for scalar or array input."""
    value = np.asarray(value, dtype=float)
    return np.exp(-0.5 * value * value) / _SQRT_2PI


def _as_1d(values):
    """Return input as a one-dimensional float array and remember scalar input."""
    array = np.asarray(values, dtype=float)
    return np.atleast_1d(array), array.ndim == 0


def _restore_shape(values, scalar):
    """Restore scalar output when the corresponding input was scalar."""
    return float(values[0]) if scalar else values


def _wfpt_base_density(time, relative_start):
    """Evaluate the unit Wiener first-passage density at the lower boundary."""
    if time <= 0 or not 0 < relative_start < 1:
        return 0.0
    if time < 0.25:
        radius = 4
        indices = np.arange(-radius, radius + 1, dtype=float)
        offsets = relative_start + 2.0 * indices
        terms = offsets * np.exp(-(offsets * offsets) / (2.0 * time))
        return max(float(np.sum(terms) / math.sqrt(2.0 * math.pi * time ** 3)), 0.0)
    radius = max(5, int(math.ceil(math.sqrt(40.0 / (math.pi * math.pi * time)))))
    indices = np.arange(1, radius + 1, dtype=float)
    terms = indices * np.sin(indices * math.pi * relative_start) * np.exp(
        -0.5 * indices * indices * math.pi * math.pi * time)
    return max(float(math.pi * np.sum(terms)), 0.0)


def _diffusion_no_variability(decision_time, boundary, a, v, z, s):
    """Evaluate a two-boundary Wiener density without across-trial variability."""
    if decision_time <= 0 or a <= 0 or s <= 0 or not 0 < z < a:
        return 0.0
    if boundary == 'upper':
        z = a - z
        v = -v
    scaled_time = decision_time * s * s / (a * a)
    relative_start = z / a
    factor = (s * s / (a * a)) * math.exp(
        -v * z / (s * s) - v * v * decision_time / (2.0 * s * s))
    return max(factor * _wfpt_base_density(scaled_time, relative_start), 0.0)


def diffusion_pdf(rt, response, a, v, t0, z=None, d=0.0, sz=0.0, sv=0.0,
                  st0=0.0, s=1.0, quadrature_points=9):
    """Return defective Ratcliff diffusion densities for lower/upper responses."""
    rt_values, scalar = _as_1d(rt)
    responses = np.broadcast_to(np.asarray(response, dtype=object), rt_values.shape)
    z = 0.5 * a if z is None else z
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    hermite_nodes, hermite_weights = np.polynomial.hermite.hermgauss(quadrature_points)
    z_values = np.array([z]) if sz <= 1e-12 else z + 0.5 * sz * nodes
    z_weights = np.array([1.0]) if sz <= 1e-12 else 0.5 * weights
    t_offsets = np.array([0.0]) if st0 <= 1e-12 else 0.5 * st0 * (nodes + 1.0)
    t_weights = np.array([1.0]) if st0 <= 1e-12 else 0.5 * weights
    v_values = np.array([v]) if sv <= 1e-12 else v + math.sqrt(2.0) * sv * hermite_nodes
    v_weights = np.array([1.0]) if sv <= 1e-12 else hermite_weights / math.sqrt(math.pi)
    output = np.zeros(rt_values.size, dtype=float)
    for index, (observed_rt, raw_boundary) in enumerate(zip(rt_values, responses)):
        boundary = str(raw_boundary).lower()
        if boundary not in ('lower', 'upper'):
            raise ValueError("Diffusion responses must be 'lower' or 'upper'.")
        boundary_t0 = t0 + (0.5 * d if boundary == 'lower' else -0.5 * d)
        total = 0.0
        for current_z, weight_z in zip(z_values, z_weights):
            for current_v, weight_v in zip(v_values, v_weights):
                for offset, weight_t in zip(t_offsets, t_weights):
                    total += weight_z * weight_v * weight_t * _diffusion_no_variability(
                        observed_rt - boundary_t0 - offset, boundary, a, current_v, current_z, s)
        output[index] = max(total, 0.0)
    return _restore_shape(output, scalar)


def _lba_single_pdf(time, A, b, mean_v, sd_v, positive_drift=True):
    """Return the first-passage density of one normal-drift LBA accumulator."""
    time = np.asarray(time, dtype=float)
    output = np.zeros_like(time)
    valid = (time > 0) & (A >= 0) & (b > A) & (sd_v > 0)
    if not np.any(valid):
        return output
    current = time[valid]
    if A < 1e-10:
        output[valid] = (b / (current * current)) * (
            _normal_pdf((b / current - mean_v) / sd_v) / sd_v)
    else:
        first = (b - A - current * mean_v) / (current * sd_v)
        second = (b - current * mean_v) / (current * sd_v)
        output[valid] = (
            mean_v * (ndtr(second) - ndtr(first))
            + sd_v * (_normal_pdf(first) - _normal_pdf(second))) / A
    if positive_drift:
        output[valid] /= max(float(ndtr(mean_v / sd_v)), 1e-12)
    return np.maximum(output, 0.0)


def _lba_single_cdf(time, A, b, mean_v, sd_v, positive_drift=True):
    """Return the first-passage CDF of one normal-drift LBA accumulator."""
    time = np.asarray(time, dtype=float)
    output = np.zeros_like(time)
    valid = (time > 0) & (A >= 0) & (b > A) & (sd_v > 0)
    if not np.any(valid):
        return output
    current = time[valid]
    if A < 1e-10:
        output[valid] = ndtr((mean_v - b / current) / sd_v)
    else:
        zs = current * sd_v
        center = b - current * mean_v
        lower = (center - A) / zs
        upper = center / zs
        output[valid] = 1.0 + (
            zs * (_normal_pdf(lower) - _normal_pdf(upper))
            + (center - A) * ndtr(lower) - center * ndtr(upper)) / A
    if positive_drift:
        output[valid] /= max(float(ndtr(mean_v / sd_v)), 1e-12)
    return np.clip(output, 0.0, 1.0)


def _race_pdf(decision_time, winner, single_pdf, single_cdf, accumulator_parameters):
    """Return a race density from independent accumulator PDF/CDF functions."""
    winner = int(winner)
    density = single_pdf(decision_time, *accumulator_parameters[winner])
    survival = 1.0
    for index, parameters in enumerate(accumulator_parameters):
        if index != winner:
            survival *= 1.0 - single_cdf(decision_time, *parameters)
    return np.maximum(density * survival, 0.0)


def _average_nondecision_pdf(rt, t0, st0, density_function):
    """Average a decision-time race density over uniform non-decision time."""
    if st0 <= 1e-12:
        return density_function(np.asarray(rt, dtype=float) - t0)
    nodes, weights = np.polynomial.legendre.leggauss(7)
    output = np.zeros_like(np.asarray(rt, dtype=float))
    for node, weight in zip(nodes, weights):
        output += 0.5 * weight * density_function(np.asarray(rt, dtype=float) - t0 - 0.5 * st0 * (node + 1))
    return output


def lba_pdf(rt, response, A, b, t0, mean_v, sd_v, st0=0.0, positive_drift=True):
    """Return defective normal-drift LBA race densities."""
    rt_values, scalar = _as_1d(rt)
    response_values = np.broadcast_to(np.asarray(response, dtype=int), rt_values.shape)
    mean_v = np.asarray(mean_v, dtype=float)
    sd_v = np.broadcast_to(np.asarray(sd_v, dtype=float), mean_v.shape)
    accumulator_parameters = [(A, b, mean, deviation, positive_drift)
                              for mean, deviation in zip(mean_v, sd_v)]
    output = np.zeros_like(rt_values)
    for winner in np.unique(response_values):
        winner_index = int(winner) - 1
        if not 0 <= winner_index < len(accumulator_parameters):
            raise ValueError('LBA response index exceeds the number of accumulators.')
        mask = response_values == winner
        output[mask] = _average_nondecision_pdf(
            rt_values[mask], t0, st0,
            lambda decision_time: _race_pdf(
                decision_time, winner_index, _lba_single_pdf, _lba_single_cdf, accumulator_parameters))
    return _restore_shape(output, scalar)


def _wald_single_pdf(time, A, b, v, s):
    """Return the first-passage density of one uniform-start Wald accumulator."""
    time = np.asarray(time, dtype=float)
    output = np.zeros_like(time)
    valid = (time > 0) & (A >= 0) & (b > A) & (v >= 0) & (s > 0)
    if not np.any(valid):
        return output
    nodes, weights = np.polynomial.legendre.leggauss(12)
    starts = np.array([0.0]) if A < 1e-10 else 0.5 * A * (nodes + 1.0)
    start_weights = np.array([1.0]) if A < 1e-10 else 0.5 * weights
    current = time[valid]
    total = np.zeros_like(current)
    for start, weight in zip(starts, start_weights):
        distance = b - start
        total += weight * distance / (s * np.sqrt(2.0 * math.pi * current ** 3)) * np.exp(
            -((distance - v * current) ** 2) / (2.0 * s * s * current))
    output[valid] = total
    return np.maximum(output, 0.0)


def _wald_single_cdf(time, A, b, v, s):
    """Return the first-passage CDF of one uniform-start Wald accumulator."""
    time = np.asarray(time, dtype=float)
    output = np.zeros_like(time)
    valid = (time > 0) & (A >= 0) & (b > A) & (v >= 0) & (s > 0)
    if not np.any(valid):
        return output
    nodes, weights = np.polynomial.legendre.leggauss(12)
    starts = np.array([0.0]) if A < 1e-10 else 0.5 * A * (nodes + 1.0)
    start_weights = np.array([1.0]) if A < 1e-10 else 0.5 * weights
    current = time[valid]
    root = s * np.sqrt(current)
    total = np.zeros_like(current)
    for start, weight in zip(starts, start_weights):
        distance = b - start
        first = ndtr((v * current - distance) / root)
        second = np.exp(np.minimum(2.0 * v * distance / (s * s), 700.0)) * ndtr(
            -(v * current + distance) / root)
        total += weight * (first + second)
    output[valid] = total
    return np.clip(output, 0.0, 1.0)


def rdm_pdf(rt, response, A, b, t0, v, s=1.0, st0=0.0):
    """Return defective Racing Diffusion Model race densities."""
    rt_values, scalar = _as_1d(rt)
    response_values = np.broadcast_to(np.asarray(response, dtype=int), rt_values.shape)
    v = np.asarray(v, dtype=float)
    accumulator_parameters = [(A, b, drift, s) for drift in v]
    output = np.zeros_like(rt_values)
    for winner in np.unique(response_values):
        winner_index = int(winner) - 1
        if not 0 <= winner_index < len(accumulator_parameters):
            raise ValueError('RDM response index exceeds the number of accumulators.')
        mask = response_values == winner
        output[mask] = _average_nondecision_pdf(
            rt_values[mask], t0, st0,
            lambda decision_time: _race_pdf(
                decision_time, winner_index, _wald_single_pdf, _wald_single_cdf, accumulator_parameters))
    return _restore_shape(output, scalar)


def _defective_cdf(rt, response, density_function, lower_bound):
    """Integrate one response-specific race density up to each requested RT."""
    rt_values, scalar = _as_1d(rt)
    responses = np.broadcast_to(np.asarray(response), rt_values.shape)
    output = np.zeros_like(rt_values)
    for index, (upper, current_response) in enumerate(zip(rt_values, responses)):
        if upper > lower_bound:
            output[index] = quad(
                lambda value: float(density_function(value, current_response)), lower_bound, upper,
                epsabs=1e-7, epsrel=1e-6, limit=150)[0]
    return _restore_shape(np.clip(output, 0.0, 1.0), scalar)


def diffusion_cdf(rt, response, **parameters):
    """Return defective Ratcliff diffusion CDF values."""
    lower = max(0.0, float(parameters.get('t0', 0.0)) - abs(float(parameters.get('d', 0.0))) / 2.0)
    return _defective_cdf(rt, response, lambda value, resp: diffusion_pdf(value, resp, **parameters), lower)


def lba_cdf(rt, response, **parameters):
    """Return defective LBA race CDF values."""
    return _defective_cdf(rt, response, lambda value, resp: lba_pdf(value, resp, **parameters),
                          float(parameters.get('t0', 0.0)))


def rdm_cdf(rt, response, **parameters):
    """Return defective RDM race CDF values."""
    return _defective_cdf(rt, response, lambda value, resp: rdm_pdf(value, resp, **parameters),
                          float(parameters.get('t0', 0.0)))


def _defective_quantile(probability, response, cdf_function, lower, upper, scale_probability):
    """Invert one defective CDF with optional response-probability scaling."""
    probabilities, scalar = _as_1d(probability)
    responses = np.broadcast_to(np.asarray(response), probabilities.shape)
    output = np.full(probabilities.shape, np.nan, dtype=float)
    for index, (current_probability, current_response) in enumerate(zip(probabilities, responses)):
        maximum = float(cdf_function(upper, current_response))
        target = float(current_probability) * maximum if scale_probability else float(current_probability)
        if target < 0 or target > maximum or maximum <= 0:
            continue
        if target == 0:
            output[index] = lower
        else:
            output[index] = brentq(
                lambda value: float(cdf_function(value, current_response)) - target,
                lower, upper, xtol=1e-8)
    return _restore_shape(output, scalar)


def diffusion_quantile(p, response, interval=(0.0, 10.0), scale_probability=False, **parameters):
    """Return Ratcliff diffusion quantiles for one response boundary."""
    return _defective_quantile(p, response, lambda rt, resp: diffusion_cdf(rt, resp, **parameters),
                               interval[0], interval[1], scale_probability)


def lba_quantile(p, response, interval=(0.0, 10.0), scale_probability=False, **parameters):
    """Return LBA race quantiles for one winning accumulator."""
    return _defective_quantile(p, response, lambda rt, resp: lba_cdf(rt, resp, **parameters),
                               interval[0], interval[1], scale_probability)


def rdm_quantile(p, response, interval=(0.0, 10.0), scale_probability=False, **parameters):
    """Return RDM race quantiles for one winning accumulator."""
    return _defective_quantile(p, response, lambda rt, resp: rdm_cdf(rt, resp, **parameters),
                               interval[0], interval[1], scale_probability)


def lba_random(n, A, b, t0, mean_v, sd_v, st0=0.0, random_state=None):
    """Generate response times and winning accumulators from a normal-drift LBA."""
    rng = np.random.default_rng(random_state)
    mean_v = np.asarray(mean_v, dtype=float)
    sd_v = np.broadcast_to(np.asarray(sd_v, dtype=float), mean_v.shape)
    drifts = rng.normal(mean_v, sd_v, size=(int(n), len(mean_v)))
    while np.any(drifts <= 0):
        invalid = drifts <= 0
        drifts[invalid] = rng.normal(
            np.broadcast_to(mean_v, drifts.shape)[invalid],
            np.broadcast_to(sd_v, drifts.shape)[invalid])
    starts = rng.uniform(0.0, A, size=drifts.shape)
    finishing_times = (b - starts) / drifts
    response = np.argmin(finishing_times, axis=1) + 1
    rt = finishing_times[np.arange(int(n)), response - 1] + t0 + rng.uniform(0.0, st0, int(n))
    return pd.DataFrame({'rt': rt, 'response': response})


def _inverse_gaussian_random(distance, drift, scale, rng):
    """Draw one Wald first-passage time for fixed start distance."""
    if drift <= 1e-12:
        return (distance / scale) ** 2 / (rng.normal() ** 2)
    mean = distance / drift
    shape = distance * distance / (scale * scale)
    squared_normal = rng.normal() ** 2
    candidate = mean + mean * mean * squared_normal / (2.0 * shape) - (
        mean / (2.0 * shape)) * math.sqrt(
            4.0 * mean * shape * squared_normal + mean * mean * squared_normal * squared_normal)
    return candidate if rng.random() <= mean / (mean + candidate) else mean * mean / candidate


def rdm_random(n, A, b, t0, v, s=1.0, st0=0.0, random_state=None):
    """Generate response times and winning accumulators from an RDM."""
    rng = np.random.default_rng(random_state)
    v = np.asarray(v, dtype=float)
    finishing_times = np.empty((int(n), len(v)), dtype=float)
    for row in range(int(n)):
        for column, drift in enumerate(v):
            distance = b - rng.uniform(0.0, A)
            finishing_times[row, column] = (
                math.inf if drift < 0 else _inverse_gaussian_random(distance, drift, s, rng))
    response = np.argmin(finishing_times, axis=1) + 1
    rt = finishing_times[np.arange(int(n)), response - 1] + t0 + rng.uniform(0.0, st0, int(n))
    return pd.DataFrame({'rt': rt, 'response': response})


def diffusion_random(n, a, v, t0, z=None, d=0.0, sz=0.0, sv=0.0, st0=0.0,
                     s=1.0, random_state=None, method='quantile', time_step=1e-4):
    """Generate Ratcliff diffusion trials by numerical inversion or Euler simulation."""
    rng = np.random.default_rng(random_state)
    z = 0.5 * a if z is None else z
    if method == 'quantile':
        parameters = dict(a=a, v=v, t0=t0, z=z, d=d, sz=sz, sv=sv, st0=st0, s=s)
        maximum_time = max(10.0, t0 + st0 + 10.0)
        lower_probability = float(diffusion_cdf(maximum_time, 'lower', **parameters))
        upper_probability = float(diffusion_cdf(maximum_time, 'upper', **parameters))
        total_probability = lower_probability + upper_probability
        output_rt = np.empty(int(n), dtype=float)
        output_response = np.empty(int(n), dtype=object)
        for trial, draw in enumerate(rng.uniform(0.0, total_probability, int(n))):
            if draw < lower_probability:
                boundary = 'lower'
                conditional_probability = draw / lower_probability
            else:
                boundary = 'upper'
                conditional_probability = (draw - lower_probability) / upper_probability
            output_rt[trial] = diffusion_quantile(
                conditional_probability, boundary, interval=(0.0, maximum_time),
                scale_probability=True, **parameters)
            output_response[trial] = boundary
        return pd.DataFrame({'rt': output_rt, 'response': output_response})
    if method != 'euler':
        raise ValueError("Diffusion random-generation method must be 'quantile' or 'euler'.")
    output_rt = np.empty(int(n), dtype=float)
    output_response = np.empty(int(n), dtype=object)
    for trial in range(int(n)):
        position = rng.uniform(z - sz / 2.0, z + sz / 2.0) if sz else z
        drift = rng.normal(v, sv) if sv else v
        decision_time = 0.0
        while 0.0 < position < a:
            position += drift * time_step + s * math.sqrt(time_step) * rng.normal()
            decision_time += time_step
        boundary = 'upper' if position >= a else 'lower'
        nondecision = t0 + rng.uniform(0.0, st0)
        nondecision += -d / 2.0 if boundary == 'upper' else d / 2.0
        output_rt[trial] = decision_time + nondecision
        output_response[trial] = boundary
    return pd.DataFrame({'rt': output_rt, 'response': output_response})


def _parameter_values(specification, free_values=None):
    """Expand fixed and free specification rows into one parameter dictionary."""
    free_values = iter([] if free_values is None else free_values)
    values = {}
    for parameter in specification['parameters']:
        values[parameter['name']] = (float(next(free_values)) if parameter['mode'] == 'free'
                                     else float(parameter['value']))
    return values


def _model_arguments(model, parameter_values, response_count):
    """Convert flat parameter names into numerical model keyword arguments."""
    if model == RATCLIFF_MODEL:
        arguments = {name: parameter_values[name]
                     for name in ('a', 'v', 't0', 'd', 'sz', 'sv', 'st0', 's')}
        arguments['z'] = parameter_values['a'] * parameter_values['zr']
        return arguments
    common = {name: parameter_values[name] for name in ('A', 'b', 't0', 'st0')}
    if model == LBA_MODEL:
        common['mean_v'] = [parameter_values[f'mean_v[{index}]'] for index in range(1, response_count + 1)]
        common['sd_v'] = [parameter_values[f'sd_v[{index}]'] for index in range(1, response_count + 1)]
    else:
        common['s'] = parameter_values['s']
        common['v'] = [parameter_values[f'v[{index}]'] for index in range(1, response_count + 1)]
    return common


def _valid_parameter_combination(model, values):
    """Return whether model parameters satisfy coupled constraints."""
    return bool(np.all(_constraint_margins(model, values) >= 0.0))


def _constraint_margins(model, values):
    """Return non-negative margins for all coupled and scale constraints."""
    epsilon = 1e-9
    if model == RATCLIFF_MODEL:
        return np.asarray([
            values['a'] - epsilon,
            values['s'] - epsilon,
            values['sz'],
            values['sv'],
            values['st0'],
            values['a'] * values['zr'] - values['sz'] / 2.0 - epsilon,
            values['a'] * (1.0 - values['zr']) - values['sz'] / 2.0 - epsilon,
            values['t0'] - abs(values['d']) / 2.0,
        ])
    margins = [
        values['A'],
        values['b'] - values['A'] - epsilon,
        values['t0'],
        values['st0'],
    ]
    if model == LBA_MODEL:
        margins.extend(
            value - epsilon for name, value in values.items() if name.startswith('sd_v['))
    else:
        margins.append(values['s'] - epsilon)
        margins.extend(value for name, value in values.items() if name.startswith('v['))
    return np.asarray(margins)


def fit_cognitive_model(dataframe, specification, validate=True, cancel_check=None):
    """Fit one shared model using the configured coding, optionally after dataset validation."""
    raise_if_fit_cancelled(cancel_check)
    if validate:
        validate_model_data(specification, dataframe)
    return _fit_response_model(dataframe, specification, cancel_check)


def _correct_response_indices(response_indices, accuracy_values):
    """Infer the correct physical boundary for every retained binary trial."""
    return np.where(np.asarray(accuracy_values) == 1, response_indices, 3 - response_indices).astype(int)


def _oriented_diffusion_arguments(arguments, correct_response, boundary_coding):
    """Return DDM arguments expressed in the requested trial's boundary coordinates."""
    oriented = dict(arguments)
    if boundary_coding == RESPONSE_CODING:
        if int(correct_response) == 1:
            oriented['v'] = -oriented['v']
    elif boundary_coding == ACCURACY_CODING:
        if int(correct_response) == 1:
            oriented['z'] = oriented['a'] - oriented['z']
            oriented['d'] = -oriented['d']
    else:
        raise ValueError(f'Unsupported Ratcliff boundary coding: {boundary_coding}.')
    return oriented


def _oriented_diffusion_pdf(rt_values, model_responses, correct_responses, arguments,
                            boundary_coding):
    """Evaluate trial densities while orienting asymmetric DDM parameters by correct response."""
    rt_values = np.asarray(rt_values, dtype=float)
    model_responses = np.asarray(model_responses, dtype=object)
    correct_responses = np.asarray(correct_responses, dtype=int)
    densities = np.zeros(rt_values.shape, dtype=float)
    for correct_response in (1, 2):
        mask = correct_responses == correct_response
        if np.any(mask):
            densities[mask] = diffusion_pdf(
                rt_values[mask], model_responses[mask],
                **_oriented_diffusion_arguments(
                    arguments, correct_response, boundary_coding))
    return densities


def _fit_response_model(dataframe, specification, cancel_check=None):
    """Fit one model jointly to all valid RT and observed-response pairs."""
    raise_if_fit_cancelled(cancel_check)
    model = specification['model']
    rt_values = pd.to_numeric(dataframe[specification['rt_variable']], errors='coerce')
    responses = dataframe[specification['response_variable']]
    valid = rt_values.notna() & np.isfinite(rt_values) & responses.notna()
    rt_values = rt_values.loc[valid].to_numpy(dtype=float)
    responses = responses.loc[valid]
    if specification.get('rt_unit') == 'milliseconds':
        rt_values = rt_values / 1000.0
    response_values = specification['response_values']
    response_mapping = specification['response_mapping']
    typed_mapping = model_response_mapping(specification)
    mapped_responses = responses.map(lambda value: typed_mapping.get(response_value_token(value))).to_numpy()
    valid_response = pd.notna(mapped_responses)
    rt_values = rt_values[valid_response]
    mapped_responses = mapped_responses[valid_response]
    if rt_values.size == 0:
        raise ValueError('No valid RT/response pairs remain for model fitting.')
    expected_levels = ({'lower', 'upper'} if model == RATCLIFF_MODEL
                       else set(range(1, len(response_values) + 1)))
    observed_levels = set(mapped_responses.tolist())
    if not observed_levels or not observed_levels.issubset(expected_levels):
        raise ValueError('The fitted group contains no valid configured response values.')
    if model == RATCLIFF_MODEL:
        model_responses = mapped_responses.astype(object)
        response_indices = np.where(model_responses == 'lower', 1, 2)
    else:
        response_indices = mapped_responses.astype(int)
        model_responses = response_indices
    accuracy_values = None
    accuracy_rate = None
    if specification.get('accuracy_variable'):
        accuracy_values = pd.to_numeric(
            dataframe.loc[valid, specification['accuracy_variable']], errors='coerce').to_numpy()[valid_response]
        if (accuracy_values.size == 0 or not np.isfinite(accuracy_values).all()
                or not np.isin(accuracy_values, (0.0, 1.0)).all()):
            raise ValueError(
                'Accuracy Variable must contain only 0 (error) and 1 (correct), with no '
                'missing values. Recode or exclude invalid rows in Filter Data.')
        accuracy_rate = float(np.mean(accuracy_values))

    boundary_coding = specification.get('boundary_coding', RESPONSE_CODING)
    correct_responses = None
    if model == RATCLIFF_MODEL:
        correct_responses = _correct_response_indices(response_indices, accuracy_values)
        if boundary_coding == ACCURACY_CODING:
            model_responses = np.where(accuracy_values == 1, 'upper', 'lower').astype(object)

    free_parameters = [parameter for parameter in specification['parameters'] if parameter['mode'] == 'free']
    bounds = [(float(parameter['lower']), float(parameter['upper'])) for parameter in free_parameters]
    pdf_function = diffusion_pdf if model == RATCLIFF_MODEL else lba_pdf if model == LBA_MODEL else rdm_pdf

    def objective(free_values):
        raise_if_fit_cancelled(cancel_check)
        values = _parameter_values(specification, free_values)
        if not _valid_parameter_combination(model, values):
            return 1e100
        arguments = _model_arguments(model, values, len(response_values))
        if model == RATCLIFF_MODEL:
            densities = _oriented_diffusion_pdf(
                rt_values, model_responses, correct_responses, arguments, boundary_coding)
        else:
            densities = np.asarray(pdf_function(rt_values, model_responses, **arguments), dtype=float)
        if np.any(~np.isfinite(densities)) or np.any(densities <= _TINY):
            return 1e100
        return float(-np.sum(np.log(densities)))

    optimizer = specification.get('optimizer', {})
    starts = max(1, int(optimizer.get('starts', 3)))
    configured_seed = optimizer.get('seed')
    random_seed = (time.time_ns() % (2 ** 32)
                   if configured_seed is None else int(configured_seed))
    rng = np.random.default_rng(random_seed)
    candidates = []
    minimum_rt = float(np.min(rt_values))
    minimum_rt_by_response = (
        {boundary: float(np.min(rt_values[mapped_responses == boundary]))
         for boundary in observed_levels} if model == RATCLIFF_MODEL else None)
    for _index in range(starts if free_parameters else 1):
        raise_if_fit_cancelled(cancel_check)
        candidate = None
        for _attempt in range(1000 if free_parameters else 1):
            raise_if_fit_cancelled(cancel_check)
            values = random_model_start(
                model, specification['parameters'], minimum_rt, rng,
                randomize_manual=_index > 0)
            sampled = np.asarray([values[parameter['name']] for parameter in free_parameters])
            if (all(lower <= value <= upper for value, (lower, upper) in zip(sampled, bounds))
                    and _valid_parameter_combination(model, values)
                    and initial_parameters_are_valid(
                        model, values, minimum_rt, minimum_rt_by_response)):
                candidate = sampled
                break
            if _index == 0 and not any(
                    is_automatic_start(parameter) for parameter in free_parameters):
                break
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        raise ValueError(
            'No starting point satisfies the parameter bounds, joint constraints and retained '
            'RT time support. Check fixed values, starting values, bounds and Filter Data.')
    if free_parameters:
        constraint = {
            'type': 'ineq',
            'fun': lambda free_values: _constraint_margins(
                model, _parameter_values(specification, free_values)),
        }
        results = [minimize(
            objective, candidate, method='SLSQP', bounds=bounds, constraints=constraint,
            options={
                'maxiter': int(optimizer.get('max_iterations', 1000)),
                'ftol': 1e-9,
            })
                   for candidate in candidates]
        raise_if_fit_cancelled(cancel_check)
        successful = [candidate for candidate in results
                      if candidate.success and np.isfinite(candidate.fun) and candidate.fun < 1e99]
        result = min(successful or results,
                     key=lambda candidate: candidate.fun if np.isfinite(candidate.fun) else np.inf)
    else:
        fixed_objective = objective(np.asarray([], dtype=float))
        result = OptimizeResult(
            x=np.asarray([], dtype=float), fun=fixed_objective,
            success=np.isfinite(fixed_objective) and fixed_objective < 1e99,
            message='All parameters were fixed; likelihood evaluated without optimization.')
    final_values = _parameter_values(specification, result.x)
    if not np.isfinite(result.fun) or result.fun >= 1e99:
        raise ValueError(
            'All optimizer runs failed to produce a valid likelihood. Check retained RTs, '
            'parameter settings and Filter Data, or try more starting points.')
    parameter_names = [parameter['name'] for parameter in specification['parameters']]
    parameter_values = [final_values[name] for name in parameter_names]
    free_count = len(free_parameters)
    log_likelihood = -float(result.fun)
    fit_warnings = []
    if observed_levels != expected_levels:
        fit_warnings.append(
            'Only one response boundary was observed in this group; response bias and drift '
            'parameters may be weakly identified.')
    if accuracy_values is not None and np.unique(accuracy_values).size == 1:
        category = 'correct' if accuracy_values[0] == 1 else 'error'
        missing = 'error' if category == 'correct' else 'correct'
        fit_warnings.append(
            f'This group contains only {category} trials and no {missing} trials; parameter '
            'recovery and fit diagnostics may be unreliable.')
    message = str(result.message)
    if fit_warnings:
        message += ' Warning: ' + ' '.join(fit_warnings)
    return {
        'model': model,
        'parameter_names': parameter_names,
        'parameters': parameter_values,
        'parameter_modes': [parameter['mode'] for parameter in specification['parameters']],
        'n_valid': int(rt_values.size),
        'response_counts': {
            str(value): int(np.sum(mapped_responses == response_mapping[str(value)]))
            for value in response_values
        },
        'accuracy': accuracy_values,
        'accuracy_rate': accuracy_rate,
        'boundary_coding': boundary_coding,
        'correct_response': correct_responses,
        'correct_response_weights': (
            np.bincount(correct_responses, minlength=3)[1:].astype(float) / correct_responses.size
            if correct_responses is not None else None),
        'fit_semantics': boundary_coding,
        'converged': bool(result.success and np.isfinite(result.fun) and result.fun < 1e99),
        'message': message,
        'fit_warnings': fit_warnings,
        'optimizer_method': 'SLSQP',
        'random_seed': random_seed,
        'initialization': 'rtdists-inspired random starts; manual first-start values preserved',
        'initial_parameters': [_parameter_values(specification, candidate) for candidate in candidates],
        'log_likelihood': log_likelihood,
        'aic': 2.0 * free_count - 2.0 * log_likelihood,
        'bic': math.log(rt_values.size) * free_count - 2.0 * log_likelihood,
        'rt': rt_values,
        'response': response_indices,
        'specification': specification,
    }


def cognitive_model_pdf(model, rt, response, parameter_names, parameter_values, response_count):
    """Evaluate a fitted cognitive model from flattened result parameters."""
    values = dict(zip(parameter_names, np.asarray(parameter_values, dtype=float)))
    arguments = _model_arguments(model, values, response_count)
    if model == RATCLIFF_MODEL:
        model_response = 'lower' if int(response) == 1 else 'upper'
        return diffusion_pdf(rt, model_response, **arguments)
    if model == LBA_MODEL:
        return lba_pdf(rt, int(response), **arguments)
    if model == RDM_MODEL:
        return rdm_pdf(rt, int(response), **arguments)
    raise ValueError(f'Unsupported cognitive RT model: {model}.')


def fitted_response_pdf(record, rt, response):
    """Predict one physical response, marginalizing over correct-response directions."""
    if record['model'] == RATCLIFF_MODEL:
        values = dict(zip(record['parameter_names'], np.asarray(record['parameters'], dtype=float)))
        arguments = _model_arguments(RATCLIFF_MODEL, values, 2)
        weights = _record_correct_response_weights(record)
        boundary = 'lower' if int(response) == 1 else 'upper'
        return sum(
            weights[correct_response - 1] * diffusion_pdf(
                rt, boundary,
                **_oriented_diffusion_arguments(
                    arguments, correct_response, RESPONSE_CODING))
            for correct_response in (1, 2)
        )
    return cognitive_model_pdf(record['model'], rt, response, record['parameter_names'],
                               record['parameters'], len(record['specification']['response_values']))


def fitted_accuracy_pdf(record, rt, correct=True):
    """Predict correct or error RTs by marginalizing over physical correct-response directions."""
    if record['model'] != RATCLIFF_MODEL:
        raise ValueError('Correct/Error model curves are currently available for Ratcliff fits only.')
    values = dict(zip(record['parameter_names'], np.asarray(record['parameters'], dtype=float)))
    arguments = _model_arguments(RATCLIFF_MODEL, values, 2)
    weights = _record_correct_response_weights(record)
    boundary = 'upper' if correct else 'lower'
    return sum(
        weights[correct_response - 1] * diffusion_pdf(
            rt, boundary,
            **_oriented_diffusion_arguments(
                arguments, correct_response, ACCURACY_CODING))
        for correct_response in (1, 2)
    )


def _record_correct_response_weights(record):
    """Return saved or inferable physical correct-response proportions for one fit record."""
    saved = record.get('correct_response_weights')
    if saved is not None:
        weights = np.asarray(saved, dtype=float)
    else:
        responses = np.asarray(record.get('response'), dtype=int)
        accuracy = np.asarray(record.get('accuracy'), dtype=float)
        correct_responses = _correct_response_indices(responses, accuracy)
        weights = np.bincount(correct_responses, minlength=3)[1:].astype(float)
        weights /= np.sum(weights)
    if weights.shape != (2,) or not np.isfinite(weights).all() or np.sum(weights) <= 0:
        raise ValueError('Correct-response weights are unavailable for this Ratcliff fit.')
    return weights
