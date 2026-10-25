"""Qt-free PsySummary CDF pooling shared by preparation and standalone replay."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


def validate_cdf_cutoff(value, allow_abort=False):
    """Return a finite cutoff strictly inside (0, 1), or the explicit abort sentinel."""
    message = 'CDF pooling cutoff must be finite and strictly between 0 and 1.'
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(message)
    try:
        cutoff = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if allow_abort and cutoff == -1:
        return cutoff
    if not np.isfinite(cutoff) or not 0 < cutoff < 1:
        raise ValueError(message)
    return cutoff


def pooled_cdf_values(frame, variable, groups=(), min_trials=50, check_cancelled=None):
    """Fit incoming groups and return original-position CDFs without changing source columns."""
    values = frame[variable]
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values) or pd.api.types.is_complex_dtype(values):
        raise ValueError('Pooling CDF requires a real Numeric variable.')
    rt = values.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(rt).all() or (rt <= 0).any():
        raise ValueError('Pooling CDF requires finite positive RTs; add an earlier eligibility rule.')
    if groups and frame[list(groups)].isna().any().any():
        raise ValueError('Pooling CDF group values cannot be missing; add an earlier eligibility rule.')
    if frame.empty:
        return np.empty(0), []
    batches = (frame.loc[:, list(groups)].reset_index(drop=True).groupby(list(groups), sort=False, observed=True, dropna=False).indices
               if groups else {(): np.arange(len(frame))})
    output, diagnostics = np.empty(len(frame)), []
    for key, positions in batches.items():
        if check_cancelled:
            check_cancelled()
        sample = rt[positions]
        if len(sample) < min_trials:
            raise ValueError(f'Pooling CDF group {key!r} has {len(sample)} trials; at least {min_trials} are required.')
        # Preserve PsySummary's moment start, positive parameters, likelihood floor,
        # and actual L-BFGS-B optimizer (its legacy method argument was ignored).
        mean, variance = np.mean(sample), np.var(sample)
        third = np.sum((sample - mean) ** 3) / (len(sample) - 1)
        tau = np.cbrt(third) / 2
        sigma = np.sqrt(max(0., variance - tau ** 2))
        mu = mean - tau
        if min(mu, sigma, tau) <= 0 or not np.isfinite([mu, sigma, tau]).all():
            tau = .8 * np.sqrt(variance)
            sigma, mu = tau * np.sqrt(1 - .8 ** 2), mean - tau
        if min(mu, sigma, tau) <= 0:
            raise ValueError(f'Pooling CDF group {key!r} has no usable RT variation.')

        def objective(parameters):
            if check_cancelled:
                check_cancelled()
            m, s, t = np.abs(parameters)
            logpdf = np.log(1 / t) + (m - sample) / t + (s ** 2) / (2 * t ** 2) + norm.logcdf((sample - m) / s - s / t)
            return -np.sum(np.maximum(logpdf, np.log(1e-10)))

        result = minimize(objective, np.array([mu, sigma, tau]), bounds=[(1e-8, None)] * 3,
                          method='L-BFGS-B', options={'maxiter': 1000})
        if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise ValueError(f'Pooling CDF group {key!r} did not converge: {result.message}')
        mu, sigma, tau = result.x
        # Evaluate the original CDF formula in log space to avoid exp * tiny CDF.
        exponent = (mu - sample) / tau + sigma ** 2 / (2 * tau ** 2)
        cdf = np.clip(norm.cdf(sample, mu, sigma) - np.exp(exponent + norm.logcdf((sample - mu) / sigma - sigma / tau)), 0, 1)
        if not np.isfinite(cdf).all():
            raise ValueError('Pooling CDF produced nonfinite probabilities.')
        output[positions] = cdf
        labels = key if isinstance(key, tuple) else (key,)
        diagnostics.append({'group': [value.item() if isinstance(value, np.generic) else value for value in labels],
                            'n': len(sample), 'parameters': result.x.tolist(), 'converged': True})
    return output, diagnostics


def outlier_mode_lnlike(parameters, cdf_values):
    """Evaluate PsySummary's uniform/slow-tail triangular mixture likelihood."""
    po, omega = parameters
    if not (0 <= po <= 1 and 0 <= omega < 1):
        return np.inf
    tail = np.zeros_like(cdf_values)
    mask = cdf_values >= omega
    tail[mask] = 2 * (cdf_values[mask] - omega) / (1 - omega) ** 2
    return -np.sum(np.log(np.maximum((1 - po) + po * tail, 1e-12)))


def fit_outlier_model(cdf_values, check_cancelled=None):
    """Estimate the same omega/contamination proposal used by PsySummary's confirmation dialog."""
    cdf_values = np.asarray(cdf_values, dtype=float)
    if not len(cdf_values) or not np.isfinite(cdf_values).all():
        raise ValueError('No finite CDF values are available for cutoff estimation.')

    def objective(parameters):
        if check_cancelled:
            check_cancelled()
        return outlier_mode_lnlike(parameters, cdf_values)

    result = minimize(objective, np.array([.01, .98]), bounds=[(.0001, .2), (.7, .999)])
    return result.x if result.success else [None, None]
