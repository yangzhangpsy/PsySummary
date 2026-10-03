import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm, chi2, expon, weibull_min, lognorm, invgauss, gamma
from scipy.optimize import minimize
from scipy.special import erfcx, log_ndtr, wofz

# reference: 1. Heathcote, A. Fitting wald and ex-wald distributions to response time data:
# An example using functions for the S-PLUS package.
# Behavior Research Methods, Instruments, & Computers 36, 678–694 (2004).

"""
Wald Distribution Functions
"""


# Wald Probability Density Function (PDF)
def wald_pdf(w, m, a, s=0):
    """
    Calculate the density of the shifted Wald distribution.
    Parameters:
        w (np.array): data points
        m    (float): parameter m - mean rate of evidence accrual
        a    (float): parameter a - response threshold
        s    (float): shift parameter
    Returns:
        np.array: density values at points w
    """
    w = w - s
    with np.errstate(divide='ignore', invalid='ignore'):
        density = a * np.exp(-(a - m * w) ** 2 / (2 * w)) / np.sqrt(2 * np.pi * w ** 3)
        density[w <= 0] = 0
    return density


# Wald Cumulative Distribution Function (CDF)
def wald_cdf(w, m, a, s=0):
    """Evaluate the Wald CDF without multiplying overflowing/underflowing terms."""
    return np.exp(_wald_logcdf(np.asarray(w, dtype=float) - s, m, a))


def _wald_logcdf(values, m, a):
    """Combine Wald CDF terms in log space using a scaled complementary error function."""
    values = np.asarray(values, dtype=float)
    result = np.full_like(values, -np.inf)
    positive = np.isfinite(values) & (values > 0)
    root = np.sqrt(values[positive])
    first = m * root - a / root
    second = m * root + a / root
    # 2*a*m - second**2/2 equals -first**2/2; no inf * 0 product.
    second_term = -0.5 * first ** 2 + np.log(erfcx(second / np.sqrt(2.0))) - np.log(2.0)
    result[positive] = np.minimum(np.logaddexp(log_ndtr(first), second_term), 0.0)
    result[np.isposinf(values)] = 0.0
    return result


# Wald random variate generation function
def wald_generate_data(n, m, a, s=0):
    """
    Generate random variates from the shifted Wald distribution.
    Parameters:
        n (int): number of samples
        m (float): parameter m - mean rate of evidence accrual
        a (float): parameter a - response threshold
        s (float): shift parameter
    Returns:
        np.array: random variates
    """
    y2 = chi2.rvs(df=1, size=n)
    y2onm = y2 / m
    u = np.random.uniform(size=n)
    r1 = (2 * a + y2onm - np.sqrt(y2onm * (4 * a + y2onm))) / (2 * m)
    r2 = (a / m) ** 2 / r1
    return np.where(u < a / (a + m * r1), s + r1, s + r2)


# Initial parameter estimates for Wald fitting
def wald_initial_value_estimate(x, shift=True, p=0.9):
    """
    Calculate initial parameter estimates for Wald fitting.
    Parameters:
        x (np.array): data points
        shift (bool): use shift parameter or not
        p    (float): proportion for calculating shift
    Returns:
        np.array: initial parameter estimates (m, a, [s])
    """
    if shift:
        s = p * np.min(x)
        x_shifted = x - s
        m = np.sqrt(np.mean(x_shifted) / np.var(x_shifted))
        a = m * np.mean(x_shifted)
        return np.array([m, a, s])
    else:
        m = np.sqrt(np.mean(x) / np.var(x))
        a = m * np.mean(x)
        return np.array([m, a])


# Negative log-likelihood for shifted Wald distribution
def wald_lnlike(p, x):
    """
    Negative log-likelihood for shifted Wald distribution.
    Parameters:
        p (np.array): parameters (m, a, s)
        x (np.array): data points
    Returns:
        float: negative log-likelihood
    """
    if len(p) == 2:
        return -np.sum(np.log(wald_pdf(x, p[0], p[1], 0)))
    else:
        return -np.sum(np.log(wald_pdf(x, p[0], p[1], p[2])))


def wald_estimate_x(rt, p=0.9):
    result = wald_estimate(rt, False, p)
    return result.x


def shift_wald_estimate_x(rt, shift=True, p=0.9):
    result = wald_estimate(rt, shift, p)
    return result.x


# Fit Wald distribution to data
def wald_estimate(rt, shift=True, p=0.9):
    """
    Fit the Wald distribution using maximum likelihood estimation.
    Parameters:
        rt (np.array): observed data
        shift (bool): indicates shift parameter usage
        p (float): proportion for initial shift estimation
    Returns:
        OptimizeResult: fitted parameters and success flag
    """
    start = wald_initial_value_estimate(rt, shift, p)
    bounds = [(1e-8, None), (1e-8, None), (None, np.min(rt))] if shift else [(1e-8, None), (1e-8, None)]

    result = minimize(wald_lnlike,
                      x0=start,
                      args=(rt,),
                      bounds=bounds,
                      method='L-BFGS-B',
                      options={'maxiter': 300})

    # fit_params = result.x
    # chisquare = chisq(rt, fit_params, dist='wald')
    return result
    # return {'parameters': fit_params, 'chisq': chisquare, 'success': result.success, 'message': result.message}


def plot_wald_fit(data, estimated_params):
    """Plot the histogram of data and fitted ex-Wald distribution."""
    m, a, t = estimated_params
    x = np.linspace(min(data), max(data), 100)
    pdf_fitted = wald_pdf(x, m, a, t)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='b', label='Histogram')
    plt.plot(x, pdf_fitted, 'r-', lw=2, label='Fitted Ex-Wald')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Wald Fit\n m: {m:.4f}, a: {a:.4f}, shift: {t:.4f}')
    plt.legend()
    plt.show()


def WaldRunDemo():
    # data = np.loadtxt('ex_wald_data2.txt')
    data = wald_generate_data(1000, m=1.5, a=0.8, s=0.5)
    np.savetxt("wald_data.txt", data, fmt="%.6f")
    # To read the data in R:
    # data <- read.table("ex_wald_data.txt", header=FALSE)[,1]
    fit_results = ex_wald_estimate_x(data)
    print(fit_results)

    # Plot the fitted distribution
    plot_wald_fit(data, fit_results)


"""
Ex-Wald Distribution Functions
"""


# Series approximation to the real and imaginary parts of the complex error function
def complex_error_function_real_imag(x, y, firstblock=20, block=0, tol=1e-8, maxseries=20):
    """
    Calculate real and imaginary parts of complex error function erf(x + iy).
    Parameters:
        x     (np.array): real parts
        y     (np.array): imaginary parts
        firstblock (int): number of initial terms
        block      (int): number of terms added if not converged
        tol      (float): tolerance for convergence
        maxseries  (int): max number of terms
    Returns:
        tuple: real and imaginary parts
    """
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    twoxy = 2 * x * y
    xsq = x ** 2
    iexpxsqpi = 1 / (np.pi * np.exp(xsq))
    sin2xy, cos2xy = np.sin(twoxy), np.cos(twoxy)

    nmat = np.tile(np.arange(1, firstblock + 1), (len(x), 1))
    nsqmat = nmat ** 2
    ny = nmat * y[:, None]
    twoxcoshny = 2 * x[:, None] * np.cosh(ny)
    nsinhny = nmat * np.sinh(ny)
    nsqfrac = np.exp(-nsqmat / 4) / (nsqmat + 4 * xsq[:, None])

    u = (2 * norm.cdf(x * np.sqrt(2)) - 1) + iexpxsqpi * (((1 - cos2xy) / (2 * x)) +
                                                          2 * np.sum(
                nsqfrac * (2 * x[:, None] - twoxcoshny * cos2xy[:, None] + nsinhny * sin2xy[:, None]), axis=1))

    v = iexpxsqpi * ((sin2xy / (2 * x)) +
                     2 * np.sum(nsqfrac * (twoxcoshny * sin2xy[:, None] + nsinhny * cos2xy[:, None]), axis=1))

    n = firstblock
    converged = np.full_like(x, False, dtype=bool)

    while block >= 1 and n < maxseries:
        if (n + block) > maxseries:
            block = maxseries - n

        idx = ~converged
        nmat = np.tile(np.arange(n + 1, n + block + 1), (idx.sum(), 1))
        nsq = nmat ** 2
        ny = nmat * y[idx, None]
        twoxcoshny = 2 * x[idx, None] * np.cosh(ny)
        nsinhny = nmat * np.sinh(ny)
        nsqfrac = np.exp(-nsq / 4) / (nsq + 4 * xsq[idx, None])

        du = iexpxsqpi[idx] * 2 * np.sum(
            nsqfrac * (2 * x[idx, None] - twoxcoshny * cos2xy[idx, None] + nsinhny * sin2xy[idx, None]), axis=1)
        dv = iexpxsqpi[idx] * 2 * np.sum(nsqfrac * (twoxcoshny * sin2xy[idx, None] + nsinhny * cos2xy[idx, None]),
                                         axis=1)

        u[idx] += du
        v[idx] += dv

        converged[idx] = (np.abs(du) < tol) & (np.abs(dv) < tol)
        if np.all(converged):
            break

        n += block

    return u, v


# Real part of w(z) function used in Ex-Wald
def exwald_w_function_real_part(x, y):
    """
    Compute real part of w(z) = exp(-z^2) * [1 - erf(-iz)].
    Parameters:
        x, y (np.ndarray): real and imaginary parts
    Returns:
        np.ndarray: real parts
    """
    return wofz(np.asarray(x) + 1j * np.asarray(y)).real


def ex_wald_logpdf(r, m, a, t):
    """Evaluate Ex-Wald log density with stable real/complex branches and full support."""
    values = np.asarray(r, dtype=float)
    result = np.full_like(values, -np.inf)
    if not all(np.isfinite(parameter) and parameter > 0 for parameter in (m, a, t)):
        return result
    positive = np.isfinite(values) & (values > 0)
    times = values[positive]
    root = np.sqrt(times)
    residual = a / root - m * root
    base = -0.5 * residual ** 2
    k = m ** 2 - 2.0 / t
    if k < 0:
        real_part = exwald_w_function_real_part(np.sqrt(-times * k / 2.0), a / (np.sqrt(2.0) * root))
        # The quadratic exponent is non-positive. wofz avoids exp/cosh series overflow.
        valid = np.isfinite(real_part) & (real_part > 0)
        log_density = np.full_like(times, -np.inf)
        log_density[valid] = base[valid] + np.log(real_part[valid]) - np.log(t)
    else:
        rate = np.sqrt(k)
        first = rate * root - a / root
        second = rate * root + a / root
        log_density = np.empty_like(times)
        leading = first <= 0
        log_density[leading] = base[leading] + np.log(
            0.5 * (erfcx(-first[leading] / np.sqrt(2.0))
                   + erfcx(second[leading] / np.sqrt(2.0)))) - np.log(t)
        # Rationalize m-rate when tau is small to avoid cancellation in that difference.
        first_term = (log_ndtr(first[~leading])
                      + 2.0 * a / (t * (m + rate)) - times[~leading] / t)
        second_term = base[~leading] + np.log(erfcx(second[~leading] / np.sqrt(2.0))) - np.log(2.0)
        log_density[~leading] = np.logaddexp(first_term, second_term) - np.log(t)
    result[positive] = log_density
    return result


# Ex-Wald PDF
def ex_wald_pdf(r, m, a, t):
    """Return the Ex-Wald density, allowing harmless tail underflow to zero."""
    return np.exp(ex_wald_logpdf(r, m, a, t))


# Ex-Wald Cumulative Distribution Function (CDF)
def ex_wald_cdf(r, m, a, t):
    """
    Calculate the cumulative density of the Ex-Wald distribution.
    Parameters:
        r (np.array): data points
        m (float): parameter m
        a (float): parameter a
        t (float): parameter t
    Returns:
        np.array: cumulative density values at points r
    """
    values = np.asarray(r, dtype=float)
    result = np.zeros_like(values)
    positive = np.isfinite(values) & (values > 0)
    log_wald = _wald_logcdf(values[positive], m, a)
    log_removed = np.log(t) + ex_wald_logpdf(values[positive], m, a, t)
    finite = np.isfinite(log_wald)
    probabilities = np.zeros_like(log_wald)
    probabilities[finite] = np.exp(log_wald[finite]) * -np.expm1(
        np.minimum(log_removed[finite] - log_wald[finite], 0.0))
    result[positive] = np.clip(probabilities, 0.0, 1.0)
    result[np.isposinf(values)] = 1.0
    return result


# Ex-Wald random variate generation function
def ex_wald_generate_data(n, m, a, t):
    """
    Generate random variates from the Ex-Wald distribution.
    Parameters:
        n   (int): number of samples
        m (float): parameter m
        a (float): parameter a
        t (float): parameter tau (τ)
    Returns:
        np.array: random variates
    """
    return wald_generate_data(n, m, a) + expon.rvs(scale=t, size=n)


# Estimate initial parameters for Ex-Wald fitting based on data moments
def ex_wald_initial_value_estimate(x, p=0.5):
    """
    Calculate initial parameter estimates for Ex-Wald fitting.
    Parameters:
        x (np.array): data points
        p (float): proportion used to calculate initial t parameter
    Returns:
        np.array: initial parameter estimates (m, a, t)
    """
    mean = float(np.mean(x))
    variance = max(float(np.var(x)), (mean * 1e-6) ** 2)
    t = np.clip(p * np.sqrt(variance), 1e-8, max(1e-8, 0.9 * mean))
    m = np.sqrt(max(mean - t, 1e-8) / max(variance - t ** 2, variance * 1e-6))
    a = m * (np.mean(x) - t)
    return np.array([m, a, t])


# Negative log-likelihood for Ex-Wald distribution
def ex_wald_lnlike(p, x):
    """
    Compute negative log-likelihood of the Ex-Wald distribution.
    Parameters:
        p (np.array): parameter array (m, a, t)
        x (np.array): observed data
    Returns:
        float: negative log-likelihood value
    """
    return _finite_negative_loglike(ex_wald_logpdf(x, p[0], p[1], p[2]))


# Fit Ex-Wald distribution to data
def ex_wald_estimate_x(rt, p=0.5, scaleit=True):
    result = ex_wald_estimate(rt, p, scaleit)
    return result.x


# Fit Ex-Wald distribution to data
def ex_wald_estimate(rt, p=0.5, scaleit=True):
    """
    Fit Ex-Wald distribution using maximum likelihood.
    Parameters:
        rt (np.array): observed data
        p     (float): proportion for initial t estimation
        scaleit (bool): whether to scale optimization parameters
    Returns:
        OptimizeResult: fitted parameters and chi-square statistics
    """
    rt = np.asarray(rt, dtype=float)
    if rt.size < 3 or not np.all(np.isfinite(rt)) or np.any(rt <= 0):
        raise ValueError('Ex-Wald fitting requires at least three finite positive RT observations.')
    start = ex_wald_initial_value_estimate(rt, p)
    scale = 1 / start if scaleit else None

    result = minimize(ex_wald_lnlike, x0=start, args=(rt,), method='L-BFGS-B',
                      bounds=[(1e-8, None), (1e-8, None), (1e-8, None)],
                      options={'maxiter': 1000})

    # fit_params = result.x
    # chisquare = chisq(rt, fit_params, dist='exw')

    return result


def plot_ex_wald_fit(data, estimated_params):
    """Plot the histogram of data and fitted ex-Wald distribution."""
    m, a, t = estimated_params
    x = np.linspace(min(data), max(data), 100)
    pdf_fitted = ex_wald_pdf(x, m, a, t)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='b', label='Histogram')
    plt.plot(x, pdf_fitted, 'r-', lw=2, label='Fitted Ex-Wald')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Ex-Wald Fit\n m: {m:.4f}, a: {a:.4f}, t: {t:.4f}')
    plt.legend()
    plt.show()


def ExWaldRunDemo():
    # data = np.loadtxt('ex_wald_data2.txt')
    data = ex_wald_generate_data(1000, m=1.5, a=0.8, t=0.5)
    # np.savetxt("ex_wald_data2.txt", data, fmt="%.6f")
    # To read the data in R:
    # data <- read.table("ex_wald_data.txt", header=FALSE)[,1]
    fit_results = ex_wald_estimate_x(data)
    print(fit_results)

    # Plot the fitted distribution
    plot_ex_wald_fit(data, fit_results)


"""
ex-gaussian
"""


# Ex-Gaussian Probability Density Function (PDF)
def ex_gaussian_pdf(x, m, s, t):
    """
    Calculate the density of the Ex-Gaussian distribution.
    Parameters:
        x (np.array): data points
        m    (float): parameter mu
        s    (float): parameter sigma
        t    (float): parameter tau
    Returns:
        np.array: density values at points x
    """
    return np.exp(((m - x) / t) + 0.5 * (s / t) ** 2) * norm.cdf(((x - m) / s) - (s / t)) / t


# Ex-Gaussian Cumulative Distribution Function (CDF)
def ex_gaussian_cdf_old(x, m, s, t):
    """
    Calculate the cumulative density of the Ex-Gaussian distribution.
    Parameters:
        x (np.array): data points
        m    (float): parameter mu
        s    (float): parameter sigma
        t    (float): parameter tau
    Returns:
        np.array: cumulative density values at points x
    """
    rtsu = (x - m) / s
    return norm.cdf(rtsu) - np.exp((s ** 2 / (2 * t ** 2)) - ((x - m) / t)) * norm.cdf(rtsu - (s / t))


def ex_gaussian_cdf(x, mu=5, sigma=1, tau=1):
    """
    Compute the CDF of the ex-Gaussian distribution, ensuring values are within [0,1].
    """
    # x = np.clip(x, -1e10, 1e10)
    cdf_values = norm.cdf(x, mu, sigma) - np.exp((mu - x) / tau + (sigma ** 2) / (2 * tau ** 2)) * norm.cdf(
        (x - mu) / sigma - sigma / tau)
    cdf_values = np.clip(cdf_values, 0, 1)
    return cdf_values


# Ex-Gaussian random variate generation function
def ex_gaussian_generate_data(n, m, s, t):
    """
    Generate random variates from the Ex-Gaussian distribution.
    Parameters:
        n (int): number of samples
        m (float): parameter mu
        s (float): parameter sigma
        t (float): parameter tau
    Returns:
        np.array: random variates
    """
    return expon.rvs(scale=t, size=n) + norm.rvs(loc=m, scale=s, size=n)


# Estimate initial parameters for Ex-Gaussian fitting based on data moments
def ex_gaussian_initial_value(rt, p=0.8):
    """
    Calculate initial parameter estimates for Ex-Gaussian fitting.
    Parameters:
        rt (np.array): data points
        p     (float): proportion of variance attributed to tau
    Returns:
        np.array: initial parameter estimates (mu, sigma, tau)
    """
    m1 = np.mean(rt)
    m2 = np.var(rt)
    m3 = np.sum((rt - m1) ** 3) / (len(rt) - 1)
    tau = (m3 ** (1 / 3)) / 2
    sig = np.sqrt(m2 - tau ** 2)
    mu = m1 - tau
    if np.any(np.array([mu, sig, tau]) <= 0):
        tau = p * np.sqrt(m2)
        sig = tau * np.sqrt(1 - p ** 2)
        mu = m1 - tau
    return np.array([mu, sig, tau])


# Negative log-likelihood for Ex-Gaussian distribution
def ex_gaussian_lnlike_old(p, x):
    """
    Compute negative log-likelihood of the Ex-Gaussian distribution.
    Parameters:
        p (np.array): parameter array (mu, sigma, tau)
        x (np.array): observed data
    Returns:
        float: negative log-likelihood value
    """
    return -np.sum((((p[0] - x) / p[2]) + 0.5 * (p[1] / p[2]) ** 2) +
                   np.log(norm.cdf(((x - p[0]) / p[1]) - (p[1] / p[2])) / (p[2] * np.sqrt(2 * np.pi))))


def ex_gaussian_lnlike(params, rts, rt_bounds=None):
    """
    Compute the log-likelihood of response times under an ex-Gaussian model.
    """
    mu, sigma, tau = np.abs(params)
    # pdf_values = (1 / tau) * np.exp((mu - rts) / tau + (sigma ** 2) / (2 * tau ** 2)) * norm.cdf(
    #     (rts - mu) / sigma - sigma / tau)

    # Use log-space calculations to prevent overflow
    log_pdf_values = np.log(1 / tau) + (mu - rts) / tau + (sigma ** 2) / (2 * tau ** 2) + \
                     norm.logcdf((rts - mu) / sigma - sigma / tau)
    pdf_values = np.exp(log_pdf_values)

    pdf_values = np.clip(pdf_values, 1e-10, np.inf)  # Avoid log(0)

    if rt_bounds is not None:
        lower_bound, upper_bound = rt_bounds
        if (np.min(rts) < lower_bound) or (np.max(rts) > upper_bound):
            raise ValueError("Likelihood cannot be computed if any RTs are outside the bounds")
        lost_prob = ex_gaussian_cdf(lower_bound, mu, sigma, tau) + (1 - ex_gaussian_cdf(upper_bound, mu, sigma, tau))
        pdf_values /= (1 - lost_prob)

    return -np.sum(np.log(pdf_values))


def ex_gaussian_estimate_x(rt, p=0.8, method='L-BFGS-B'):
    result = ex_gaussian_estimate(rt, p, method)
    return result.x


# Fit Ex-Gaussian distribution to data
def ex_gaussian_estimate(rt, p=0.8, method='L-BFGS-B'):
    """
    Fit the Ex-Gaussian distribution to data using maximum likelihood estimation.
    Parameters:
        rt (np.array): observed data
        p     (float): proportion of variance attributed to tau for initial parameter estimation
    Returns:
        OptimizeResult: fitted parameters and success flag
    """
    start = ex_gaussian_initial_value(rt, p)
    bounds = [(1e-8, None), (1e-8, None), (1e-8, None)]

    result = minimize(ex_gaussian_lnlike,
                      x0=start,
                      args=(rt,),
                      bounds=bounds,
                      method='L-BFGS-B',
                      options={'maxiter': 1000})

    # fit_params = result.x
    return result
    # chisquare = chisq(rt, fit_params, dist='exg')


# demo code:
def plot_exgaussian_fit(data, estimated_params):
    """Plot the histogram of the data and the fitted ex-Gaussian distribution."""
    mu, sigma, tau = estimated_params
    x = np.linspace(min(data), max(data), 100)
    pdf_fitted = ex_gaussian_pdf(x, mu, sigma, tau)

    # pdf_fitted = (1 / tau) * np.exp((mu - x) / tau + (sigma ** 2) / (2 * tau ** 2)) * norm.cdf(
    #     (x - mu) / sigma - sigma / tau)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='b', label='Histogram')
    plt.plot(x, pdf_fitted, 'r-', lw=2, label='Fitted Ex-Gaussian')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Ex-Gaussian Fit\nMu: {mu:.4f}, Sigma: {sigma:.4f}, Tau: {tau:.4f}')
    plt.legend()
    plt.show()


def exGaussianRunDemo():
    rts = ex_gaussian_generate_data(1000, 900, 150, 300)
    np.savetxt("exGaussian_data.txt", rts, fmt="%.6f")
    # To read the data in R:
    # rts <- read.table("rts_data.txt", header=FALSE)[,1]
    fitResult = ex_gaussian_estimate_x(rts)
    print(fitResult - [900, 150, 110])

    # Plot the fitted distribution
    plot_exgaussian_fit(rts, fitResult)


"""
weibull distribution
"""


def _prepare_shifted_rt_data(data):
    """Return finite positive RT data suitable for a three-parameter fit."""
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 4:
        raise ValueError("At least four finite RT observations are required for a shifted distribution fit.")
    if np.any(values <= 0):
        raise ValueError("Shifted RT distributions require strictly positive observations.")
    return values


def _shift_upper_bound(data):
    """Return a regularized upper bound below the smallest observed RT."""
    minimum = float(np.min(data))
    unique_values = np.unique(data)
    positive_steps = np.diff(unique_values)
    resolution_margin = 0.5 * np.min(positive_steps) if positive_steps.size else 0.0
    scale_margin = 0.01 * minimum
    numeric_margin = np.finfo(float).eps * max(1.0, minimum) * 100
    margin = max(resolution_margin, scale_margin, numeric_margin)
    upper_bound = minimum - margin
    if upper_bound <= 0:
        raise ValueError("The observed RT range does not permit a non-negative shift parameter.")
    return upper_bound


def _shifted_distribution_lnlike(params, data, distribution, data_bounds=None):
    """Compute a support-aware negative log-likelihood for a shifted distribution."""
    shape, scale, shift = params
    if shape <= 0 or scale <= 0 or shift < 0 or np.any(data <= shift):
        return 1e100

    log_pdf_values = (_weibull_logpdf(data, shape, scale, shift) if distribution is weibull_min
                      else distribution.logpdf(data, shape, loc=shift, scale=scale))
    if not np.all(np.isfinite(log_pdf_values)):
        return 1e100

    result = _finite_negative_loglike(log_pdf_values)
    if data_bounds is not None:
        lower_bound, upper_bound = data_bounds
        if np.min(data) < lower_bound or np.max(data) > upper_bound:
            raise ValueError("Likelihood cannot be computed if any data points are outside the bounds")
        retained_probability = (
            (_weibull_cdf(upper_bound, shape, scale, shift)
             - _weibull_cdf(lower_bound, shape, scale, shift)) if distribution is weibull_min else
            (distribution.cdf(upper_bound, shape, loc=shift, scale=scale)
             - distribution.cdf(lower_bound, shape, loc=shift, scale=scale))
        )
        if not 0 < retained_probability <= 1:
            return 1e100
        result += data.size * np.log(retained_probability)
    return result if np.isfinite(result) and result < 1e100 else 1e100


def _estimate_shifted_distribution(data, distribution, start_shape_vals, scale_initializer,
                                   data_bounds=None, method="L-BFGS-B"):
    """Fit a shifted positive distribution from several stable starting points."""
    values = _prepare_shifted_rt_data(data)
    shift_upper = _shift_upper_bound(values)
    data_scale = max(float(np.max(values)), float(np.ptp(values)), 1.0)
    parameter_bounds = [(1e-4, 100.0), (data_scale * 1e-8, data_scale * 100.0), (0.0, shift_upper)]
    shift_starts = np.unique(np.array([0.0, 0.25, 0.5, 0.75]) * shift_upper)
    best_result = None

    for shape_guess in start_shape_vals:
        for shift_guess in shift_starts:
            shifted_values = values - shift_guess
            scale_guess = float(scale_initializer(shifted_values, shape_guess))
            scale_guess = np.clip(scale_guess, parameter_bounds[1][0], parameter_bounds[1][1])
            start_params = np.array([shape_guess, scale_guess, shift_guess])
            result = minimize(
                _shifted_distribution_lnlike,
                start_params,
                args=(values, distribution, data_bounds),
                method=method,
                bounds=parameter_bounds,
                options={'maxiter': 1000},
            )
            if result.success and np.isfinite(result.fun) and result.fun < 1e100 and (best_result is None or result.fun < best_result.fun):
                best_result = result

    if best_result is None:
        raise RuntimeError("No finite shifted-distribution fit could be found.")
    return best_result


def weibull_cdf(x, shape, scale):
    """
    Compute the CDF of the Weibull distribution, ensuring values are within [0,1].
    """
    return _weibull_cdf(x, shape, scale)


def _finite_negative_loglike(log_values):
    """Penalize every invalid observation without infinite finite-difference steps."""
    if not np.all(np.isfinite(log_values)):
        return 1e100
    result = -float(np.sum(log_values))
    return result if np.isfinite(result) and result < 1e100 else 1e100


def _weibull_logpdf(x, shape, scale, shift=0.0):
    """Evaluate Weibull log density without raising an unbounded power."""
    values = np.asarray(x, dtype=float) - shift
    result = np.full_like(values, -np.inf)
    if not np.isfinite(shape) or not np.isfinite(scale) or shape <= 0 or scale <= 0:
        return result
    positive = np.isfinite(values) & (values > 0)
    log_ratio = np.log(values[positive]) - np.log(scale)
    log_power = shape * log_ratio
    # Beyond this limit the likelihood is already worse than the optimizer penalty.
    evaluable = log_power <= np.log(1e100)
    log_values = np.full_like(log_ratio, -np.inf)
    log_values[evaluable] = (np.log(shape) - np.log(scale) + (shape - 1.0) * log_ratio[evaluable]
                             - np.exp(log_power[evaluable]))
    result[positive] = log_values
    return result


def _weibull_cdf(x, shape, scale, shift=0.0):
    """Evaluate Weibull probabilities without overflow, including the far right tail."""
    values = np.asarray(x, dtype=float) - shift
    if not np.isfinite(shape) or not np.isfinite(scale) or shape <= 0 or scale <= 0:
        return np.full_like(values, np.nan)
    result = np.zeros_like(values)
    positive = np.isfinite(values) & (values > 0)
    log_power = shape * (np.log(values[positive]) - np.log(scale))
    # exp(-power) is below double precision at power=745; the CDF is then exactly 1.
    saturated = log_power > np.log(745.0)
    probabilities = np.ones_like(log_power)
    probabilities[~saturated] = -np.expm1(-np.exp(log_power[~saturated]))
    result[positive] = probabilities
    result[np.isposinf(values)] = 1.0
    return result


def weibull_lnlike(params, data, data_bounds=None):
    """
    Compute the log-likelihood of the data under a Weibull model.
    """
    shape, scale = np.asarray(params, dtype=float)
    if shape <= 0 or scale <= 0:
        return 1e100

    values = np.asarray(data, dtype=float)
    log_pdf_values = _weibull_logpdf(values, shape, scale)
    if not np.all(np.isfinite(log_pdf_values)):
        return 1e100

    result = -float(np.sum(log_pdf_values))

    if data_bounds is not None:
        lower_bound, upper_bound = data_bounds
        if np.min(values) < lower_bound or np.max(values) > upper_bound:
            raise ValueError("Likelihood cannot be computed if any data points are outside the bounds")
        with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
            retained_probability = (
                weibull_cdf(upper_bound, shape, scale)
                - weibull_cdf(lower_bound, shape, scale)
            )
        if not np.isfinite(retained_probability) or not 0 < retained_probability <= 1:
            return 1e100
        result += values.size * np.log(retained_probability)

    return result if np.isfinite(result) else 1e100


def weibull_estimate_x(data, start_shape_vals=None, data_bounds=None, method="L-BFGS-B"):
    weibull_estimated = weibull_estimate(data, start_shape_vals, data_bounds, method)
    return weibull_estimated['x']


def weibull_estimate(data, start_shape_vals=None, data_bounds=None, method="L-BFGS-B"):
    """
    Estimate the parameters shape and scale using maximum likelihood estimation.
    """
    if start_shape_vals is None:
        start_shape_vals = [0.5, 1.0, 1.5, 2.0]
    if method == "BFGS":
        method = "L-BFGS-B"

    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        raise ValueError("At least three finite RT observations are required for a Weibull fit.")
    if np.any(values <= 0):
        raise ValueError("Weibull RT fitting requires strictly positive observations.")

    data_mean = float(np.mean(values))
    data_scale = max(float(np.max(values)), float(np.ptp(values)), 1.0)
    parameter_bounds = [
        (1e-4, 100.0),
        (data_scale * 1e-8, data_scale * 100.0),
    ]
    best_result = None

    for shape_guess in start_shape_vals:
        unit_mean = float(weibull_min.mean(shape_guess, scale=1.0))
        scale_guess = data_mean / unit_mean
        start_params = np.array([
            np.clip(shape_guess, *parameter_bounds[0]),
            np.clip(scale_guess, *parameter_bounds[1]),
        ])
        result = minimize(
            weibull_lnlike,
            start_params,
            args=(values, data_bounds),
            method=method,
            bounds=parameter_bounds,
            options={'maxiter': 1000},
        )

        if np.isfinite(result.fun) and result.fun < 1e100 \
                and (best_result is None or result.fun < best_result.fun):
            best_result = result
            best_result.start_shape = shape_guess

    if best_result is None:
        raise RuntimeError("No finite Weibull fit could be found.")
    return best_result


def shifted_weibull_cdf(x, shape, scale, shift):
    """Compute the CDF of a three-parameter shifted Weibull distribution."""
    return _weibull_cdf(x, shape, scale, shift)


def shifted_weibull_lnlike(params, data, data_bounds=None):
    """Compute the negative log-likelihood of a shifted Weibull model."""
    return _shifted_distribution_lnlike(params, np.asarray(data, dtype=float), weibull_min, data_bounds)


def shifted_weibull_estimate_x(data, start_shape_vals=None, data_bounds=None, method="L-BFGS-B"):
    """Return shifted Weibull shape, scale, and shift estimates."""
    return shifted_weibull_estimate(data, start_shape_vals, data_bounds, method).x


def shifted_weibull_estimate(data, start_shape_vals=None, data_bounds=None, method="L-BFGS-B"):
    """Estimate shifted Weibull parameters with constrained multi-start likelihood optimization."""
    if start_shape_vals is None:
        start_shape_vals = [0.75, 1.0, 1.5, 2.0, 3.0]
    return _estimate_shifted_distribution(
        data,
        weibull_min,
        start_shape_vals,
        lambda shifted, _shape: np.mean(shifted),
        data_bounds,
        method,
    )


def weibull_generate_data(n=100, shape=2.0, scale=100):
    """Generate random data from a Weibull distribution."""
    return weibull_min.rvs(shape, scale=scale, size=n)


def plot_weibull_fit(data, estimated_params):
    """Plot the histogram of the data and the fitted Weibull distribution."""
    shape, scale = estimated_params
    x = np.linspace(min(data), max(data), 100)
    pdf_fitted = weibull_min.pdf(x, shape, scale=scale)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='b', label='Histogram')
    plt.plot(x, pdf_fitted, 'r-', lw=2, label='Fitted Weibull')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Weibull Fit\nShape: {shape:.4f}, Scale: {scale:.4f}')
    plt.legend()
    plt.show()
    plt.pause(0.001)  # 让 Python 继续执行后续代码
    plt.ioff()  # 关闭交互模式


# Demo code:
def weibullRunDemo():
    data = weibull_generate_data(1000, 2.5, 120)
    np.savetxt("weibull_data.txt", data, fmt="%.6f")
    # To read the data in R:
    # data <- read.table("weibull_data.txt", header=FALSE)[,1]
    fitResults = weibull_estimate(data)
    print(fitResults['x'] - [2.5, 120])
    plot_weibull_fit(data, fitResults['x'])


"""
log normal distribution
"""


def log_normal_cdf(x, shape, scale):
    """
    Compute the CDF of the Lognormal distribution, ensuring values are within [0,1].
    """
    return np.clip(lognorm.cdf(x, shape, scale=scale), 0, 1)


def log_normal_lnlike(params, data, data_bounds=None):
    """
    Compute the log-likelihood of the data under a Lognormal model.
    """
    shape, scale = np.abs(params)  # Ensure positive parameters
    pdf_values = lognorm.pdf(data, shape, scale=scale)
    pdf_values = np.clip(pdf_values, 1e-10, np.inf)  # Avoid log(0)

    if data_bounds is not None:
        lower_bound, upper_bound = data_bounds
        if (np.min(data) < lower_bound) or (np.max(data) > upper_bound):
            raise ValueError("Likelihood cannot be computed if any data points are outside the bounds")
        lost_prob = log_normal_cdf(lower_bound, shape, scale) + (1 - log_normal_cdf(upper_bound, shape, scale))
        pdf_values /= (1 - lost_prob)

    return -np.sum(np.log(pdf_values))


def log_normal_estimate_x(data, start_shape_vals=None, data_bounds=None, method="BFGS"):
    lognormal_estimated = log_normal_estimate(data, start_shape_vals, data_bounds, method)
    return lognormal_estimated['x']


def log_normal_estimate(data, start_shape_vals=None, data_bounds=None, method="BFGS"):
    """
    Estimate the parameters shape and scale using maximum likelihood estimation.
    """
    if start_shape_vals is None:
        start_shape_vals = [0.5, 1.0, 1.5, 2.0, 2.5]

    data_mean = np.mean(data)
    best_result = None

    for shape_guess in start_shape_vals:
        scale_guess = data_mean / np.exp(shape_guess)  # Initial scale estimate
        start_params = [shape_guess, scale_guess]
        result = minimize(log_normal_lnlike, np.array(start_params), args=(data, data_bounds), method=method)

        if best_result is None or result.fun < best_result.fun:
            best_result = result
            best_result.start_shape = shape_guess

    best_result.x = np.abs(best_result.x)  # Ensure parameters are positive
    return best_result


def shifted_log_normal_cdf(x, shape, scale, shift):
    """Compute the CDF of a three-parameter shifted lognormal distribution."""
    return np.clip(lognorm.cdf(x, shape, loc=shift, scale=scale), 0, 1)


def shifted_log_normal_lnlike(params, data, data_bounds=None):
    """Compute the negative log-likelihood of a shifted lognormal model."""
    return _shifted_distribution_lnlike(params, np.asarray(data, dtype=float), lognorm, data_bounds)


def shifted_log_normal_estimate_x(data, start_shape_vals=None, data_bounds=None, method="L-BFGS-B"):
    """Return shifted lognormal shape, scale, and shift estimates."""
    return shifted_log_normal_estimate(data, start_shape_vals, data_bounds, method).x


def shifted_log_normal_estimate(data, start_shape_vals=None, data_bounds=None, method="L-BFGS-B"):
    """Estimate shifted lognormal parameters with constrained multi-start likelihood optimization."""
    if start_shape_vals is None:
        start_shape_vals = [0.25, 0.5, 0.8, 1.2, 2.0]
    return _estimate_shifted_distribution(
        data,
        lognorm,
        start_shape_vals,
        lambda shifted, _shape: np.exp(np.mean(np.log(shifted))),
        data_bounds,
        method,
    )


def log_normal_generate_data(n=100, shape=0.5, scale=100):
    """Generate random data from a Lognormal distribution."""
    return lognorm.rvs(shape, scale=scale, size=n)


def plot_log_normal_fit(data, estimated_params):
    """Plot the histogram of the data and the fitted lognormal distribution."""
    shape, scale = estimated_params
    x = np.linspace(min(data), max(data), 100)
    pdf_fitted = lognorm.pdf(x, shape, scale=scale)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='b', label='Histogram')
    plt.plot(x, pdf_fitted, 'r-', lw=2, label='Fitted Lognormal')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Lognormal Fit\nShape: {shape:.4f}, Scale: {scale:.4f}')
    plt.title('Lognormal Fit')
    plt.legend()
    plt.show()


def logNormalRunDemo():
    data = log_normal_generate_data(1000, 0.8, 120)
    np.savetxt("lognormal_data.txt", data, fmt="%.6f")
    # To read the data in R:
    # data <- read.table("lognormal_data.txt", header=FALSE)[,1]
    fitResults = log_normal_estimate(data)
    print(fitResults['x'] - [0.8, 120])

    # Plot the fitted distribution
    plot_log_normal_fit(data, fitResults['x'])


"""
gamma distribution
"""


def gamma_cdf(x, shape, scale):
    """
    Compute the CDF of the Gamma distribution, ensuring values are within [0,1].
    """
    return np.clip(gamma.cdf(x, shape, scale=scale), 0, 1)


def gamma_lnlike(params, data, data_bounds=None):
    """
    Compute the log-likelihood of the data under a Gamma model.
    """
    shape, scale = np.abs(params)  # Ensure positive parameters
    pdf_values = gamma.pdf(data, shape, scale=scale)
    pdf_values = np.clip(pdf_values, 1e-10, np.inf)  # Avoid log(0)

    if data_bounds is not None:
        lower_bound, upper_bound = data_bounds
        if (np.min(data) < lower_bound) or (np.max(data) > upper_bound):
            raise ValueError("Likelihood cannot be computed if any data points are outside the bounds")
        lost_prob = gamma_cdf(lower_bound, shape, scale) + (1 - gamma_cdf(upper_bound, shape, scale))
        pdf_values /= (1 - lost_prob)

    return -np.sum(np.log(pdf_values))


def gamma_estimate_x(data, start_shape_vals=None, data_bounds=None, method="BFGS"):
    gamma_estimated = gamma_estimate(data, start_shape_vals, data_bounds, method)
    return gamma_estimated['x']


def gamma_estimate(data, start_shape_vals=None, data_bounds=None, method="BFGS"):
    """
    Estimate the parameters shape and scale using maximum likelihood estimation.
    """
    if start_shape_vals is None:
        start_shape_vals = [0.5, 1.0, 1.5, 2.0, 2.5]

    data_mean = np.mean(data)
    best_result = None

    for shape_guess in start_shape_vals:
        scale_guess = data_mean / shape_guess  # Initial scale estimate
        start_params = [shape_guess, scale_guess]
        result = minimize(gamma_lnlike, np.array(start_params), args=(data, data_bounds), method=method)

        if best_result is None or result.fun < best_result.fun:
            best_result = result
            best_result.start_shape = shape_guess

    best_result.x = np.abs(best_result.x)  # Ensure parameters are positive
    return best_result


def shifted_gamma_cdf(x, shape, scale, shift):
    """Compute the CDF of a three-parameter shifted gamma distribution."""
    return np.clip(gamma.cdf(x, shape, loc=shift, scale=scale), 0, 1)


def shifted_gamma_lnlike(params, data, data_bounds=None):
    """Compute the negative log-likelihood of a shifted gamma model."""
    return _shifted_distribution_lnlike(params, np.asarray(data, dtype=float), gamma, data_bounds)


def shifted_gamma_estimate_x(data, start_shape_vals=None, data_bounds=None, method="L-BFGS-B"):
    """Return shifted gamma shape, scale, and shift estimates."""
    return shifted_gamma_estimate(data, start_shape_vals, data_bounds, method).x


def shifted_gamma_estimate(data, start_shape_vals=None, data_bounds=None, method="L-BFGS-B"):
    """Estimate shifted gamma parameters with constrained multi-start likelihood optimization."""
    if start_shape_vals is None:
        start_shape_vals = [0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    return _estimate_shifted_distribution(
        data,
        gamma,
        start_shape_vals,
        lambda shifted, shape: np.mean(shifted) / shape,
        data_bounds,
        method,
    )


def gamma_generate_data(n=100, shape=2.0, scale=100):
    """Generate random data from a Gamma distribution."""
    return gamma.rvs(shape, scale=scale, size=n)


def plot_gamma_fit(data, estimated_params):
    """Plot the histogram of the data and the fitted gamma distribution."""
    shape, scale = estimated_params
    x = np.linspace(min(data), max(data), 100)
    pdf_fitted = gamma.pdf(x, shape, scale=scale)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='b', label='Histogram')
    plt.plot(x, pdf_fitted, 'r-', lw=2, label='Fitted Gamma')
    plt.xlabel('Value')
    plt.ylabel('Density')

    plt.title(f'Gamma Fit\nShape: {shape:.4f}, Scale: {scale:.4f}')
    plt.legend()
    plt.show()
    plt.pause(0.001)  # 让 Python 继续执行后续代码
    plt.ioff()  # 关闭交互模式


def gammaRunDemo():
    data = gamma_generate_data(1000, 2.5, 120)
    # np.savetxt("gamma_data.txt", data, fmt="%.6f")
    # To read the data in R:
    # data <- read.table("gamma_data.txt", header=FALSE)[,1]
    fitResults = gamma_estimate(data)
    print(fitResults['x'] - [2.5, 120])

    # Plot the fitted distribution
    plot_gamma_fit(data, fitResults['x'])


"""
inverse gaussian distribution
"""


def inverse_gaussian_cdf(x, mu, lambda_):
    """
    Compute the CDF of the Inverse Gaussian distribution, ensuring values are within [0,1].
    """
    return np.clip(invgauss.cdf(x, mu=mu / lambda_, scale=lambda_), 0, 1)


def inverse_gaussian_lnlike(params, data, data_bounds=None):
    """
    Compute the log-likelihood of the data under an Inverse Gaussian model.
    """
    mu, lambda_ = params
    if mu <= 0 or lambda_ <= 0:
        return np.inf
    log_pdf_values = invgauss.logpdf(data, mu=mu / lambda_, scale=lambda_)
    if not np.all(np.isfinite(log_pdf_values)):
        return np.inf

    result = -np.sum(log_pdf_values)

    if data_bounds is not None:
        lower_bound, upper_bound = data_bounds
        if (np.min(data) < lower_bound) or (np.max(data) > upper_bound):
            raise ValueError("Likelihood cannot be computed if any data points are outside the bounds")
        retained_probability = (
            inverse_gaussian_cdf(upper_bound, mu, lambda_)
            - inverse_gaussian_cdf(lower_bound, mu, lambda_)
        )
        if not 0 < retained_probability <= 1:
            return np.inf
        result += len(data) * np.log(retained_probability)

    return result


def inverse_gaussian_estimate_x(data, start_vals=None, data_bounds=None, method="L-BFGS-B"):
    inverse_gaussian_estimated = inverse_gaussian_estimate(data, start_vals, data_bounds, method)
    return inverse_gaussian_estimated['x']


def inverse_gaussian_estimate(data, start_vals=None, data_bounds=None, method="L-BFGS-B"):
    """
    Estimate the parameters mu and lambda using maximum likelihood estimation.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 3 or np.any(data <= 0):
        raise ValueError("Inverse-Gaussian fitting requires at least three finite positive observations.")
    if start_vals is None:
        mean = np.mean(data)
        variance = np.var(data)
        start_vals = [(mean, mean ** 3 / max(variance, np.finfo(float).eps))]

    best_result = None
    data_scale = max(float(np.max(data)), 1.0)
    bounds = [(data_scale * 1e-8, data_scale * 100.0),
              (data_scale * 1e-8, data_scale ** 2 * 100.0)]

    for mu_guess, lambda_guess in start_vals:
        start_params = [mu_guess, lambda_guess]
        result = minimize(inverse_gaussian_lnlike, np.array(start_params), args=(data, data_bounds), method=method,
                          bounds=bounds, options={'maxiter': 1000})

        if result.success and np.isfinite(result.fun) and (best_result is None or result.fun < best_result.fun):
            best_result = result

    if best_result is None:
        raise RuntimeError("No finite inverse-Gaussian fit could be found.")
    return best_result


def inverse_gaussian_generate_data(n=100, mu=100, lambda_=200):
    """Generate random data from an Inverse Gaussian distribution."""
    return invgauss.rvs(mu=mu / lambda_, scale=lambda_, size=n)


def plot_inverse_gaussian_fit(data, estimated_params):
    """Plot the histogram of the data and the fitted inverse Gaussian distribution."""
    mu, lambda_ = estimated_params
    x = np.linspace(min(data), max(data), 100)
    pdf_fitted = invgauss.pdf(x, mu=mu / lambda_, scale=lambda_)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='b', label='Histogram')
    plt.plot(x, pdf_fitted, 'r-', lw=2, label='Fitted Inverse Gaussian')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Inverse Gaussian Fit\nMu: {mu:.4f}, Lambda: {lambda_:.4f}')
    plt.legend()
    plt.show()
    plt.pause(0.001)  # 让 Python 继续执行后续代码
    plt.ioff()  # 关闭交互模式


def inverseGaussianRunDemo():
    data = inverse_gaussian_generate_data(1000, 500, 200)
    np.savetxt("inverse_gaussian_data.txt", data, fmt="%.6f")
    # To read the data in R:
    # data <- read.table("inverse_gaussian_data.txt", header=FALSE)[,1]
    fitResults = inverse_gaussian_estimate(data)
    print(fitResults['x'] - [500, 200])

    # Plot the fitted distribution
    plot_inverse_gaussian_fit(data, fitResults['x'])


"""
shifted inverse gaussian distribution
"""


def shifted_inverse_gaussian_cdf(x, mu, lambda_, shift):
    """
    Compute the CDF of the Shifted Inverse Gaussian distribution, ensuring values are within [0,1].
    """
    return np.clip(invgauss.cdf(x, mu=mu / lambda_, loc=shift, scale=lambda_), 0, 1)


def shifted_inverse_gaussian_lnlike(params, data, data_bounds=None):
    """
    Compute the log-likelihood of the data under a Shifted Inverse Gaussian model.
    """
    mu, lambda_, shift = params
    if mu <= 0 or lambda_ <= 0 or shift < 0 or np.any(data <= shift):
        return np.inf
    log_pdf_values = invgauss.logpdf(data, mu=mu / lambda_, loc=shift, scale=lambda_)
    if not np.all(np.isfinite(log_pdf_values)):
        return np.inf

    result = -np.sum(log_pdf_values)

    if data_bounds is not None:
        lower_bound, upper_bound = data_bounds
        if (np.min(data) < lower_bound) or (np.max(data) > upper_bound):
            raise ValueError("Likelihood cannot be computed if any data points are outside the bounds")
        retained_probability = (
            shifted_inverse_gaussian_cdf(upper_bound, mu, lambda_, shift)
            - shifted_inverse_gaussian_cdf(lower_bound, mu, lambda_, shift)
        )
        if not 0 < retained_probability <= 1:
            return np.inf
        result += len(data) * np.log(retained_probability)

    return result


def shifted_inverse_gaussian_estimate_x(data, start_vals=None, data_bounds=None, method="L-BFGS-B"):
    shifted_estimated = shifted_inverse_gaussian_estimate(data, start_vals, data_bounds, method)
    return shifted_estimated['x']


def shifted_inverse_gaussian_estimate(data, start_vals=None, data_bounds=None, method="L-BFGS-B"):
    """
    Estimate the parameters mu, lambda, and shift using maximum likelihood estimation.
    """
    data = _prepare_shifted_rt_data(data)
    shift_upper = _shift_upper_bound(data)
    if start_vals is None:
        start_vals = []
        for shift_guess in np.unique(np.array([0.0, 0.25, 0.5, 0.75]) * shift_upper):
            shifted_data = data - shift_guess
            mean = np.mean(shifted_data)
            variance = np.var(shifted_data)
            lambda_guess = mean ** 3 / max(variance, np.finfo(float).eps)
            start_vals.append((mean, lambda_guess, shift_guess))

    best_result = None
    data_scale = max(float(np.max(data)), 1.0)
    bounds = [(data_scale * 1e-8, data_scale * 100.0),
              (data_scale * 1e-8, data_scale ** 2 * 100.0),
              (0.0, shift_upper)]

    for mu_guess, lambda_guess, shift_guess in start_vals:
        start_params = [mu_guess, lambda_guess, shift_guess]
        result = minimize(shifted_inverse_gaussian_lnlike, np.array(start_params), args=(data, data_bounds),
                          method=method, bounds=bounds, options={'maxiter': 1000})

        if result.success and np.isfinite(result.fun) and (best_result is None or result.fun < best_result.fun):
            best_result = result

    if best_result is None:
        raise RuntimeError("No finite shifted inverse-Gaussian fit could be found.")
    return best_result


def generate_shifted_inverse_gaussian_data(n=100, mu=100, lambda_=200, shift=50):
    """Generate random data from a Shifted Inverse Gaussian distribution."""
    return invgauss.rvs(mu=mu / lambda_, loc=shift, scale=lambda_, size=n)


def plot_shifted_inverse_gaussian_fit(data, estimated_params):
    """Plot the histogram of the data and the fitted Shifted Inverse Gaussian distribution."""
    mu, lambda_, shift = estimated_params
    x = np.linspace(min(data), max(data), 100)
    pdf_fitted = invgauss.pdf(x, mu=mu / lambda_, loc=shift, scale=lambda_)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='b', label='Histogram')
    plt.plot(x, pdf_fitted, 'r-', lw=2, label='Fitted Shifted Inverse Gaussian')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Shifted Inverse Gaussian Fit\nMu: {mu:.4f}, Lambda: {lambda_:.4f}, Shift: {shift:.4f}')
    plt.legend()
    plt.draw()  # 让 Matplotlib 绘制图像但不阻塞
    plt.show()


def shiftedInverseGaussianDemo():
    data = generate_shifted_inverse_gaussian_data(1000, 500, 200, 50)
    np.savetxt("shifted_inverse_gaussian_data.txt", data, fmt="%.6f")
    # To read the data in R:
    # data <- read.table("shifted_inverse_gaussian_data.txt", header=FALSE)[,1]
    fitResults = shifted_inverse_gaussian_estimate(data)
    # Plot the fitted distribution
    plot_shifted_inverse_gaussian_fit(data, fitResults['x'])

    print(fitResults['x'] - [500, 200, 50])


FIT_DIAGNOSTIC_NAMES = [
    'N valid', 'Converged', 'Parameter boundary', 'Shift warning',
    'Log-likelihood', 'AIC', 'BIC',
]


def _rt_distribution_estimator(distribution_name):
    """Return the full optimizer for a named RT distribution."""
    estimators = {
        'Gamma (k, θ)': gamma_estimate, 'Shifted Gamma (k, θ, shift)': shifted_gamma_estimate,
        'Wald (m, a)': lambda data: wald_estimate(data, shift=False),
        'Ex-Wald (m, a, τ)': ex_wald_estimate,
        'Shifted Wald (m, a, shift)': lambda data: wald_estimate(data, shift=True),
        'Ex-Gaussian (μ, σ, τ)': ex_gaussian_estimate,
        'Inv-Gaussian (μ, λ)': inverse_gaussian_estimate,
        'Shifted Inv-Gaussian (μ, λ, shift)': shifted_inverse_gaussian_estimate,
        'Weibull (k, θ)': weibull_estimate, 'Shifted Weibull (k, θ, shift)': shifted_weibull_estimate,
        'LogNormal (k, θ)': log_normal_estimate, 'Shifted LogNormal (k, θ, shift)': shifted_log_normal_estimate,
    }
    if distribution_name not in estimators:
        raise ValueError(f"Unknown RT distribution: {distribution_name}")
    return estimators[distribution_name]


def _rt_distribution_parameter_names(distribution_name):
    """Return parameter labels for a named RT distribution."""
    names = {
        'Gamma (k, θ)': ['shape (k)', 'scale (θ)'],
        'Shifted Gamma (k, θ, shift)': ['shape (k)', 'scale (θ)', 'shift'],
        'Wald (m, a)': ['mean rate (m)', 'response threshold (a)'],
        'Ex-Wald (m, a, τ)': ['mean rate (m)', 'response threshold (a)', 'τ'],
        'Shifted Wald (m, a, shift)': ['mean rate (m)', 'response threshold (a)', 'shift'],
        'Ex-Gaussian (μ, σ, τ)': ['mu (μ)', 'sigma (σ)', 'tau (τ)'],
        'Inv-Gaussian (μ, λ)': ['mu (μ)', 'lambda (λ)'],
        'Shifted Inv-Gaussian (μ, λ, shift)': ['mu (μ)', 'lambda (λ)', 'shift'],
        'Weibull (k, θ)': ['shape (k)', 'scale (θ)'],
        'Shifted Weibull (k, θ, shift)': ['shape (k)', 'scale (θ)', 'shift'],
        'LogNormal (k, θ)': ['shape (k)', 'scale (θ)'],
        'Shifted LogNormal (k, θ, shift)': ['shape (k)', 'scale (θ)', 'shift'],
    }
    if distribution_name not in names:
        raise ValueError(f"Unknown RT distribution: {distribution_name}")
    return names[distribution_name]


def _rt_distribution_bounds(distribution_name, data):
    """Return diagnostic parameter bounds used by a named fit."""
    data_scale = max(float(np.max(data)), float(np.ptp(data)), 1.0)
    positive = [(1e-8, None), (1e-8, None)]
    if distribution_name == 'Weibull (k, θ)':
        return [(1e-4, 100.0), (data_scale * 1e-8, data_scale * 100.0)]
    if distribution_name in {'Gamma (k, θ)', 'LogNormal (k, θ)'}:
        return [(0.0, None), (0.0, None)]
    if distribution_name in {'Shifted Gamma (k, θ, shift)', 'Shifted Weibull (k, θ, shift)', 'Shifted LogNormal (k, θ, shift)'}:
        return [(1e-4, 100.0), (data_scale * 1e-8, data_scale * 100.0), (0.0, _shift_upper_bound(data))]
    if distribution_name == 'Wald (m, a)':
        return positive
    if distribution_name == 'Shifted Wald (m, a, shift)':
        return positive + [(None, float(np.min(data)))]
    if distribution_name in {'Ex-Wald (m, a, τ)', 'Ex-Gaussian (μ, σ, τ)'}:
        return positive + [(1e-8, None)]
    inverse_bounds = [(data_scale * 1e-8, data_scale * 100.0), (data_scale * 1e-8, data_scale ** 2 * 100.0)]
    if distribution_name == 'Shifted Inv-Gaussian (μ, λ, shift)':
        return inverse_bounds + [(0.0, _shift_upper_bound(data))]
    return inverse_bounds


def _near_bound(value, lower, upper):
    """Return whether a fitted value is numerically close to a finite bound."""
    finite_bounds = [bound for bound in (lower, upper) if bound is not None and np.isfinite(bound)]
    if not finite_bounds:
        return False
    span = abs(upper - lower) if lower is not None and upper is not None else max(abs(value), *(abs(x) for x in finite_bounds), 1.0)
    tolerance = max(1e-7 * span, 1e-8)
    return any(abs(value - bound) <= tolerance for bound in finite_bounds)


def fit_rt_distribution(data, distribution_name):
    """Fit an RT distribution and return parameters, diagnostics, and plot data."""
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    parameter_names = _rt_distribution_parameter_names(distribution_name)
    n_valid = int(values.size)
    failure = {'distribution': distribution_name, 'data': values,
               'parameters': np.full(len(parameter_names), np.nan), 'n_valid': n_valid,
               'converged': False, 'parameter_boundary': 'NA', 'boundary_details': 'Not assessed',
               'shift_warning': 'NA', 'shift_warning_details': 'Not assessed',
               'log_likelihood': np.nan, 'aic': np.nan, 'bic': np.nan, 'message': ''}
    try:
        if n_valid < 3:
            raise ValueError('At least three finite observations are required.')
        result = _rt_distribution_estimator(distribution_name)(values)
        parameters = np.asarray(result.x, dtype=float)
        negative_log_likelihood = float(result.fun)
        converged = bool(
            result.success and np.isfinite(negative_log_likelihood) and negative_log_likelihood < 1e100
            and np.all(np.isfinite(parameters)))
        bounds = _rt_distribution_bounds(distribution_name, values)
        boundary_names = [name for name, value, (lower, upper) in zip(parameter_names, parameters, bounds)
                          if _near_bound(value, lower, upper)]
        shift_warning = 'No'
        shift_warning_details = 'None'
        if 'shift' in parameter_names:
            shift = float(parameters[parameter_names.index('shift')])
            minimum = float(np.min(values))
            regularized_gap = minimum - _shift_upper_bound(values)
            if minimum - shift <= max(2.0 * regularized_gap, np.finfo(float).eps * max(1.0, minimum) * 100):
                shift_warning = 'Near minimum RT'
                shift_warning_details = f'shift={shift:.6g}, minimum RT={minimum:.6g}, gap={minimum - shift:.6g}'
                if 'shift' not in boundary_names:
                    boundary_names.append('shift')
        log_likelihood = -negative_log_likelihood
        parameter_count = len(parameters)
        failure.update({'parameters': parameters, 'converged': converged,
                        'parameter_boundary': 'Yes' if boundary_names else 'No',
                        'boundary_details': ', '.join(boundary_names) if boundary_names else 'None',
                        'shift_warning': 'Yes' if shift_warning != 'No' else 'No',
                        'shift_warning_details': shift_warning_details, 'log_likelihood': log_likelihood,
                        'aic': 2 * parameter_count - 2 * log_likelihood,
                        'bic': parameter_count * np.log(n_valid) - 2 * log_likelihood,
                        'message': str(result.message)})
    except Exception as error:
        failure['message'] = str(error)
    return failure


def fit_rt_distribution_values(data, distribution_name, log_diagnostics=False):
    """Return a table-ready vector containing parameters and fit diagnostics."""
    fit = fit_rt_distribution(data, distribution_name)
    if log_diagnostics:
        print(
            f"Fit diagnostics: Converged={'Yes' if fit['converged'] else 'No'}; "
            f"N valid={fit['n_valid']}; Parameter boundary={fit['parameter_boundary']} "
            f"({fit['boundary_details']}); Shift warning={fit['shift_warning']} "
            f"({fit['shift_warning_details']}); optimizer={fit['message']}"
        )
    diagnostics = [fit['n_valid'], 'Yes' if fit['converged'] else 'No', fit['parameter_boundary'],
                   fit['shift_warning'], fit['log_likelihood'], fit['aic'], fit['bic']]
    return np.asarray(list(fit['parameters']) + diagnostics, dtype=object)


def rt_distribution_pdf(distribution_name, x, parameters):
    """Evaluate the fitted PDF for a named RT distribution."""
    p = np.asarray(parameters, dtype=float)
    functions = {
        'Gamma (k, θ)': lambda v: gamma.pdf(v, p[0], scale=p[1]),
        'Shifted Gamma (k, θ, shift)': lambda v: gamma.pdf(v, p[0], loc=p[2], scale=p[1]),
        'Wald (m, a)': lambda v: wald_pdf(v, p[0], p[1]), 'Ex-Wald (m, a, τ)': lambda v: ex_wald_pdf(v, p[0], p[1], p[2]),
        'Shifted Wald (m, a, shift)': lambda v: wald_pdf(v, p[0], p[1], p[2]),
        'Ex-Gaussian (μ, σ, τ)': lambda v: ex_gaussian_pdf(v, p[0], p[1], p[2]),
        'Inv-Gaussian (μ, λ)': lambda v: invgauss.pdf(v, mu=p[0] / p[1], scale=p[1]),
        'Shifted Inv-Gaussian (μ, λ, shift)': lambda v: invgauss.pdf(v, mu=p[0] / p[1], loc=p[2], scale=p[1]),
        'Weibull (k, θ)': lambda v: np.exp(_weibull_logpdf(v, p[0], p[1])),
        'Shifted Weibull (k, θ, shift)': lambda v: np.exp(_weibull_logpdf(v, p[0], p[1], p[2])),
        'LogNormal (k, θ)': lambda v: lognorm.pdf(v, p[0], scale=p[1]),
        'Shifted LogNormal (k, θ, shift)': lambda v: lognorm.pdf(v, p[0], loc=p[2], scale=p[1]),
    }
    return np.asarray(functions[distribution_name](np.asarray(x, dtype=float)), dtype=float)


def rt_distribution_cdf(distribution_name, x, parameters):
    """Evaluate the fitted CDF for a named RT distribution."""
    p = np.asarray(parameters, dtype=float)
    functions = {
        'Gamma (k, θ)': lambda v: gamma.cdf(v, p[0], scale=p[1]),
        'Shifted Gamma (k, θ, shift)': lambda v: gamma.cdf(v, p[0], loc=p[2], scale=p[1]),
        'Wald (m, a)': lambda v: wald_cdf(v, p[0], p[1]), 'Ex-Wald (m, a, τ)': lambda v: ex_wald_cdf(v, p[0], p[1], p[2]),
        'Shifted Wald (m, a, shift)': lambda v: wald_cdf(v, p[0], p[1], p[2]),
        'Ex-Gaussian (μ, σ, τ)': lambda v: ex_gaussian_cdf(v, p[0], p[1], p[2]),
        'Inv-Gaussian (μ, λ)': lambda v: inverse_gaussian_cdf(v, p[0], p[1]),
        'Shifted Inv-Gaussian (μ, λ, shift)': lambda v: shifted_inverse_gaussian_cdf(v, p[0], p[1], p[2]),
        'Weibull (k, θ)': lambda v: _weibull_cdf(v, p[0], p[1]),
        'Shifted Weibull (k, θ, shift)': lambda v: _weibull_cdf(v, p[0], p[1], p[2]),
        'LogNormal (k, θ)': lambda v: lognorm.cdf(v, p[0], scale=p[1]),
        'Shifted LogNormal (k, θ, shift)': lambda v: lognorm.cdf(v, p[0], loc=p[2], scale=p[1]),
    }
    values = np.asarray(x, dtype=float)
    result = np.zeros_like(values)
    support = values > (p[2] if 'Shifted ' in distribution_name else 0.0)
    if distribution_name == 'Ex-Gaussian (μ, σ, τ)':
        support = np.ones_like(values, dtype=bool)
    result[support] = functions[distribution_name](values[support])
    return np.clip(result, 0.0, 1.0)


def CDF_pooling_main(rt_df, sub_vars_list, cond_vars_list, rt_var_name,
                     min_trials_required=50,
                     rt_bounds=None,
                     method="BFGS"):
    """
    Main function for CDF pooling analysis using multi-column composite grouping,
    without modifying original DataFrame columns.

    Parameters:
    - rt_df: DataFrame with RT data.
    - sub_vars_list: List of column names that define subject identity.
    - cond_vars_list: List of column names that define condition identity.
    - rt_var_name: Name of the RT column.
    - min_trials_required: Minimum number of trials required for estimation.
    - rt_bounds: Optional bounds on RT.
    - method: Optimization method.
    """

    # if start_prop_var_in_tau is None:
    #     start_prop_var_in_tau = [0.1, 0.3, 0.5, 0.7]

    rt_cdf_var_name = f"{rt_var_name}_cdf"

    rt_df[rt_cdf_var_name] = np.nan

    group_keys = sub_vars_list + cond_vars_list

    if group_keys:
        unique_combinations = rt_df.loc[:, group_keys].drop_duplicates()
    else:
        unique_combinations = pd.DataFrame({'no_var': [1]})

    for _, combo in unique_combinations.iterrows():
        # 构建行筛选条件
        condition = np.ones(len(rt_df), dtype=bool)
        for col in group_keys:
            condition &= (rt_df[col] == combo[col])

        selected_rts = rt_df.loc[condition, rt_var_name].values

        if len(selected_rts) < min_trials_required:
            raise RuntimeError(f"Failed estimation for {dict(combo)}, insufficient trials: {len(selected_rts)}")
        else:
            est_result = ex_gaussian_estimate_x(selected_rts, 0.8, method)

            mu, sigma, tau = est_result
            rt_df.loc[condition, rt_cdf_var_name] = ex_gaussian_cdf(selected_rts, mu, sigma, tau)
    return rt_df


"""
demos only for debug only
"""
# shiftedInverseGaussianDemo()
# inverseGaussianRunDemo()
# gammaRunDemo()
# logNormalRunDemo()
# weibullRunDemo()
# ExWaldRunDemo()
# WaldRunDemo()
# exGaussianRunDemo()
