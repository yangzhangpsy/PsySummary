import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm, chi2, expon, weibull_min, lognorm, invgauss, gamma
from scipy.optimize import minimize

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
    w = w - s
    sqrtw = np.sqrt(w)
    k1 = (m * w - a) / sqrtw
    k2 = (m * w + a) / sqrtw

    p1 = np.exp(2 * a * m)
    p2 = norm.cdf(-k2)
    bad = (p1 == np.inf) | (p2 == 0)
    p = p1 * p2

    p[bad] = np.exp(-(k1[bad] ** 2) / 2 - 0.94 / (k2[bad] ** 2)) / (k2[bad] * np.sqrt(2 * np.pi))
    return p + norm.cdf(k1)


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
    u, v = complex_error_function_real_imag(y, x)
    return np.exp(y ** 2 - x ** 2) * (np.cos(2 * x * y) * (1 - u) + np.sin(2 * x * y) * v)


# Ex-Wald PDF
def ex_wald_pdf(r, m, a, t):
    k = m ** 2 - (2 / t)
    if k < 0:
        density = np.exp(m * a - (a ** 2) / (2 * r) - r * (m ** 2) / 2) * exwald_w_function_real_part(
            np.sqrt(-r * k / 2), a / np.sqrt(2 * r)) / t
    else:
        k = np.sqrt(k)
        density = wald_cdf(r, k, a) * np.exp(a * (m - k) - (r / t)) / t
    return density


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
    return wald_cdf(r, m, a) - t * ex_wald_pdf(r, m, a, t)


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
    t = p * np.std(x)
    m = np.sqrt((np.mean(x) - t) / (np.var(x) - t ** 2))
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
    density = ex_wald_pdf(x, p[0], p[1], p[2])
    return -np.sum(np.log(density[density > 0]))


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


def weibull_cdf(x, shape, scale):
    """
    Compute the CDF of the Weibull distribution, ensuring values are within [0,1].
    """
    return np.clip(weibull_min.cdf(x, shape, scale=scale), 0, 1)


def weibull_lnlike(params, data, data_bounds=None):
    """
    Compute the log-likelihood of the data under a Weibull model.
    """
    shape, scale = np.abs(params)  # Ensure positive parameters
    pdf_values = weibull_min.pdf(data, shape, scale=scale)
    pdf_values = np.clip(pdf_values, 1e-10, np.inf)  # Avoid log(0)

    if data_bounds is not None:
        lower_bound, upper_bound = data_bounds
        if (np.min(data) < lower_bound) or (np.max(data) > upper_bound):
            raise ValueError("Likelihood cannot be computed if any data points are outside the bounds")
        lost_prob = weibull_cdf(lower_bound, shape, scale) + (1 - weibull_cdf(upper_bound, shape, scale))
        pdf_values /= (1 - lost_prob)

    return -np.sum(np.log(pdf_values))


def weibull_estimate_x(data, start_shape_vals=None, data_bounds=None, method="BFGS"):
    weibull_estimated = weibull_estimate(data, start_shape_vals, data_bounds, method)
    return weibull_estimated['x']


def weibull_estimate(data, start_shape_vals=None, data_bounds=None, method="BFGS"):
    """
    Estimate the parameters shape and scale using maximum likelihood estimation.
    """
    if start_shape_vals is None:
        start_shape_vals = [0.5, 1.0, 1.5, 2.0]

    data_mean = np.mean(data)
    best_result = None

    for shape_guess in start_shape_vals:
        scale_guess = data_mean / shape_guess  # Initial scale estimate
        start_params = [shape_guess, scale_guess]
        result = minimize(weibull_lnlike, np.array(start_params), args=(data, data_bounds), method=method)

        if best_result is None or result.fun < best_result.fun:
            best_result = result
            best_result.start_shape = shape_guess

    best_result.x = np.abs(best_result.x)  # Ensure parameters are positive
    return best_result


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
    return np.clip(invgauss.cdf(x, mu=lambda_ / mu, scale=mu), 0, 1)


def inverse_gaussian_lnlike(params, data, data_bounds=None):
    """
    Compute the log-likelihood of the data under an Inverse Gaussian model.
    """
    mu, lambda_ = np.abs(params)  # Ensure positive parameters
    pdf_values = invgauss.pdf(data, mu=lambda_ / mu, scale=mu)
    pdf_values = np.clip(pdf_values, 1e-10, np.inf)  # Avoid log(0)

    if data_bounds is not None:
        lower_bound, upper_bound = data_bounds
        if (np.min(data) < lower_bound) or (np.max(data) > upper_bound):
            raise ValueError("Likelihood cannot be computed if any data points are outside the bounds")
        lost_prob = inverse_gaussian_cdf(lower_bound, mu, lambda_) + (
                1 - inverse_gaussian_cdf(upper_bound, mu, lambda_))
        pdf_values /= (1 - lost_prob)

    return -np.sum(np.log(pdf_values))


def inverse_gaussian_estimate_x(data, start_vals=None, data_bounds=None, method="BFGS"):
    inverse_gaussian_estimated = inverse_gaussian_estimate(data, start_vals, data_bounds, method)
    return inverse_gaussian_estimated['x']


def inverse_gaussian_estimate(data, start_vals=None, data_bounds=None, method="BFGS"):
    """
    Estimate the parameters mu and lambda using maximum likelihood estimation.
    """
    if start_vals is None:
        start_vals = [(np.mean(data), np.var(data))]

    best_result = None

    for mu_guess, lambda_guess in start_vals:
        start_params = [mu_guess, lambda_guess]
        result = minimize(inverse_gaussian_lnlike, np.array(start_params), args=(data, data_bounds), method=method)

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    best_result.x = np.abs(best_result.x)  # Ensure parameters are positive
    return best_result


def inverse_gaussian_generate_data(n=100, mu=100, lambda_=200):
    """Generate random data from an Inverse Gaussian distribution."""
    return invgauss.rvs(mu=lambda_ / mu, scale=mu, size=n)


def plot_inverse_gaussian_fit(data, estimated_params):
    """Plot the histogram of the data and the fitted inverse Gaussian distribution."""
    mu, lambda_ = estimated_params
    x = np.linspace(min(data), max(data), 100)
    pdf_fitted = invgauss.pdf(x, mu=lambda_ / mu, scale=mu)

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
    return np.clip(invgauss.cdf(x - shift, mu=lambda_ / mu, scale=mu), 0, 1)


def shifted_inverse_gaussian_lnlike(params, data, data_bounds=None):
    """
    Compute the log-likelihood of the data under a Shifted Inverse Gaussian model.
    """
    mu, lambda_, shift = np.abs(params)  # Ensure positive parameters
    pdf_values = invgauss.pdf(data - shift, mu=lambda_ / mu, scale=mu)
    pdf_values = np.clip(pdf_values, 1e-10, np.inf)  # Avoid log(0)

    if data_bounds is not None:
        lower_bound, upper_bound = data_bounds
        if (np.min(data) < lower_bound) or (np.max(data) > upper_bound):
            raise ValueError("Likelihood cannot be computed if any data points are outside the bounds")
        lost_prob = shifted_inverse_gaussian_cdf(lower_bound, mu, lambda_, shift) + (
                1 - shifted_inverse_gaussian_cdf(upper_bound, mu, lambda_, shift))
        pdf_values /= (1 - lost_prob)

    return -np.sum(np.log(pdf_values))


def shifted_inverse_gaussian_estimate_x(data, start_vals=None, data_bounds=None, method="BFGS"):
    shifted_estimated = shifted_inverse_gaussian_estimate(data, start_vals, data_bounds, method)
    return shifted_estimated['x']


def shifted_inverse_gaussian_estimate(data, start_vals=None, data_bounds=None, method="BFGS"):
    """
    Estimate the parameters mu, lambda, and shift using maximum likelihood estimation.
    """
    if start_vals is None:
        start_vals = [(np.mean(data), np.var(data), np.min(data) - 1)]

    best_result = None

    for mu_guess, lambda_guess, shift_guess in start_vals:
        start_params = [mu_guess, lambda_guess, shift_guess]
        result = minimize(shifted_inverse_gaussian_lnlike, np.array(start_params), args=(data, data_bounds),
                          method=method)

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    best_result.x = np.abs(best_result.x)  # Ensure parameters are positive
    return best_result


def generate_shifted_inverse_gaussian_data(n=100, mu=100, lambda_=200, shift=50):
    """Generate random data from a Shifted Inverse Gaussian distribution."""
    return invgauss.rvs(mu=lambda_ / mu, scale=mu, size=n) + shift


def plot_shifted_inverse_gaussian_fit(data, estimated_params):
    """Plot the histogram of the data and the fitted Shifted Inverse Gaussian distribution."""
    mu, lambda_, shift = estimated_params
    x = np.linspace(min(data), max(data), 100)
    pdf_fitted = invgauss.pdf(x - shift, mu=lambda_ / mu, scale=mu)

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
