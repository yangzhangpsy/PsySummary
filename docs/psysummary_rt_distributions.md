# PsySummary reaction-time distributions

PsySummary provides descriptive distribution fits for positive reaction-time (RT) observations. Distribution
parameters summarize an empirical RT distribution; they should not automatically be interpreted as distinct cognitive
processes.

## Shifted positive distributions

Gamma, Weibull, and lognormal models are available in both unshifted and shifted forms. Their shifted forms use

`RT = shift + X`, where `X > 0`.

The shift parameter therefore controls the leading edge, or earliest supported response time, independently of the
shape and scale that describe the remainder of the distribution. The unshifted models are retained for compatibility
and as simpler two-parameter alternatives in which `shift = 0`.

| Option | Reported parameters |
| --- | --- |
| Gamma | shape (`k`), scale (`θ`) |
| Shifted Gamma | shape (`k`), scale (`θ`), shift |
| Weibull | shape (`k`), scale (`θ`) |
| Shifted Weibull | shape (`k`), scale (`θ`), shift |
| LogNormal | log-space standard deviation (`k`), scale (`θ = exp(μlog)`) |
| Shifted LogNormal | log-space standard deviation (`k`), scale (`θ = exp(μlog)`), shift |

Shifted fits require at least four finite, strictly positive observations. PsySummary uses constrained, multi-start
maximum-likelihood optimization. The shift is constrained to be non-negative and below the smallest observed RT. A
small margin based on the RT scale and observed resolution is retained below the minimum to avoid the singular
likelihood solutions that can occur in unrestricted three-parameter fits.

## Gaussian-related names

`Shifted Inv-Gaussian` means the shifted **inverse Gaussian**, not a shifted ordinary Gaussian. Its parameters are the
inverse-Gaussian mean (`μ`), shape (`λ`), and shift. A separate shift would be redundant for an ordinary Gaussian
because translation is already represented by its mean, but it is not redundant for an inverse Gaussian.

`Ex-Gaussian` is the convolution of Gaussian and exponential components and is also distinct from an ordinary shifted
Gaussian.

## Interpretation and comparison

The shift is a distributional lower-bound parameter, not necessarily a pure estimate of nondecision time. Estimates
can be unstable when there are few observations or when the fitted shift lies near the smallest RT. Users should
inspect the data and fitted distribution and should account for the additional parameter when comparing shifted and
unshifted models.

## Fit diagnostics

Each fitted group reports the number of finite observations used (`N valid`), convergence (`Yes` or `No`), whether a
parameter reached or approached an optimizer boundary (`Yes` or `No`), a separate shift warning (`Yes` or `No`), the
maximized log-likelihood, AIC, and BIC. A failed group produces a diagnostic row with unavailable estimates instead of
aborting all other groups.

Detailed boundary names, the distance between a fitted shift and the minimum observed RT, and the optimizer message
are written to the output log. Each group's `Converged` cell includes a `View Fit` button that opens its diagnostic
view with one click. Double-clicking any other cell belonging to a distribution fit provides the same shortcut. The
activated group is selected automatically; additional groups can be checked in
the selector to overlay several conditions. Conditions follow MATLAB's default plot color order. Each condition's
histogram, fitted density, empirical CDF, and fitted CDF share one color, while empirical CDFs are dashed and fitted
CDFs are solid. A converged optimizer is necessary but does not by itself establish that a model fits adequately; the
plotted discrepancies and information criteria should also be considered.
