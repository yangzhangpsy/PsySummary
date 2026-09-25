import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.stats import gamma, invgauss, lognorm, weibull_min

from app.lib.fitRTsDistThread import FitRTsDistThread
from app.rtDist import (
    FIT_DIAGNOSTIC_NAMES,
    fit_rt_distribution,
    fit_rt_distribution_values,
    inverse_gaussian_cdf,
    inverse_gaussian_estimate_x,
    shifted_gamma_cdf,
    shifted_gamma_estimate_x,
    shifted_log_normal_cdf,
    shifted_log_normal_estimate_x,
    shifted_weibull_cdf,
    shifted_weibull_estimate_x,
)
from app import rtDist


class ShiftedRTDistributionTests(unittest.TestCase):
    def test_shifted_distribution_cdfs_respect_support(self):
        self.assertEqual(shifted_gamma_cdf(199.0, 3.0, 50.0, 200.0), 0.0)
        self.assertEqual(shifted_weibull_cdf(199.0, 2.0, 100.0, 200.0), 0.0)
        self.assertEqual(shifted_log_normal_cdf(199.0, 0.5, 250.0, 200.0), 0.0)

    def test_shifted_gamma_parameter_recovery(self):
        rng = np.random.default_rng(1201)
        data = gamma.rvs(3.0, loc=250.0, scale=60.0, size=1000, random_state=rng)
        shape, scale, shift = shifted_gamma_estimate_x(data)
        self.assertAlmostEqual(shape, 3.0, delta=0.8)
        self.assertAlmostEqual(scale, 60.0, delta=18.0)
        self.assertAlmostEqual(shift, 250.0, delta=25.0)

    def test_shifted_lognormal_parameter_recovery(self):
        rng = np.random.default_rng(1202)
        data = lognorm.rvs(0.45, loc=220.0, scale=280.0, size=1000, random_state=rng)
        shape, scale, shift = shifted_log_normal_estimate_x(data)
        self.assertAlmostEqual(shape, 0.45, delta=0.12)
        self.assertAlmostEqual(scale, 280.0, delta=45.0)
        self.assertAlmostEqual(shift, 220.0, delta=35.0)

    def test_shifted_weibull_parameter_recovery(self):
        rng = np.random.default_rng(1203)
        data = weibull_min.rvs(2.2, loc=240.0, scale=300.0, size=1000, random_state=rng)
        shape, scale, shift = shifted_weibull_estimate_x(data)
        self.assertAlmostEqual(shape, 2.2, delta=0.5)
        self.assertAlmostEqual(scale, 300.0, delta=40.0)
        self.assertAlmostEqual(shift, 240.0, delta=30.0)

    def test_inverse_gaussian_uses_mean_shape_parameterization(self):
        mu = 500.0
        lambda_ = 900.0
        rng = np.random.default_rng(1204)
        data = invgauss.rvs(mu / lambda_, scale=lambda_, size=3000, random_state=rng)
        estimated_mu, estimated_lambda = inverse_gaussian_estimate_x(data)
        self.assertAlmostEqual(estimated_mu, mu, delta=30.0)
        self.assertAlmostEqual(estimated_lambda, lambda_, delta=160.0)
        expected = invgauss.cdf(550.0, mu / lambda_, scale=lambda_)
        self.assertAlmostEqual(inverse_gaussian_cdf(550.0, mu, lambda_), expected, places=12)

    def test_fit_thread_exposes_shifted_options_and_three_parameters(self):
        expected_options = {
            'Shifted Gamma (k, θ, shift)',
            'Shifted Weibull (k, θ, shift)',
            'Shifted LogNormal (k, θ, shift)',
        }
        self.assertTrue(expected_options.issubset(FitRTsDistThread.DISTRIBUTION_MAP))
        for option in expected_options:
            parameter_names, _estimate = FitRTsDistThread.DISTRIBUTION_MAP[option]
            self.assertEqual(len(parameter_names), 3)
            self.assertEqual(parameter_names[-1], 'shift')

    def test_fit_diagnostics_include_information_criteria_and_valid_count(self):
        rng = np.random.default_rng(1205)
        data = gamma.rvs(3.0, scale=60.0, size=300, random_state=rng)
        fit = fit_rt_distribution(data, 'Gamma (k, θ)')
        values = fit_rt_distribution_values(data, 'Gamma (k, θ)')

        self.assertEqual(fit['n_valid'], 300)
        self.assertIsInstance(fit['converged'], bool)
        self.assertAlmostEqual(fit['aic'], 2 * 2 - 2 * fit['log_likelihood'])
        self.assertAlmostEqual(fit['bic'], 2 * np.log(300) - 2 * fit['log_likelihood'])
        self.assertEqual(len(values), 2 + len(FIT_DIAGNOSTIC_NAMES))
        self.assertIn(values[3], {'Yes', 'No'})

    def test_failed_fit_returns_a_diagnostic_row_instead_of_raising(self):
        fit = fit_rt_distribution([250.0, np.nan], 'Gamma (k, θ)')
        self.assertEqual(fit['n_valid'], 1)
        self.assertFalse(fit['converged'])
        self.assertEqual(fit['parameter_boundary'], 'NA')
        self.assertIn('At least three', fit['message'])

    def test_shift_near_minimum_is_reported_as_yes_with_log_details(self):
        fake_result = SimpleNamespace(
            x=np.array([2.0, 30.0, 89.9]), fun=20.0, success=True, message='synthetic fit')
        with patch.object(rtDist, '_rt_distribution_estimator',
                          return_value=lambda _data: fake_result):
            fit = fit_rt_distribution(
                [100.0, 120.0, 140.0, 160.0], 'Shifted Gamma (k, θ, shift)')

        self.assertEqual(fit['shift_warning'], 'Yes')
        self.assertEqual(fit['parameter_boundary'], 'Yes')
        self.assertIn('minimum RT=100', fit['shift_warning_details'])


if __name__ == '__main__':
    unittest.main()
