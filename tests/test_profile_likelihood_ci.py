import numpy as np

from scigraph.analyses.curvefit import CurveFit, Linear
from scigraph.datatables import XYTable


class NonFiniteProfileModel(CurveFit):
    """Test model whose hypothetical lower profile values are invalid."""

    @staticmethod
    def _f(x, scale):
        if scale < 0.5:
            return np.full_like(x, np.nan, dtype=float)
        return scale * x

    def _profile_initial_step(self, best, *_):
        return best


def test_profile_likelihood_matches_linear_model_ci():
    """For a linear model the profile and covariance intervals coincide."""
    rng = np.random.default_rng(4)
    x = np.linspace(0, 10, 30)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.5, x.size)
    table = XYTable(np.column_stack((x, y)), 1, 1, 1, ["data"])
    fit = Linear(table)
    fit.fit()

    profile = fit.profile_likelihood_CI()
    approximate = fit.approximate_CI()

    np.testing.assert_allclose(profile.to_numpy(), approximate.to_numpy(), rtol=1e-6)


def test_result_uses_profile_likelihood_intervals_by_default():
    rng = np.random.default_rng(12)
    x = np.linspace(0, 10, 24)
    y = 1.5 * x + 2.0 + rng.normal(0, 0.4, x.size)
    table = XYTable(np.column_stack((x, y)), 1, 1, 1, ["data"])
    fit = Linear(table)
    fit.fit()

    result = fit.result
    profile = fit.profile_likelihood_CI()

    np.testing.assert_allclose(
        result.loc[("Lower CI 95%", slice(None)), "data"].to_numpy(),
        profile.loc[("Lower CI 95%", slice(None)), "data"].to_numpy(),
    )
    np.testing.assert_allclose(
        result.loc[("Upper CI 95%", slice(None)), "data"].to_numpy(),
        profile.loc[("Upper CI 95%", slice(None)), "data"].to_numpy(),
    )


def test_result_can_use_approximate_or_no_confidence_intervals():
    x = np.linspace(0, 5, 12)
    y = 3.0 * x + 1.0
    table = XYTable(np.column_stack((x, y)), 1, 1, 1, ["data"])

    approximate_fit = Linear(table, confidence_interval_method="approximate")
    approximate_fit.fit()
    assert ("Lower CI 95%", "m") in approximate_fit.result.index

    no_ci_fit = Linear(table, confidence_interval_method="none")
    no_ci_fit.fit()
    assert "Lower CI 95%" not in no_ci_fit.result.index.get_level_values(0)


def test_profile_likelihood_handles_shared_global_parameter():
    rng = np.random.default_rng(8)
    x = np.linspace(0, 10, 20)
    y_a = 2.0 * x + 1.0 + rng.normal(0, 0.4, x.size)
    y_b = 2.0 * x - 3.0 + rng.normal(0, 0.4, x.size)
    table = XYTable(np.column_stack((x, y_a, y_b)), 1, 1, 2, ["A", "B"])
    fit = Linear(table)
    fit.add_constraint("m")
    fit.fit()

    profile = fit.profile_likelihood_CI()

    lower = profile.loc[("Lower CI 95%", "m")]
    upper = profile.loc[("Upper CI 95%", "m")]
    np.testing.assert_allclose(lower["A"], lower["B"])
    np.testing.assert_allclose(lower["A"], lower["Global (Shared)"])
    np.testing.assert_allclose(upper["A"], upper["B"])
    np.testing.assert_allclose(upper["A"], upper["Global (Shared)"])


def test_profile_likelihood_returns_nan_when_a_restricted_fit_is_nonfinite():
    x = np.linspace(1, 6, 8)
    y = x + np.array([0.2, -0.1, 0.1, -0.2, 0.2, -0.1, 0.1, -0.2])
    table = XYTable(np.column_stack((x, y)), 1, 1, 1, ["data"])
    fit = NonFiniteProfileModel(table)
    fit.fit()

    confidence_intervals = fit.profile_likelihood_CI()

    assert np.isnan(confidence_intervals.loc[("Lower CI 95%", "scale"), "data"])
