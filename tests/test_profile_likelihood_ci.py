import numpy as np

from scigraph.analyses.curvefit import Linear
from scigraph.datatables import XYTable


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
