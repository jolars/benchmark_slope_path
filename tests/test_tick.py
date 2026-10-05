import numpy as np
import pytest
from scipy import sparse
from scipy.stats import norm

from objective import Objective
from solvers.tick import Solver, _slope_parameters

pytest.importorskip("tick")


@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("sparse_input", [False, True])
def test_tick_solves_bh_path(fit_intercept, sparse_input):
    X = np.array([[-2.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    if sparse_input:
        X = sparse.csc_array(X)
    y = np.arange(5.0)
    q = 0.2
    lambdas = norm.ppf(1 - q * np.arange(1, 3) / 4)
    alphas = np.array([0.5, 0.1, 0.02])

    solver = Solver()
    assert solver.skip(X, y, fit_intercept, alphas, lambdas) == (False, None)
    solver.set_objective(X, y, fit_intercept, alphas, lambdas)
    solver.run(0)
    baseline = solver.get_result()
    np.testing.assert_array_equal(baseline["coefs"], np.zeros((2, 3)))
    np.testing.assert_array_equal(baseline["intercepts"], np.zeros(3))

    solver.run(300)
    result = solver.get_result()
    objective = Objective(path_length=3, q=q, fit_intercept=fit_intercept)
    objective.X, objective.y, objective.n_samples = X, y, len(y)
    objective.lambdas, objective.alphas = lambdas, alphas
    objective.actual_path_length = len(alphas)
    assert objective.evaluate_result(**result)["max_rel_duality_gap"] < 1e-6
    if fit_intercept:
        np.testing.assert_allclose(result["intercepts"], 2.0, atol=1e-6)
        np.testing.assert_allclose(
            solver.prox.value(np.array([1.0, 0.5, 7.0])),
            alphas[-1] * (lambdas @ np.array([1.0, 0.5])),
        )
    else:
        np.testing.assert_array_equal(result["intercepts"], np.zeros(3))


def test_tick_skips_non_bh_weights():
    solver = Solver()
    skip, reason = solver.skip(None, None, False, np.ones(2), np.ones(2))
    assert skip
    assert "Benjamini-Hochberg" in reason


def test_tick_accepts_generated_objective():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 5))
    y = 1.5 + X[:, 0] - X[:, 1]
    objective = Objective(q=0.2, path_length=4, fit_intercept=True)
    objective.set_data(X, y)
    assert set(objective.get_objective()) == {
        "X",
        "y",
        "fit_intercept",
        "alphas",
        "lambdas",
    }

    solver = Solver()
    assert solver._set_objective(objective) == (False, None)
    solver.run(1)
    result = solver.get_result()
    assert result["coefs"].shape == (5, objective.actual_path_length)
    assert result["intercepts"].shape == (objective.actual_path_length,)


@pytest.mark.parametrize("q", [0.05, 0.2])
@pytest.mark.parametrize("n_features", [1, 5, 200])
def test_tick_recovers_penalty_from_weights(q, n_features):
    lambdas = norm.ppf(1 - q * np.arange(1, n_features + 1) / (2 * n_features))
    recovered_q, strength = _slope_parameters(lambdas)
    reconstructed = strength * norm.ppf(
        1 - recovered_q * np.arange(1, n_features + 1) / (2 * n_features)
    )
    np.testing.assert_allclose(reconstructed, lambdas, rtol=1e-7)
    if n_features > 1:
        np.testing.assert_allclose(recovered_q, q, rtol=1e-7)
        np.testing.assert_allclose(strength, 1.0, rtol=1e-7)
