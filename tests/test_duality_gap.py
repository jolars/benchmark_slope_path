import numpy as np

from objective import Objective
from solvers.pgd import Solver


def make_intercept_objective():
    objective = Objective(path_length=1, fit_intercept=True)
    objective.X = np.array([[-1.0], [1.0]])
    objective.y = np.ones(2)
    objective.n_samples = 2
    objective.actual_path_length = 1
    objective.alphas = np.ones(1)
    objective.lambdas = np.ones(1)
    return objective


def test_intercept_dual_point_is_centered():
    objective = make_intercept_objective()

    result = objective.evaluate_result(
        coefs=np.zeros((1, 1)), intercepts=np.array([0.75])
    )

    np.testing.assert_allclose(result["max_abs_duality_gap"], 0.03125)
    assert result["max_abs_duality_gap"] >= 0.0


def test_pgd_gap_uses_the_current_primal_point():
    solver = Solver()
    solver.X = np.array([[-1.0], [1.0]])
    solver.y = np.ones(2)
    solver.fit_intercept = True

    gap = solver._primal_dual_gap(
        coef=np.zeros(1), intercept=0.75, lambdas=np.ones(1)
    )

    np.testing.assert_allclose(gap, 0.03125)
