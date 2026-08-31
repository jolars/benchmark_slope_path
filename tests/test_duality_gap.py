import numpy as np
import pytest

from objective import Objective, TargetObjectiveCriterion
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
    assert result["target_rel_duality_gap"] == 1e-7


def test_pgd_gap_uses_the_current_primal_point():
    solver = Solver()
    solver.X = np.array([[-1.0], [1.0]])
    solver.y = np.ones(2)
    solver.fit_intercept = True

    gap = solver._primal_dual_gap(
        coef=np.zeros(1), intercept=0.75, lambdas=np.ones(1)
    )

    np.testing.assert_allclose(gap, 0.03125)


@pytest.mark.parametrize(
    ("gap", "should_stop"),
    [(2e-7, False), (1e-7, True), (5e-8, True)],
)
def test_stops_at_target_max_relative_duality_gap(gap, should_stop):
    criterion = TargetObjectiveCriterion(key_to_monitor="max_rel_duality_gap")
    criterion.terminal = None
    objective_list = [
        {
            "objective_max_rel_duality_gap": gap,
            "objective_target_rel_duality_gap": 1e-7,
        }
    ]

    stop, _ = criterion.check_convergence(objective_list)

    assert stop is should_stop


def test_target_relative_duality_gap_must_be_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        Objective(target_rel_duality_gap=0.0)
