from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    from sortedl1 import Slope


class Objective(BaseObjective):
    name = "SLOPE_Path"
    min_benchopt_version = "1.5"
    requirements = ["numpy", "pip:sortedl1"]
    parameters = {
        "path_length": [100],
        "q": [0.1],
        "fit_intercept": [True, False],
    }

    def __init__(self, path_length=100, q=0.1, fit_intercept=False):
        self.path_length = path_length
        self.q = q
        self.fit_intercept = fit_intercept

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.n_samples, self.n_features = self.X.shape

        model = Slope(fit_intercept=self.fit_intercept, q=self.q)
        _, _, self.alphas, self.lambdas = model.path(
            self.X, self.y, path_length=self.path_length
        )

        self.actual_path_length = len(self.alphas)

    def get_one_result(self):
        return dict(
            coefs=np.zeros([self.n_features, self.actual_path_length]),
            intercepts=np.zeros(self.actual_path_length),
        )

    def evaluate_result(self, coefs, intercepts):
        path_length = self.actual_path_length
        primals = np.empty(path_length, dtype=np.float64)
        duals = np.empty(path_length, dtype=np.float64)

        n = self.n_samples

        for i in range(path_length):
            coef = coefs[:, i]
            intercept = intercepts[i]
            lambdas = self.lambdas * self.alphas[i]

            residual = self.y - self.X @ coef - intercept

            primals[i] = 1.0 / (2 * n) * residual @ residual + np.sum(
                lambdas * np.sort(np.abs(coef))[::-1]
            )

            # feasible dual through dual scaling
            theta = residual
            theta /= max(1, self._dual_norm_slope(theta, lambdas))

            duals[i] = (norm(self.y) ** 2 - norm(self.y - theta * n) ** 2) / (2 * n)

        gaps = primals - duals

        max_rel_duality_gap = np.max(gaps / primals)
        max_abs_duality_gap = np.max(gaps)
        mean_rel_duality_gaps = np.mean(gaps / primals)
        mean_abs_duality_gaps = np.mean(gaps)

        return dict(
            value=np.mean(primals),
            max_rel_duality_gap=max_rel_duality_gap,
            max_abs_duality_gap=max_abs_duality_gap,
            mean_rel_duality_gaps=mean_rel_duality_gaps,
            mean_abs_duality_gaps=mean_abs_duality_gaps,
        )

    def get_objective(self):
        return dict(
            X=self.X,
            y=self.y,
            fit_intercept=self.fit_intercept,
            alphas=self.alphas,
            lambdas=self.lambdas,
        )

    def _dual_norm_slope(self, theta, lambdas):
        Xtheta = np.sort(np.abs(self.X.T @ theta))[::-1]
        taus = 1 / np.cumsum(lambdas)
        return np.max(np.cumsum(Xtheta) * taus)
