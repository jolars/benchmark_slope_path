from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import INFINITY, SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from sortedl1 import Slope


class Solver(BaseSolver):
    name = "sortedl1"
    stopping_criterion = SufficientProgressCriterion(
        patience=10, eps=1e-10, strategy="tolerance"
    )
    install_cmd = "conda"
    requirements = ["pip:sortedl1"]
    references = [
        "J. Larsson, Q. Klopfenstein, M. Massias, and J. Wallin, "
        "“Coordinate descent for SLOPE,” in Proceedings of the 26th "
        "international conference on artificial intelligence and statistics, "
        "F. Ruiz, J. Dy, and J.-W. van de Meent, Eds., in Proceedings of "
        "machine learning research, vol. 206. Valencia, Spain: PMLR, Apr. 2023, "
        "pp. 4802–4821. [Online]. Available: "
        "https://proceedings.mlr.press/v206/larsson23a.html"
    ]

    def set_objective(self, X, y, fit_intercept, alphas, lambdas):
        self.n_samples, self.n_features = X.shape
        self.X, self.y, self.fit_intercept, self.alphas, self.lambdas = (
            X,
            y,
            fit_intercept,
            alphas,
            lambdas,
        )

        self.model = Slope(
            lam=self.lambdas,
            fit_intercept=self.fit_intercept,
            max_iter=1_000_000,
            solver="hybrid",
        )

    def run(self, tol):
        if tol == INFINITY:
            self.coefs = np.zeros((self.n_features, len(self.alphas)))
            self.intercepts = np.zeros(len(self.alphas))
        else:
            self.model.tol = tol
            coefs, intercepts, _, _ = self.model.path(
                self.X, self.y, alphas=self.alphas
            )

            self.coefs = coefs[:, 0, :]
            self.intercepts = intercepts[0, :]

    def get_result(self):
        return dict(coefs=self.coefs, intercepts=self.intercepts)
