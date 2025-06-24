from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import INFINITY

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    from scipy import sparse
    from sklearn.isotonic import isotonic_regression


class Solver(BaseSolver):
    name = "PGD"
    sampling_strategy = "tolerance"
    install_cmd = "conda"
    requirements = ["numpy", "scipy", "scikit-learn"]
    parameters = {
        "acceleration": ["none", "fista"],
    }
    references = [
        'A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding '
        'algorithm for linear inverse problems", SIAM J. Imaging Sci., '
        "vol. 2, no. 1, pp. 183-202 (2009)",
    ]

    def set_objective(self, X, y, fit_intercept, alphas, lambdas):
        self.X, self.y, self.fit_intercept, self.alphas, self.lambdas = (
            X,
            y,
            fit_intercept,
            alphas,
            lambdas,
        )

    def get_result(self):
        return dict(coefs=self.coefs, intercepts=self.intercepts)

    def run(self, tol):
        n, p = self.X.shape
        path_length = len(self.alphas)

        self.coefs = np.zeros((p, path_length))
        self.intercepts = np.zeros(len(self.alphas))

        if tol == INFINITY:
            return

        coef = np.zeros(p)
        intercept = 0.0

        if sparse.issparse(self.X):
            L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2 / n
        else:
            L = np.linalg.norm(self.X, ord=2) ** 2 / n

        for i, alpha in enumerate(self.alphas):
            lambdas = alpha * self.lambdas

            # FISTA variables
            t = 1.0
            z = coef.copy()

            while True:
                if self.acceleration == "fista":
                    # FISTA acceleration
                    coef_prev = coef.copy()

                    residual = self.X @ z + intercept - self.y
                    grad = self.X.T @ residual / n

                    coef[:] = self._prox(z - grad / L, lambdas / L)

                    t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
                    z[:] = coef + ((t - 1) / t_next) * (coef - coef_prev)
                    t = t_next

                    if self.fit_intercept:
                        intercept -= np.mean(residual)

                else:
                    residual = self.X @ coef + intercept - self.y
                    grad = self.X.T @ residual / n
                    coef[:] = self._prox(coef - grad / L, lambdas / L)

                    if self.fit_intercept:
                        intercept -= np.mean(residual)

                primal = 1.0 / (2 * n) * residual @ residual + np.sum(
                    lambdas * np.sort(np.abs(coef))[::-1]
                )

                theta = residual / max(1, self._dual_norm_slope(grad * n, lambdas))
                dual = (norm(self.y) ** 2 - norm(self.y + theta * n) ** 2) / (2 * n)

                gap = primal - dual

                if gap <= tol:
                    self.coefs[:, i] = coef
                    self.intercepts[i] = intercept
                    break

    def _prox(self, beta, lambdas):
        """Proximal operator of the OWL norm
        dot(lambdas, reversed(sort(abs(beta))))
        Follows description and notation from:
        X. Zeng, M. Figueiredo,
        The ordered weighted L1 norm: Atomic formulation, dual norm,
        and projections.
        eprint http://arxiv.org/abs/1409.4271
        (From pyowl)
        XXX

        Parameters
        ----------
        beta: array
            vector of coefficients
        lambdas: array
            vector of regularization weights

        Returns
        -------
        array
            the result of the proximal operator
        """
        # from https://github.com/svaiter/gslope_oracle_inequality/
        # blob/master/graphslope/core.py
        beta_abs = np.abs(beta)
        ix = np.argsort(beta_abs)[::-1]
        beta_abs = beta_abs[ix]
        # project to K+ (monotone non-negative decreasing cone)
        beta_abs = isotonic_regression(beta_abs - lambdas, y_min=0, increasing=False)

        # undo the sorting
        inv_ix = np.zeros_like(ix)
        inv_ix[ix] = np.arange(len(beta))
        beta_abs = beta_abs[inv_ix]

        return np.sign(beta) * beta_abs

    def _dual_norm_slope(self, grad, lambdas):
        grad_abs_sorted = np.sort(np.abs(grad))[::-1]
        return np.max(np.cumsum(grad_abs_sorted) / np.cumsum(lambdas))
