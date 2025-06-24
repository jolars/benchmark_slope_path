from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import INFINITY

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    from scipy import sparse
    from scipy.linalg import cholesky, solve_triangular
    from scipy.sparse.linalg import lsqr
    from sklearn.isotonic import isotonic_regression


class Solver(BaseSolver):
    name = "ADMM"
    sampling_strategy = "tolerance"
    parameters = {"rho": [10, 100, 1000]}
    install_cmd = "conda"
    requirements = ["numpy", "scipy", "scikit-learn"]
    references = [
        "Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2010). "
        "Distributed optimization and statistical learning via the "
        "alternating direction method of multipliers. Foundations and "
        "Trends in Machine Learning, 3(1), 1-122. "
        "https://doi.org/10.1561/2200000016"
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
        # Implementation based on
        # https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html

        n, p = self.X.shape
        path_length = len(self.alphas)

        self.coefs = np.zeros((p, path_length))
        self.intercepts = np.zeros(path_length)

        if tol == INFINITY:
            return

        gamma = 1.0
        rho = self.rho
        lsqr_atol = 1e-6
        lsqr_btol = 1e-6

        X = self.X

        if self.fit_intercept:
            X = self._add_intercept_column(X)
            p += 1

        w = np.zeros(p)
        z = np.zeros(p)
        u = np.zeros(p)

        z_old = z.copy()

        do_lsqr = sparse.issparse(X) and min(n, p) > 1000

        # cache factorizations if dense
        if not do_lsqr:
            if n >= p:
                XtX = X.T @ X
                if sparse.issparse(X):
                    XtX = XtX.toarray()
                np.fill_diagonal(XtX, XtX.diagonal() + rho)
                L = cholesky(XtX, lower=True)
            else:
                XXt = X @ X.T
                if sparse.issparse(X):
                    XXt = XXt.toarray()
                XXt *= 1 / rho
                np.fill_diagonal(XXt, XXt.diagonal() + 1)
                L = cholesky(XXt, lower=True)

            U = L.T

        Xty = X.T @ self.y

        for i, alpha in enumerate(self.alphas):
            lambdas = alpha * self.lambdas

            while True:
                if do_lsqr:
                    res = lsqr(
                        sparse.vstack((X, np.sqrt(rho) * sparse.eye(p))),
                        np.hstack((self.y, np.sqrt(rho) * (z - u))),
                        x0=w,
                        atol=lsqr_atol,
                        btol=lsqr_btol,
                    )
                    w[:] = res[0]
                else:
                    q = Xty + rho * (z - u)

                    U = L.T

                    if n >= p:
                        w[:] = solve_triangular(U, solve_triangular(L, q, lower=True))
                    else:
                        tmp = solve_triangular(
                            U, solve_triangular(L, X @ q, lower=True)
                        )
                        w[:] = q / rho - (X.T @ tmp) / (rho**2)

                self.coefs[:, i] = w[self.fit_intercept :]
                if self.fit_intercept:
                    self.intercepts[i] = w[0]

                z_old = z.copy()
                w_hat = gamma * w + (1 - gamma) * z_old

                z[:] = w_hat + u
                z[self.fit_intercept :] = self._prox(
                    z[self.fit_intercept :], lambdas * (n / rho)
                )

                u += w_hat - z

                r_norm = norm(w - z)
                s_norm = norm(-rho * (z - z_old))

                eps_pri = np.sqrt(p) * tol * 0.1 + tol * max(norm(w), norm(z))
                eps_dual = np.sqrt(p) * tol * 0.1 + tol * norm(rho * u)

                if r_norm < eps_pri and s_norm < eps_dual:
                    break

    def _prox(self, beta, lambdas):
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

    def _add_intercept_column(self, X):
        n = X.shape[0]

        if sparse.issparse(X):
            return sparse.hstack((sparse.csc_array(np.ones((n, 1))), X), format="csc")
        else:
            return np.hstack((np.ones((n, 1)), X))
