from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from modules import full_path
    from scipy import sparse


class Solver(BaseSolver):
    name = "SolutionPath"
    sampling_strategy = "iteration"
    install_cmd = "conda"
    requirements = ["pip:git+https://github.com/jolars/slope-path"]
    # TODO: when benchopt 1.7 is released, update to
    # "pip::git..."

    references = [
        "Dupuis, X., & Tardivel, P. (2024). The solution path of SLOPE. "
        "Proceedings of The 27th International Conference on Artificial "
        "Intelligence and Statistics, 238, 775–783. "
        "https://proceedings.mlr.press/v238/dupuis24a.html"
    ]

    def set_objective(self, X, y, fit_intercept, alphas, lambdas):
        self.n_samples, self.n_features = X.shape

        self.X, self.y, self.fit_intercept, self.lambdas, self.alphas = (
            X,
            y,
            fit_intercept,
            lambdas * self.n_samples,
            alphas,
        )

    def warm_up(self):
        # Needs numba JIT compilation
        self.run_once()

    def skip(self, X, y, fit_intercept, alphas, lambdas):
        if fit_intercept:
            return True, f"{self.name} does not handle intercept fitting"

        if sparse.issparse(X):
            return True, f"{self.name} does not handle sparse design matrices"

        return False, None

    def run(self, it):
        alphas = self.alphas

        alpha_max = np.max(alphas)
        alpha_min = np.min(alphas)
        ratio = 1 - alpha_min / alpha_max

        # NOTE: This is the tolerance values used in the experiments in the
        # package. I've tried using tolerance as stopping criterion as well,
        # but the function throws for large tolerances.
        tol = 1e-10

        self.alpha_exact, self.coefs_exact, _, _, _, _, _ = full_path(
            self.X,
            self.y,
            self.lambdas,
            ratio=ratio,
            k_max=it,
            rtol_pattern=tol,
            atol_pattern=tol,
            rtol_gamma=tol,
            split_max=10,
            log=False,
        )

    def get_result(self):
        coefs = self._interpolate_coefs(self.alpha_exact, self.coefs_exact)
        return dict(coefs=coefs, intercepts=np.zeros(len(self.alphas)))

    def _interpolate_coefs(self, alpha_exact, coefs_exact):
        coefs = np.zeros((self.n_features, len(self.alphas)))
        for i, alpha in enumerate(self.alphas):
            if alpha >= alpha_exact[0]:  # gamma is >= the largest gamma in the path
                coefs[:, i] = coefs_exact[0].copy()
            elif alpha <= alpha_exact[-1]:  # gamma is <= the smallest gamma in the path
                coefs[:, i] = coefs_exact[-1].copy()
            else:
                # Find the two consecutive gammas that gamma falls between
                # We need to search in reverse since Gamma is in descending order
                idx = np.searchsorted(-np.array(alpha_exact), -alpha)
                if idx == 0:
                    coefs[:, i] = coefs_exact[0].copy()
                elif idx == len(alpha_exact):
                    coefs[:, i] = coefs_exact[-1].copy()
                else:
                    # Linear interpolation between solutions
                    t = (alpha - alpha_exact[idx]) / (
                        alpha_exact[idx - 1] - alpha_exact[idx]
                    )
                    sol = (1 - t) * coefs_exact[idx] + t * coefs_exact[idx - 1]
                    coefs[:, i] = sol

        return coefs
