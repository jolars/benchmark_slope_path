from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import INFINITY

with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.helpers.r_lang import import_rpackages
    from rpy2 import robjects
    from rpy2.robjects import numpy2ri, packages
    from scipy import sparse

    # Setup the system to allow rpy2 running
    numpy2ri.activate()
    import_rpackages("SLOPE")


class Solver(BaseSolver):
    name = "rSLOPE"

    install_cmd = "conda"
    requirements = ["r-base", "rpy2", "r:r-slope", "r-matrix", "scipy"]
    references = [
        "M. Bogdan, E. van den Berg, C. Sabatti, W. Su, and E. J. Candès, ",
        "“SLOPE – adaptive variable selection via convex optimization,” ",
        "Ann Appl Stat, vol. 9, no. 3, pp. 1103–1140, Sep. 2015, ",
        "doi: 10.1214/15-AOAS842.",
    ]
    support_sparse = True

    sampling_strategy = "tolerance"
    # stopping_criterion = SufficientProgressCriterion(
    #     patience=5, eps=1e-18, strategy="tolerance"
    # )

    def set_objective(self, X, y, fit_intercept, alphas, lambdas):
        self.y = y
        self.fit_intercept = fit_intercept
        self.alphas = alphas
        self.lambdas = lambdas

        if sparse.issparse(X):
            r_Matrix = packages.importr("Matrix")
            X = X.tocoo()
            self.X = r_Matrix.sparseMatrix(
                i=robjects.IntVector(X.row + 1),
                j=robjects.IntVector(X.col + 1),
                x=robjects.FloatVector(X.data),
                dims=robjects.IntVector(X.shape),
            )
        else:
            self.X = X

        self.slope = robjects.r["SLOPE"]

    def run(self, tol):
        if tol == INFINITY:
            max_passes = 1
            tol = 1
        else:
            max_passes = 1_000_000

        fit_dict = {"lambda": self.lambdas, "alpha": self.alphas}

        self.fit = self.slope(
            self.X,
            self.y,
            intercept=self.fit_intercept,
            scale="none",
            center=False,
            max_passes=max_passes,
            tol_rel_gap=tol * 0.1,
            tol_infeas=tol,
            tol_rel_coef_change=tol,
            **fit_dict,
        )

    def get_result(self):
        results = dict(zip(self.fit.names, list(self.fit)))
        r_as = robjects.r["as"]
        coefs_array = np.array(r_as(results["coefficients"], "array"))

        coefs = coefs_array[1:, 0, :] if self.fit_intercept else coefs_array[:, 0, :]
        intercepts = (
            coefs_array[0, 0, :] if self.fit_intercept else np.zeros(len(self.alphas))
        )

        return dict(coefs=coefs, intercepts=intercepts)
