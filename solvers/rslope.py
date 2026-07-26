from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import INFINITY

with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.helpers.r_lang import import_rpackages
    from rpy2 import robjects
    from rpy2.robjects import default_converter, numpy2ri, packages
    from rpy2.robjects.conversion import localconverter
    from scipy import sparse

    # Setup the system to allow rpy2 running. `numpy2ri.activate()` is
    # deprecated (and raises in recent rpy2), so use a local converter instead.
    np_cv_rules = default_converter + numpy2ri.converter
    import_rpackages("SLOPE")


class Solver(BaseSolver):
    name = "rSLOPE"

    install_cmd = "conda"
    requirements = ["r-base", "r:r-slope", "r-matrix", "rpy2", "scipy"]
    references = [
        "M. Bogdan, E. van den Berg, C. Sabatti, W. Su, and E. J. Candès, ",
        "“SLOPE – adaptive variable selection via convex optimization,” ",
        "Ann Appl Stat, vol. 9, no. 3, pp. 1103–1140, Sep. 2015, ",
        "doi: 10.1214/15-AOAS842.",
    ]
    support_sparse = True

    sampling_strategy = "tolerance"

    def set_objective(self, X, y, fit_intercept, alphas, lambdas):
        self.fit_intercept = fit_intercept
        self.n_alphas = len(alphas)

        # Convert the numpy inputs to R objects up front so that the fit call
        # itself does not need an active converter (which would also convert the
        # returned R object into a Python structure).
        with localconverter(np_cv_rules):
            conv = robjects.conversion.get_conversion()
            self.y = conv.py2rpy(y)
            self.alphas = conv.py2rpy(alphas)
            self.lambdas = conv.py2rpy(lambdas)

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
                self.X = conv.py2rpy(X)

        self.slope = robjects.r["SLOPE"]

    def run(self, tol):
        if tol == INFINITY:
            max_passes = 1
            tol = 1
        else:
            max_passes = 1_000_000

        fit_dict = {"lambda": self.lambdas, "alpha": self.alphas}

        # All arguments are already R objects, so no converter is active here and
        # the returned fit stays an R object for `get_result`.
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
        # Extract the coefficients in R to avoid relying on the Python wrapper
        # type of the fit object (which varies across rpy2 versions).
        get_coefs = robjects.r("function(fit) as.array(fit$coefficients)")
        with localconverter(np_cv_rules):
            coefs_array = np.array(get_coefs(self.fit))

        coefs = coefs_array[1:, 0, :] if self.fit_intercept else coefs_array[:, 0, :]
        intercepts = (
            coefs_array[0, 0, :] if self.fit_intercept else np.zeros(self.n_alphas)
        )

        return dict(coefs=coefs, intercepts=intercepts)
