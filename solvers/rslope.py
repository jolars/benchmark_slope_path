from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import INFINITY

with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.helpers.r_lang import import_rpackages
    from rpy2 import robjects
    from rpy2.robjects import numpy2ri, packages
    from scipy import sparse

    import_rpackages("SLOPE")


class Solver(BaseSolver):
    name = "rSLOPE"

    install_cmd = "conda"
    install_cmd = "conda"
    requirements = [
        "conda-forge::r-slope",
        "conda-forge::r-matrix",
        "conda-forge::rpy2",
        "conda-forge::scipy",
        # On Windows, rpy2 runs `R CMD config --ldflags` at import time to find
        # the directory holding R.dll. R implements `CMD config` by querying
        # etc/Makeconf with make, so without make on PATH it reports "R was not
        # built as a library" and rpy2 fails with a TypeError. Windows has no
        # system make, so pull it in from conda-forge.
        "conda-forge::make",
    ]
    references = [
        "M. Bogdan, E. van den Berg, C. Sabatti, W. Su, and E. J. Candes, ",
        "'SLOPE - adaptive variable selection via convex optimization,' ",
        "Ann Appl Stat, vol. 9, no. 3, pp. 1103-1140, Sep. 2015, ",
        "doi: 10.1214/15-AOAS842.",
    ]
    support_sparse = True

    sampling_strategy = "tolerance"

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

        # Convert numpy inputs to R explicitly. rpy2 removed the global
        # numpy2ri.activate(), and a conversion context would also turn the
        # returned SLOPE object into a plain dict, so we convert at the boundary
        # and keep self.fit as an R object.
        X = numpy2ri.numpy2rpy(self.X) if isinstance(self.X, np.ndarray) else self.X

        fit_dict = {
            "lambda": numpy2ri.numpy2rpy(self.lambdas),
            "alpha": numpy2ri.numpy2rpy(self.alphas),
        }

        self.fit = self.slope(
            X,
            numpy2ri.numpy2rpy(self.y),
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
        # SLOPE returns `coefficients` as a list with one sparse dgCMatrix per
        # alpha in the path. Build each dense coefficient vector from the matrix
        # slots directly; R-side S4 coercion is not dispatched reliably by rpy2.
        coefs_obj = self.fit.rx2("coefficients")
        if tuple(coefs_obj.rclass) == ("list",):
            mats = [coefs_obj[i] for i in range(len(coefs_obj))]
        else:
            mats = [coefs_obj]

        betas = np.column_stack([self._to_dense(m) for m in mats])

        if self.fit_intercept:
            coefs = betas[1:, :]
            intercepts = betas[0, :]
        else:
            coefs = betas
            intercepts = np.zeros(betas.shape[1])

        return dict(coefs=coefs, intercepts=intercepts)

    @staticmethod
    def _to_dense(m):
        data = np.asarray(m.do_slot("x"))
        indices = np.asarray(m.do_slot("i"))
        indptr = np.asarray(m.do_slot("p"))
        dim = tuple(int(d) for d in m.do_slot("Dim"))
        dense = sparse.csc_matrix((data, indices, indptr), shape=dim).toarray()
        return dense.ravel()
