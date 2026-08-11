from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import os
    import urllib.request

    import appdirs
    import numpy as np
    from rpy2 import robjects
    from rpy2.robjects import default_converter, numpy2ri
    from rpy2.robjects.conversion import localconverter
    from scipy import sparse
    from scipy.sparse import csc_array
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MaxAbsScaler, StandardScaler


def fetch_breheny(dataset: str):
    base_dir = appdirs.user_cache_dir("benchmark_slope_path")

    path = os.path.join(base_dir, dataset + ".rds")

    # download raw data unless it is stored in data folder already
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = (
            f"https://github.com/IowaBiostat/data-sets/raw/main/{dataset}/{dataset}.rds"
        )
        urllib.request.urlretrieve(url, path)

    read_rds = robjects.r["readRDS"]

    # `numpy2ri.activate()` is deprecated (and raises in recent rpy2), so use a
    # local converter to turn the R arrays into numpy arrays instead.
    np_cv_rules = default_converter + numpy2ri.converter
    with localconverter(np_cv_rules):
        data = read_rds(path)
        X = np.asarray(data[0])
        y = np.asarray(data[1])

    density = np.sum(X != 0) / X.size

    if density <= 0.2:
        X = csc_array(X)

    return X, y


class Dataset(BaseDataset):
    name = "breheny"

    parameters = {
        "dataset": ["brca1", "Rhee2006", "Scheetz2006"],
        "standardize": [True, False],
    }

    install_cmd = "conda"
    requirements = [
        "conda-forge::r-base",
        "conda-forge::rpy2",
        "conda-forge::numpy",
        "conda-forge::scipy",
        "conda-forge::appdirs",
        "conda-forge::scikit-learn",
        # On Windows, rpy2 runs `R CMD config --ldflags` at import time to find
        # the directory holding R.dll. R implements `CMD config` by querying
        # etc/Makeconf with make, so without make on PATH it reports "R was not
        # built as a library" and rpy2 fails with a TypeError. Windows has no
        # system make, so pull it in from conda-forge. See solvers/rslope.py.
        "conda-forge::make",
    ]

    def __init__(self, dataset="brca1", standardize=True):
        super().__init__()
        self.dataset = dataset
        self.standardize = standardize

    def get_data(self):
        X, y = fetch_breheny(self.dataset)

        if self.standardize:
            X = VarianceThreshold().fit_transform(X)

            if sparse.issparse(X):
                X = MaxAbsScaler().fit_transform(X).tocsc()
            else:
                X = StandardScaler().fit_transform(X)

        return dict(X=X, y=y)
