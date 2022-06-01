from RSMVFS_v2 import RSMVFS
import numpy as np
from data_loader import get_data
import sys
import time
import seaborn as sns


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    gettrace = getattr(sys, 'gettrace', lambda : None)
    return gettrace() is not None


def auto_calculate_configs(X, y):
    config = {"num_view": len(X),
    "num_instances" : y.shape[0],
    "num_class" : y.shape[1],
    "num_total_features" : np.sum([x.shape[1] for x in X]),
    "lambda1" : 10**-2,
    "lambda2" : 10**-1
     }

    return config


def one_run(DATA):
    X, y = get_data(DATA)
    reg_value = {"ad": (10 ** -2, 10 ** -3, 10**-50), "MF": (10 ** -2, 10 ** -2, 10**-3),
                 "sample": (10 ** -2, 10 ** -2, 10**-50)}

    n = y.shape[0]
    c = y.shape[1]
    v = len(X)

    l1, l2, eps_0 = reg_value[DATA]
    W = [(10 ** -3) * np.eye(x.shape[1], y.shape[1]) for x in X]
    Z = np.zeros((n, c))
    U = np.zeros((n, c))
    F = np.zeros((n, n))

    model = RSMVFS(X, y, Z, U, F, W, l1=l1, l2=l2, verbose=True, eps_0=eps_0)
    start = time.time()
    W, a = model.run()
    print(time.time() - start)
    print(a)

    import matplotlib.pyplot as plt
    import os

    path = f"../result_fig/{DATA}/non_parallel/"

    if not os.path.isdir(path):
        os.makedirs(path)

    for i, w in enumerate(W):
        fig = plt.figure()
        ax = sns.heatmap(w)
        plt.figure(tight_layout=True)

        fig.savefig(path + f"W_{i}.jpg")

if __name__ == '__main__':

    DATA = ["MF", "ad"]
    # DATA = ["sample"]
    for d in DATA:
        one_run(d)


