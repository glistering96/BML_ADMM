from RSMVFS import *
import pandas as pd

DATA_ROOT = "../data/mfeat/"
X_i = {"mfeat-" + x: None for x in ["fou", "fac", "kar", "mor", "pix", "zer"]}


def setup_data(data_path, data_dict):
    a = np.repeat(range(10), 200)
    label = np.zeros((a.shape[0], 10))
    label[np.arange(a.size), a] = 1
    index = np.array([x for x in range(0, 2000, 8)], dtype=np.int32)

    for k, _ in data_dict.items():
        if debugger_is_active():
            data_dict[k] = np.loadtxt(data_path + k)[index, :]
            label = label[index]
        else:
            data_dict[k] = np.loadtxt(data_path + k)
        # print(f"Original -> {k}: {data_dict[k].shape}")
        #
        # data_dict[k] = np.concatenate([data_dict[k], np.copy(label).reshape(-1, 1)], axis=1)
        # print(f"After concat -> {k}: {data_dict[k].shape}")

    return list(data_dict.values()), label


def auto_calculate_configs(X, y):
    config = {"num_view": len(X),
    "num_instances" : y.shape[0],
    "num_class" : y.shape[1],
    "num_total_features" : np.sum([x.shape[1] for x in X]),
    "lambda1" : 10**-2,
    "lambda2" : 10**-1
     }

    return config

import sys

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    gettrace = getattr(sys, 'gettrace', lambda : None)
    return gettrace() is not None

if __name__ == '__main__':
    X, y = setup_data(DATA_ROOT, X_i)
    config = auto_calculate_configs(X, y)
    model = RSMVFSGlobal(X, y, **config)

    model.run()
