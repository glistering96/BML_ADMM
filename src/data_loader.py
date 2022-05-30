import numpy as np


def get_data(name):
    if name == "ad" or name == "ad-dataset":
        DATA_ROOT = "../data/ad-dataset/"
        X, Y = _load_ad(DATA_ROOT)

    elif name == "MF" or name=="mf":
        DATA_ROOT = "../data/mfeat/"
        X, Y = _load_MF(DATA_ROOT)

    else:
        raise NotImplementedError()

    return X, Y


def _load_MF(DATA_ROOT):
    X_i = {"mfeat-" + x: None for x in ["fou", "fac", "kar", "mor", "pix", "zer"]}

    a = np.repeat(range(10), 200)
    label = np.zeros((a.shape[0], 10))
    label[np.arange(a.size), a] = 1

    for k, _ in X_i.items():
        X_i[k] = np.loadtxt(DATA_ROOT + k)

    return list(X_i.values()), label

def _load_ad(DATA_ROOT):
    X = []

    for i in range(5):
        X.append(np.loadtxt(DATA_ROOT + str(i) + ".csv", delimiter=","))

    Y = np.loadtxt(DATA_ROOT + "Y.csv", delimiter=",")

    return X, Y