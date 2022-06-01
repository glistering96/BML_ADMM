import numpy as np


def get_data(name):
    if name == "ad" or name == "ad-dataset":
        DATA_ROOT = "../data/ad-dataset/"
        X, Y = _load_ad(DATA_ROOT)

    elif name == "MF" or name=="mf":
        DATA_ROOT = "../data/mfeat/"
        X, Y = _load_MF(DATA_ROOT)

    elif name == "sample":
        DATA_ROOT = "../data/sample/"
        X, Y = _load_sample(DATA_ROOT)

    else:
        raise NotImplementedError()

    return X, Y


def _load_MF(DATA_ROOT):
    X_i = {"mfeat-" + x: None for x in ["fou", "fac", "kar", "mor", "pix", "zer"]}

    a = np.repeat(range(10), 200)
    label = np.zeros((a.shape[0], 10))
    label[np.arange(a.size), a] = 1

    i = 0
    for k, _ in X_i.items():
        data = np.loadtxt(DATA_ROOT + k)
        np.savetxt(DATA_ROOT + str(i) + ".csv", data, delimiter=",")
        X_i[k] = data
        i += 1

    np.savetxt(DATA_ROOT + "Y.csv", label, delimiter=",")

    return list(X_i.values()), label

def _load_ad(DATA_ROOT):
    X = []

    for i in range(5):
        X.append(np.loadtxt(DATA_ROOT + str(i) + ".csv", delimiter=","))

    Y = np.loadtxt(DATA_ROOT + "Y.csv", delimiter=",")

    return X, Y

def _load_sample(DATA_ROOT):
    X = []

    for i in range(5):
        X.append(np.loadtxt(DATA_ROOT + str(i) + ".csv", delimiter=","))

    Y = np.loadtxt(DATA_ROOT + "Y.csv", delimiter=",")

    return X, Y
def create_sample():
    DATA_ROOT = "../data/sample/"
    n = 200
    c = 4
    d = [60, 40, 40, 60, 50]

    a = np.repeat(range(4), 50)
    label = np.zeros((a.shape[0], c))
    label[np.arange(a.size), a] = 1

    X = [np.random.rand(n, di) for di in d]
    for i in range(len(d)):
        np.savetxt(DATA_ROOT + str(i) + ".csv", X[i], delimiter=",")

    np.savetxt(DATA_ROOT + "Y.csv", label, delimiter=",")

if __name__ == '__main__':
    # get_data("MF")
    create_sample()