from RSMVFS import *
import pandas as pd

DATA_ROOT = "../data/mfeat/"
X_i = {"mfeat-" + x: None for x in ["fou", "fac", "kar", "mor", "pix", "zer"]}

def setup_data(data_path, data_dict):
    label = np.repeat(range(10), 200)

    for k, _ in data_dict.items():
        data_dict[k] = np.loadtxt(data_path + k)
        # print(f"Original -> {k}: {data_dict[k].shape}")
        #
        # data_dict[k] = np.concatenate([data_dict[k], np.copy(label).reshape(-1, 1)], axis=1)
        # print(f"After concat -> {k}: {data_dict[k].shape}")

    return list(data_dict.values()), label

if __name__ == '__main__':
    X, y = setup_data(DATA_ROOT, X_i)