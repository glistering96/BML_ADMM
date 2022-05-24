import numpy as np

class Model:
    def __init__(self, X, Y, Z, U, F, W):
        self.X = X
        self.y = Y
        self.Z = Z
        self.U = U
        self.F = F
        self.W = W

    def calculate_a(self, W: list, G):
        a = [
            np.sqrt(
                np.trace(
                    np.linalg.multi_dot([W_i.T, G_i, W_i])
                    )
                ) for W_i, G_i in zip(W, G)
            ]

        total = np.sum(a)
        return np.array(a) / total

    def calculate_F(self, X, W):
        chi_iq_W_i = [np.dot(X_i, W_i) for X_i, W_i in zip(X, W)]

    def calculate_G_i(self, W_i, eps):
        diag = 1 / (np.linalg.norm(W_i, axis=0) + eps)
        return np.diag(diag)

    def update_W(self, W):
        pass

    def run(self, eps=10**-6):
        # calculate G_i for all W_i
        G = [self.calculate_G_i(W_i, eps) for W_i in self.W]

        # calculate a_i for all view
        a = self.calculate_a(self.W, G)

        # update W_i



