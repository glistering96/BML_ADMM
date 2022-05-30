import copy
import multiprocessing

import numpy as np

INV = np.linalg.inv

"""

Considering the computational complexity of each method, 

parallelize update W first

then consider a

"""

class RSMVFS_multiprocess:
    def __init__(self, X, Y, Z, U, F, W,
                 lo=1, l1=10**-3, l2=10**-3, eps=10**-6, lo_max=10**6, eps_0=10**-3,
                 num_process=1, verbose=True):
        self.X = X  # list of view matrix, [X1, X2, ..., Xv], Xv: [n, di]
        self.Y = Y  # [n, c]
        self.Z = Z  # [n, c]
        self.U = U  # [n, c]
        self.F = F  # [n, n]
        self.W = W  # [W1, W2, ..., Wv], Wv: [di, c]
        self.lo = lo # lo value, scalar
        self.l1 = l1    # lambda1
        self.l2 = l2    # lambda2
        self.eps = eps  # eps
        self.eps_0 = eps_0  # convergence threshold
        self.lo_max = lo_max    # maximum lo
        self.verbose = verbose

        # fixed values
        self.yTy = np.dot(Y.T, Y)
        self.y_chunk = np.linalg.multi_dot([Y, INV(self.yTy), Y.T])
        self.S = [self.calculate_S_i(X_i) for X_i in X]

        # other variables
        self.v = len(X)     # number of views
        self.n = Y.shape[0] # number of instances
        self.c = Y.shape[1] # number of class
        self.d = [x.shape[1] for x in X]    # [d1, d2, ..., dv]

        if num_process > 1:
            print(f"Distributed learning with {num_process} processes")
            self.num_process = num_process

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

    def calculate_F(self, X, W, Y):
        chi_iq_W_i = [np.dot(X_i, W_i) for X_i, W_i in zip(X, W)]
        summation = np.sum(chi_iq_W_i)
        term = summation - Y
        norm = np.linalg.norm(term, axis=1) # calculate norm of each row

        off_indicies = np.where(norm > self.eps)
        F = np.diag(0.5 * (1 / norm))
        F[off_indicies, off_indicies] = 0
        return F

    def calculate_G_i(self, W_i, eps):
        diag = 1 / (np.linalg.norm(W_i, axis=1) + eps)
        return np.diag(diag)

    def calculate_S_i(self, X_i):
        # with eq 6, 7, and 10, S_i = np.dot(X.T, X) - 2*S_b
        S_b = np.linalg.multi_dot([X_i.T, self.y_chunk, X_i])
        S_i = np.dot(X_i.T, X_i) - 2 * S_b
        return S_i

    def calculate_XW(self, X, W):
        return np.mean([np.dot(X_i, W_i) for X_i, W_i in zip(X, W)], axis=0)

    def calculate_W_i(self, X_i, W_i, S_i, G_i, a_i, Z, XW, U):
        XTX = np.dot(X_i.T, X_i)
        term1 = (2*self.l1/a_i) * G_i + self.lo*XTX + self.l2*S_i
        term2 = np.dot(X_i.T, Z) + np.dot(XTX, W_i) - np.dot(X_i.T, XW) - np.dot(X_i.T, U)\

        W_next = self.lo * np.dot(INV(term1), term2)
        return W_next

    def calculate_Z(self, F, Y, XW, U):
        term1 = 2*self.v*F + self.lo*np.identity(self.n)
        term2 = 2*self.v*np.dot(F, Y) + self.lo*XW + self.lo*U
        Z_next = np.dot(INV(term1), term2)
        return Z_next

    def update_U(self, U, XW, Z):
        U_next = U + XW - Z
        return U_next

    def calculate_error(self, prev_W, W):
        all_W_prev = np.concatenate(prev_W)
        all_W = np.concatenate(W)
        term = all_W - all_W_prev
        error = np.linalg.norm(term, ord='fro')
        return error, W

    def run(self, eps=10**-6):
        # initial set up
        W = self.W
        Z = self.Z
        U = self.U
        XW = self.calculate_XW(self.X, W) # XW_k
        prev_W = copy.deepcopy(W)

        error = float('inf')
        i = 0

        MP_POOL = multiprocessing.Pool(self.num_process)

        while error > self.eps_0:
            # calculate G_i for all W_i
            G = [self.calculate_G_i(W_i, eps) for W_i in W]     # can be parallelized

            # calculate a_i for all view
            a = self.calculate_a(W, G)  # can be parallelized

            # update W_i
            # W = Parallel(n_jobs=self.num_process)(delayed(self.calculate_W_i)(X_i, W_i, S_i, G_i, a_i, Z, XW, U)
            #      for X_i, W_i, S_i, G_i, a_i in zip(self.X, W, self.S, G, a))   # can be parallelized

            arguments = [(X_i, W_i, S_i, G_i, a_i, Z, XW, U)
                         for X_i, W_i, S_i, G_i, a_i in zip(self.X, W, self.S, G, a)]
            W = MP_POOL.starmap(self.calculate_W_i, arguments)

            # update XW
            XW = self.calculate_XW(self.X, W) # XW_(k+1)

            # calculate F
            F = self.calculate_F(self.X, W, self.Y)

            # calculate Z
            Z = self.calculate_Z(F, self.Y, XW, U)

            # calculate U
            U = self.update_U(U, XW, Z)

            # calculate error
            error, prev_W = self.calculate_error(prev_W, W)

            self.lo = min(self.lo*1.1, self.lo_max)

            i += 1

            if self.verbose:
                print(f"[Iter {i:>3}] Error: {error: .4}")

        return W, a

class RSMVFS_local:
    def __init__(self, ):
        pass