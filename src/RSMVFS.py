import numpy as np


class RSMVFConfig:
    def __init__(self, num_view, num_class, num_instances, num_total_features, lambda1, lambda2,
                 eps=10**-6, converge_condition=10**-3, lo=1, lo_max=10**6):
        """
        :param num_view: number of total view in the whole dataset
        :param num_class: number of target class
        :param num_instances: number of instances
        :param lambda1: regularization param 1
        :param lambda2: regularization param 2
        :param eps:
        :param converge_condition: convergence limit
        :param lo: lo parameter for scaled LM
        :param lo_max: maximum bound for lo

        """
        self.v = num_view
        self.c = num_class
        self.n = num_instances
        self.d = num_total_features

        self.lo = lo
        self.lo_max = lo_max
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eps = eps
        self.converge_condition = converge_condition


class RSMVFSGlobal:
    def __init__(self, X: list, y, Z_init=None, U_init=None, F_init=None, **config):
        """
        :param X: list of view matrix
        :param W_init: initial value of W
        :param Z_init: initial value of Z
        :param U_init: initial value of U
        :param F_init: initial value of F

        If initial value is None, initialize with given paper value

        """

        self.X = X # list of view matrix
        self.y = y
        self.config = RSMVFConfig(**config)

        # global variables

        # number of features of each view
        self.d_i_list = [d.shape[1] for d in self.X]

        # local models list
        self.local_models = [
            RSMVFSLocal(self.X[i], y, self.d_i_list[i],  **config)
            for i in range(self.config.v)
        ]

        # W_init=self._xavier_init(self.X[i].shape[1], y.shape[1]),

        # a_i, the importance scalar for each view
        self.a_i_list = [1/self.config.v for _ in range(self.config.v)]
        #
        self.prev_W = np.zeros((self.config.d, self.config.c))

        # initialize matrix
        self.Z = np.zeros((self.config.n, self.config.c)) if Z_init is None else Z_init # (n, c)
        self.U = np.zeros((self.config.n, self.config.c)) if U_init is None else U_init # (n, c)
        self.F = np.zeros((self.config.n, self.config.n)) if F_init is None else F_init # (n, n)
        self.U = np.zeros((self.config.n, self.config.c)) # (n, c)

    def _xavier_init(self, n, c):
        scale = 1 / max(1., (2 + 2) / 2.)
        limit = np.sqrt(3.0 * scale)
        weights = np.random.uniform(-limit, limit, size=(n, c))
        return weights

    def run(self):
        error = float('inf')
        iter = 0

        while error > self.config.converge_condition:
            iter += 1
            XW = np.mean([np.dot(m.X_i, m.W_i) for m in self.local_models], axis=0)

            for i, m in enumerate(self.local_models):
                m.update_W(self.a_i_list[i], self.Z, XW, self.U)

            self._update_a_i(self.local_models)
            F = self._update_F(self.local_models, self.y)
            Z = self._update_Z(F, self.y, XW, self.U)
            U = self._update_U(self.U, XW, Z)

            error, W_norm = self._update_error()
            Z_norm = np.linalg.norm(Z)
            U_norm = np.linalg.norm(U)

            self.config.lo = min(1.1*self.config.lo, self.config.lo_max)

            print(f"Iter {iter}: norm of W: {W_norm}, norm of Z: {Z_norm}, norm of U: {U_norm}, a: {self.a_i_list}")

        return [np.dot(m.X_i, m.W_i) for m in self.local_models]

    def _update_F(self, local_models, y):
        summation = np.sum([np.dot(self.X[i], m.W_i) for i, m in enumerate(local_models)], axis=0)
        norm = np.linalg.norm((summation - y), axis=1)
        threshold = 0.001
        norm = np.where(norm <= threshold, norm, self.config.eps)
        self.F = np.diag(1/2 * norm**-1)
        return self.F

    def _update_a_i(self, local_models):
        """ Must be called after updating W_i
            works as inplace method
         """
        W_norms = [np.sqrt(
            np.trace(np.linalg.multi_dot([m.W_i.T, m.G_i, m.W_i])) + self.config.eps)
            for m in local_models
        ] # Between eq. 19 and 20
        total_W_norm = np.sum(W_norms)

        self.a_i_list = W_norms / total_W_norm
        return self.a_i_list

    def _update_Z(self, F, y, XW, U):
        v = self.config.v
        lo = self.config.lo

        term1 = np.linalg.inv(2*v*F+lo*np.identity(F.shape[0]))
        term2 = 2*v*np.dot(F, y + lo*XW + lo*U)

        self.Z = np.dot(term1, term2)
        return self.Z

    def _update_U(self, U, XW, Z):
        self.U = U + XW - Z
        return self.U

    def _update_error(self):
        W = np.concatenate([m.W_i for m in self.local_models], axis=0) # equals to np.vstack, current W

        # TODO: Meaning of || W ||_F ?
        error_norm = np.linalg.norm(W - self.prev_W, ord='fro')**2
        self.prev_W = W
        return error_norm, np.linalg.norm(W)


class RSMVFSLocal:
    """
    Responsible for calculating primal W_i
    """
    def __init__(self, X_i, y, dim, W_init=None, **config):
        # fixed
        self.config = RSMVFConfig(**config)
        self.X_i = X_i # (n, di)
        self.y = y # (n, c)
        self.X_T_X = np.dot(X_i.T, X_i)

        self.yTy_inv = np.linalg.inv(np.dot(self.y.T, self.y))
        self.y_chunk = np.linalg.multi_dot([self.y, self.yTy_inv, self.y.T])

        self.S_b = np.linalg.multi_dot([self.X_i.T, self.y_chunk, self.X_i])
        self.I_y = np.identity(self.config.n) - self.y_chunk
        self.S_w = np.linalg.multi_dot([self.X_i.T, self.I_y, self.X_i])
        self.S_i = self.S_w - self.S_b

        # learnable
        self.W_i = (10 **-3) * np.eye(dim, self.config.c) if W_init is None else W_init # (n, c)
        self.G_i = None # (n, n)

    def compute_G(self, W_i, c, di):
        # TODO: G_i의 매트릭스 shape이 어떻게 되먹은건가. 논문에서는 c X c라 되어 있는데 그러면 계산이 안됨.
        G_i = np.zeros((di, di))
        diag_elem = 1 / (np.linalg.norm(W_i[:c], ord=2, axis=1) + self.config.eps)

        for i, v in enumerate(diag_elem):
            G_i[i, i] = v

        return G_i

    def update_W(self, a_i, Z, XW, U):
        # equation 23
        l1, l2, lo = self.config.lambda1, self.config.lambda2, self.config.lo
        # S_i = self._update_S_i(a_i)
        S_i = self.S_i
        X_i = self.X_i
        W_i = self.W_i

        c = self.config.c
        di = X_i.shape[1]

        G_i = self.compute_G(W_i, c, di) # first line in Algorithm 1.

        term_1 = (2 * l1 / a_i) * G_i + lo * self.X_T_X + l2 * S_i

        term_2 = (
                np.dot(X_i.T, Z)
                + np.linalg.multi_dot([X_i.T, X_i, W_i])
                - np.dot(X_i.T, XW)
                - np.dot(X_i.T, U)
        )

        try:
            W_i = lo * np.dot(np.linalg.inv(term_1), term_2)

        except np.linalg.LinAlgError:
            W_i = lo * np.dot(np.linalg.pinv(term_1), term_2)

        self.G_i = G_i
        self.W_i = W_i
        return W_i

    def _update_S_i(self, a_i):
        S_b = (a_i ** 2) * np.linalg.multi_dot([self.X_i.T, self.y_chunk, self.X_i])  # eq 6
        I_y = np.identity(self.config.n) - self.y_chunk

        S_w = (a_i ** 2) * np.linalg.multi_dot([self.X_i.T, I_y, self.X_i])  # eq 7
        S_i = (S_w - S_b) / (a_i ** 2)

        s_b = np.linalg.multi_dot([self.X_i.T, self.y_chunk, self.X_i])
        s_w = np.linalg.multi_dot([self.X_i.T, I_y, self.X_i])
        self.S_i = S_i # eq10
        return S_i