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
        # v개 만큼의 local model hold
        self.local_models = [
            RSMVFSLocal(self.X[i], y, self.d_i_list[i],  **config)
            for i in range(self.config.v)
        ]

        # W_init=self._xavier_init(self.X[i].shape[1], y.shape[1]),

        # a_i, the importance scalar for each view
        self.a_i_list = [1/self.config.v for _ in range(self.config.v)]

        # initialize matrix
        self.Z = np.zeros((self.config.n, self.config.c)) if Z_init is None else Z_init # (n, c)
        self.U = np.zeros((self.config.n, self.config.c)) if U_init is None else U_init # (n, c)
        self.F = np.zeros((self.config.n, self.config.n)) if F_init is None else F_init # (n, n)
        self.U = np.zeros((self.config.n, self.config.c)) # (n, c)

        # prev_value
        self.prev_W = np.zeros((self.config.d, self.config.c))
        self.prev_Z = np.copy(self.Z)
        self.prev_U = np.copy(self.U)
        self.prev_F = np.copy(self.F)



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

            e_W, e_U, e_Z, e_F = self._update_error(F=F, Z=Z, U=U)
            Z_norm = np.linalg.norm(Z)
            U_norm = np.linalg.norm(U)

            self.config.lo = min(1.1*self.config.lo, self.config.lo_max)

            print(f"Iter {iter}: norm of W: {e_W}, norm of Z: {e_Z}, norm of U: {e_U}, norm of F: {e_F}, a: {self.a_i_list}")

        return [np.dot(m.X_i, m.W_i) for m in self.local_models]

    def _update_F(self, local_models, y):
        # X dot W_i for every view and sum up
        summation = np.sum([np.dot(self.X[i], m.W_i) for i, m in enumerate(local_models)], axis=0)

        # summation - y
        norm = np.linalg.norm((summation - y), ord=2, axis=1)
        threshold = 10**6
        norm = np.where(norm <= threshold, norm, self.config.eps)
        self.F = np.diag(0.5 * norm**-1)
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
        # eq 25
        v = self.config.v
        lo = self.config.lo

        term1 = np.linalg.inv(2*v*F + lo*np.identity(F.shape[0]))
        term2 = 2*v*np.dot(F, y) + lo*XW + lo*U

        self.Z = np.dot(np.linalg.inv(term1), term2)
        return self.Z

    def _update_U(self, U, XW, Z):
        self.U = U + XW - Z
        return self.U

    def _update_error(self, F=None, U=None, Z=None):
        W = np.concatenate([m.W_i for m in self.local_models], axis=0) # equals to np.vstack, current W

        # TODO: Meaning of || W ||_F ?
        error_norm = np.linalg.norm(W - self.prev_W, ord='fro')**2
        error_U = np.linalg.norm(U - self.prev_U, ord='fro') ** 2
        error_Z = np.linalg.norm(Z - self.prev_Z, ord='fro') ** 2
        error_F = np.linalg.norm(F - self.prev_F, ord='fro') ** 2

        self.prev_W = W
        self.prev_F = F if F is not None else None
        self.prev_U = U if U is not None else None
        self.prev_Z = Z if Z is not None else None

        return error_norm, error_U, error_Z, error_F


class RSMVFSLocal:
    """
    Responsible for calculating primal W_i
    All data related w.r.t. view i are implemented

    """
    def __init__(self, X_i, y, di, W_init=None, **config):
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
        self.S_i = self.S_w - self.S_b # independent of a_i

        # learnable
        self.W_i = (10 **-3) * np.eye(di, self.config.c) if W_init is None else W_init # (di, c)
        self.G_i = None # (n, n)

    def compute_G(self, W_i):
        # TODO: G_i의 매트릭스 shape이 어떻게 되먹은건가. 논문에서는 c X c라 되어 있는데 그러면 계산이 안됨.
        diag_elem = 1 / (np.linalg.norm(W_i, ord=2, axis=1) + 10**-10)
        G_i = np.diag(diag_elem) # (di, di)
        return G_i

    def update_W(self, a_i, Z, XW, U):
        # equation 23
        l1, l2, lo = self.config.lambda1, self.config.lambda2, self.config.lo

        # S_i = self._update_S_i(a_i)
        S_i = self.S_i
        X_i = self.X_i
        W_i = self.W_i

        G_i = self.compute_G(W_i) # first line in Algorithm 1.

        term_1 = (2 * l1 / a_i) * G_i + lo * self.X_T_X + l2 * S_i

        term_2 = np.dot(X_i.T, Z) + np.dot(self.X_T_X, W_i) - np.dot(X_i.T, XW) - np.dot(X_i.T, U)


        try:
            W_i = lo * np.dot(np.linalg.inv(term_1), term_2)
        
        except np.linalg.LinAlgError: # 가끔 term_1이  singular matrix가 되어서 역행렬 불가능할 때 psuedo-inverse 적용
            W_i = lo * np.dot(np.linalg.pinv(term_1), term_2)

        self.G_i = G_i
        self.W_i = W_i
        return W_i

    # def _update_S_i(self, a_i):
    #     S_b = (a_i ** 2) * np.linalg.multi_dot([self.X_i.T, self.y_chunk, self.X_i])  # eq 6
    #     I_y = np.identity(self.config.n) - self.y_chunk
    #
    #     S_w = (a_i ** 2) * np.linalg.multi_dot([self.X_i.T, I_y, self.X_i])  # eq 7
    #     S_i = (S_w - S_b) / (a_i ** 2)
    #
    #     s_b = np.linalg.multi_dot([self.X_i.T, self.y_chunk, self.X_i])
    #     s_w = np.linalg.multi_dot([self.X_i.T, I_y, self.X_i])
    #     self.S_i = S_i # eq10
    #     return S_i