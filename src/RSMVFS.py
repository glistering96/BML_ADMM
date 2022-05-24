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

        """

        self.X = X # list of view matrix
        self.y = y
        self.config = RSMVFConfig(**config)

        # global variables
        self.d_i_list = [d.shape[1] for d in self.X]
        self.local_models = [
            RSMVFSLocal(self.X[i], y, self.d_i_list[i], **config)
            for i in range(self.config.v)
        ]
        self.a_i_list = [1/self.config.v for _ in range(self.config.v)]
        self.prev_W = np.zeros((self.config.d, self.config.c))

        # initialize matrix
        self.Z = np.zeros((self.config.n, self.config.c)) if Z_init is None else Z_init
        self.U = np.zeros((self.config.n, self.config.c)) if U_init is None else U_init
        self.F = np.zeros((self.config.n, self.config.n)) if F_init is None else F_init
        self.U = np.zeros((self.config.n, self.config.c))

    def run(self):
        error = float('inf')
        iter = 0

        while error > self.config.eps:
            iter += 1
            XW = np.mean([np.dot(m.X_i, m.W_i) for m in self.local_models], axis=0)

            for i, m in enumerate(self.local_models):
                m.update_W(self.a_i_list[i], self.Z, XW, self.U)

            self._update_a_i(self.local_models)
            F = self._update_F(self.local_models, self.y)
            Z = self._update_Z(F, self.y, XW, self.U)
            U = self._update_U(self.U, XW, Z)

            error, W_norm = self._update_error()
            self.config.lo = min(1.1*self.config.lo, self.config.lo_max)

            print(f"Iter {iter}: norm of W: {W_norm}, a: {self.a_i_list}")

        return [np.dot(m.X_i, m.W_i) for m in self.local_models]

    def _update_F(self, local_models, y):
        summation = np.sum([np.dot(self.X[i], m.W_i) for i, m in enumerate(local_models)], axis=0)
        norm = (1/2)*np.linalg.norm(summation - y, axis=1)
        norm = np.where(norm <= self.config.eps, norm, 0)
        self.F = np.diag(1/(norm + self.config.eps))
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
        error_norm = np.linalg.norm(W - self.prev_W)**2
        self.prev_W = W
        return error_norm, np.linalg.norm(W)


class RSMVFSLocal:
    """
    Responsible for calculating primal W_i
    """
    def __init__(self, X_i, y, dim, W_init=None, **config):
        # fixed
        self.config = RSMVFConfig(**config)
        self.X_i = X_i
        self.y = y

        # learnable
        self.W_i = (10 **-3) * np.eye(dim, self.config.c) if W_init is None else W_init # W_i
        self.G_i = 1/(2*np.linalg.norm(self.W_i, ord=2, axis=1) + self.config.eps)

        # calculated
        self.S_i = None

    def update_W(self, a_i, Z, XW, U):
        # equation 23
        l1, l2, lo = self.config.lambda1, self.config.lambda2, self.config.lo
        S_i = self._update_S_i(a_i)
        X_i = self.X_i
        W_i = self.W_i

        X_T_X = np.dot(X_i.T, X_i)

        diag_elem = 1/(2*np.linalg.norm(W_i, ord=2, axis=1) + self.config.eps)
        G_i = np.diag(diag_elem)

        term_1 = (2 * l1 / a_i) * G_i + lo * X_T_X + l2 * S_i

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
        yTy_inv = np.linalg.inv(np.dot(self.y.T, self.y))
        y_chunk = np.linalg.multi_dot([self.y, yTy_inv, self.y.T])

        S_b = (a_i ** 2) * np.linalg.multi_dot([self.X_i.T, y_chunk, self.X_i])  # eq 6
        I_y = np.identity(self.config.n) - y_chunk

        S_w = (a_i ** 2) * np.linalg.multi_dot([self.X_i.T, I_y, self.X_i])  # eq 7
        S_i = (S_w - S_b) / (a_i ** 2)
        self.S_i = S_i # eq10
        return S_i