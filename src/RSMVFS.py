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
        self.d_i_list = [d.shape(1) for d in self.X]
        self.local_models = [
            RSMVFSLocal(self.X[i], self.d_i_list[i], **config)
            for i in self.config.v
        ]
        self.a_i_list = [1/self.config.v for _ in range(self.config.v)]
        self.prev_W = np.zeros(self.config.d, self.config.c)

        # initialize matrix
        self.Z = np.zeros(self.config.n, self.config.c) if Z_init is None else Z_init
        self.U = np.zeros(self.config.n, self.config.c) if U_init is None else U_init
        self.F = np.zeros(self.config.n, self.config.n) if F_init is None else F_init
        self.U = np.zeros((self.config.n, self.config.c))

    def run(self):
        converged = False
        error = float('inf')

        while error < self.config.eps:
            XW = np.mean([np.dot(m.X_i, m.W_i) for m in self.local_models])

            for i, m in enumerate(self.local_models):
                m.update_W(self.a_i_list[i], self.Z, XW, self.U)

            F = self._update_F(self.local_models, self.y)
            Z = self._update_Z(F, self.y, XW, self.U)
            U = self._update_U(self.U, XW, Z)

            error = self._update_error()
            self.config.lo = min(1.1*self.config.lo, self.config.lo_max)

        return [np.dot(m.X_i, m.W_i) for m in self.local_models]

    def _update_F(self, local_models, y):
        summation = np.sum([np.dot(self.X[i], m) for i, m in enumerate(local_models)], axis=1)
        norm = (1/2)*np.linalg.norm(summation - y, axis=1)
        norm = np.where(norm <= self.config.eps, norm, 0)
        self.F = np.diag(1/norm)
        return self.F

    def _update_a_i(self, local_models):
        """ Must be called after updating W_i
            works as inplace method
         """
        W_norms = np.array([np.sqrt(np.linalg.norm(model.W_i, ord=2, axis=1) + self.config.eps)
                            for model in local_models])
        total_W_norm = np.sum(W_norms)

        self.a_i_list = W_norms / total_W_norm
        return self.a_i_list

    def _update_Z(self, F, y, XW, U):
        v = self.config.v
        lo = self.config.lo

        term1 = np.inv(2*v*F+lo*np.identity(F.shape[0]))
        term2 = 2*v*np.dot(F, y + lo*XW + lo*U)

        self.Z = np.dot(term1, term2)
        return self.Z

    def _update_U(self, U, XW, Z):
        self.U = U + XW - Z
        return self.U

    def _update_error(self):
        W = np.concatenate([m.W_i for m in self.local_models], axis=0) # equals to np.vstack

        # TODO: Meaning of || W ||_F ?
        norm = np.linalg.norm(W - self.prev_W)**2
        self.prev_W = W
        return norm


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
        S_i = self._update_S_i(a_i)

        self.G_i = 1/(2*np.linalg.norm(self.W_i, ord=2, axis=1) + self.config.eps)

        term_1 = 2 * (self.config.lambda1 / a_i) * self.G_i + self.config.lo * (
                    np.matmul(self.X_i.transpose(), self.X_i) + self.config.lambda2 * S_i)
        term_2 = (self.X_i.T * Z + self.X_i.T * self.X_i * self.W_i - self.X_i.T * XW - self.X_i.T * U)

        self.W_i = np.linalg.inv(term_1) * term_2
        return self.W_i

    def _update_S_i(self, a_i):
        S_b = (a_i ** 2) * np.linalg.multi_dot([self.X_i.T, self.y,
                                                np.linalg.inv(self.y.T.dot(self.y)), self.y.T, self.X_i])  # eq 6
        S_w = (a_i ** 2) * np.linalg.multi_dot([self.X_i.T,
                                                (np.identity(self.config.n)
                                                 - np.dot(self.y, np.linalg.inv(self.y.T.dot(self.y)))),
                                                self.y.T, self.X_i])  # eq 7
        S_i = (S_w - S_b) / (a_i ** 2)
        self.S_i = S_i
        return S_i