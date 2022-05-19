import numpy as np

class RSMVFConfig:
    def __init__(self, num_view, num_class, num_instances, lambda1, lambda2,
                 eps=10^-6, converge_condition=10^-3, lo=1, lo_max = 10^6):
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

        # initialize matrix
        self.Z = np.zeros(self.config.n, self.config.c) if Z_init is None else Z_init
        self.U = np.zeros(self.config.n, self.config.c) if U_init is None else U_init

        # global variables
        self.d_i_list = [d.shape(1) for d in self.X]
        self.local_models = [
            RSMVFSLocal(self.X[i], self.d_i_list[i], **config)
            for i in self.config.v
        ]
        self.a_i_list = [1/self.config.v for _ in range(self.config.v)]

    def update_F(self):
        np.sum([a*m.X_i*m.W_i for a, m in zip(self.a_i_list, self.local_models)])

    def update_a_i(self):
        """ Must be called after updating W_i
            works as inplace method
         """
        W_norms = np.array([np.sqrt(np.linalg.norm(model.W_i, ord=2, axis=1) + self.config.eps)
                            for model in self.local_models])
        total_W_norm = sum(W_norms)
        self.a_i_list = W_norms / total_W_norm

    def update_Z(self, FF, y, XW_bar, U):
        pass


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
        self.W_i = (10 **-3) * np.eye(dim, self.config.c) if W_init is None else W_init
        self.G_i = 1/(2*np.linalg.norm(self.W_i, ord=2, axis=1) + self.config.eps)

    def update_W(self, a_i, Z_bar, XW_bar, U_bar):
        S_b = a_i*self.X_i.T*self.y*np.linalg.inv(self.y.T*self.y)*self.y.T*self.X_i*a_i # eq 6
        S_w = a_i*self.X_i.T*(np.identity(self.config.n) -
                              self.y*np.linalg.inv(self.y.T*self.y)*self.y.T)*self.X_i*a_i # eq 7
        S_i = (S_w - S_b)/(a_i**2)

        self.G_i = 1/(2*np.linalg.norm(self.W_i, ord=2, axis=1) + self.config.eps)

        term_1 = 2 * (self.config.lambda1 / a_i) * self.G_i + self.config.lo * (
                    np.matmul(self.X_i.transpose(), self.X_i) + self.config.lambda2 * S_i)
        term_2 = (self.X_i.T * Z_bar + self.X_i.T * self.X_i * self.W_i - self.X_i.T * XW_bar - self.X_i.T * U_bar)

        return np.linalg.inv(term_1) * term_2