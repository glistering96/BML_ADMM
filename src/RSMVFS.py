import numpy as np

class RSMVFSGlobal:
    def __init__(self, num_view, dim, num_class, num_instances,
                 eps=10^-6, converge_condition=10^-3, lo=1, lo_max = 10^6,
                 W_init=None, Z_init=None, U_init=None, F_init=None):
        """

        :param num_view: number of total view in the whole dataset
        :param dim: number of dimension in the ith view of X
        :param num_class: number of target class
        :param num_instances: number of instances
        :param lambda1: regularization param 1
        :param lambda2: regularization param 2
        :param eps: 
        :param converge_condition: convergence limit
        :param lo: lo parameter for scaled LM
        :param lo_max: maximum bound for lo
        :param W_init: initial value of W
        :param Z_init: initial value of Z
        :param U_init: initial value of U
        :param F_init: initial value of F
        """
        self.v = num_view
        self.d_i = dim
        self.c = num_class
        self.n = num_instances
        self.a_i = 1/self.v
        self.lo = lo
        self.lo_max = lo_max
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eps = eps
        self.converge_condition = converge_condition

        # initialize matrix
        self.Z = np.zeros(self.n, self.c) if Z_init is None else Z_init
        self.U = np.zeros(self.n, self.c) if U_init is None else U_init
        self.F = np.zeros(self.n, self.n) if F_init is None else F_init
        self.G = 1/2*np.linalg.norm(self.W_i, ord=2, axis=1)

    def run_opt(self, X_i, y, all_W, Z, U, XW):
        """
        :param X_i: ith view of matrix
        :param y: total label matrix
        :param all_W: list of other view's W, all_W = [W_1, W_2, ..., W_v]
        :param Z: mean of all Z
        :param U:
        :param XW:
        :return:
        """
        total_W_nrom = sum(np.linalg.norm(W, ord=2, axis=1) for W in all_W)
        a_i_k = np.sqrt(np.linalg.norm(self.W_i, ord=2, axis=1)) / total_W_nrom
        S_b = self.a_i*X_i.T*y*np.linalg.inv(y.T*y)*y.T*X_i*self.a_i # eq 6
        S_w = self.a_i*X_i.T*(np.identity(self.n) -y*np.linalg.inv(y.T*y)*y.T)*X_i*self.a_i # eq 7
        S_i = (S_w - S_b)/self.a_i^2

        # update for W_i
        self.W_i =

    def update_W(self, a_i_k, X_i, S_i, Z_bar, XW_bar, U_bar):
        term_1 = 2 * (self.lambda1 / a_i_k) * self.G + self.lo * (np.matmul(X_i.transpose(), X_i) + self.lambda2 * S_i)
        term_2 = (X_i.T * Z_bar + X_i.T * X_i * self.W_i - X_i.T * XW_bar - X_i.T * U_bar)
        return np.linalg.inv(term_1)*term_2

    def update_F(self, ):
    def update_Z(self, FF, y, XW_bar, U):


class RSMVFSLocal(RSMVFSGlobal):
    def __init__(self, W_init, num_view, dim, num_class, num_instances):
        super().__init__(num_view, dim, num_class, num_instances)
        self.W_i = 10 ^ -3 * np.eye(self.d_i, self.c) if W_init is None else W_init
        self.G_i = None

    def update_W(self, a_i_k, X_i, S_i, Z_bar, XW_bar, U_bar):
        self.G_i =
        term_1 = 2 * (self.lambda1 / a_i_k) * self.G_i + self.lo * (
                    np.matmul(X_i.transpose(), X_i) + self.lambda2 * S_i)
        term_2 = (X_i.T * Z_bar + X_i.T * X_i * self.W_i - X_i.T * XW_bar - X_i.T * U_bar)
        return np.linalg.inv(term_1) * term_2