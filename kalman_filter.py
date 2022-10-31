from cmath import sqrt
import cv2
import numpy as np


class UnscentedKalmanFilter:
    L : int # states number
    m : int # measurements number
    alpha : float # The alpha coefficient, characterize sigma-points dispersion around mean
    ki : float
    beta : float # The beta coefficient, characterize type of distribution (2 for normal one)
    lambda_ : float # scale factor
    c : float # Scale factor
    Wm : np.ndarray # mean weights
    Wc : np.ndarray # covariance weights
    x : np.ndarray # state matrix
    P : np.ndarray # covariance matrix
    q : float # std of process
    r : float # std of measurement
    Q : np.ndarray # process covariance
    R : np.ndarray # measurement covariance


    def __init__(self, L : int = 0):
        self.L = L
        self.m = 0

    def init(self):
        self.q = .05
        self.r = .3
        self.x = self.q * np.random.random((self.L, 1))
        self.P = np.identity(self.L)
        self.Q = np.identity(self.L) * (self.q * self.q)
        self.R = np.identity(self.L) * (self.r * self.r)
        self.alpha = 1e-3
        self.beta = 2.
        self.ki = 0.
        self.lambda_ = self.alpha * self.alpha * (self.L + self.ki) - self.L
        self.c = self.L + self.lambda_
        self.Wm = np.ones((1, 2 * self.L + 1)) * (.5 / self.c)
        self.Wm[0, 0] = self.lambda_ / self.c
        self.Wc = self.Wm.copy()
        self.Wc[0, 0] += 1 - self.alpha * self.alpha + self.beta
        self.c = sqrt(self.c).real

    def get_state(self) -> list[float]:
        return list(self.x[:, 0])

    def get_covariance(self) -> np.ndarray:
        return self.P

    def get_sigma_points(self):
        n = self.x.shape[0]
        A = np.linalg.cholesky(self.P)
        A = np.transpose(A * self.c)
        Y = np.ones((n, n))

        for j in range(0, n):
            Y[0:n, j:1] = self.x

        X = np.zeros((n, 2 * n + 1))
        X[0:n, 0:1] = self.x
        X[0:n, 1:n + 1] = Y + A
        X[0:n, n+1:2 * n + 1] = Y - A

        return X

    def unscented_transform(self, X : np.ndarray, n : int, R : np.ndarray):
        L = X.shape[1]
        y = np.zeros((n, 1))
        Y = np.zeros((n, L))
        X_rows : np.ndarray

        for k in range(0, L):
            X_rows = X[0:X.shape[0], k:k + 1]
            Y[0:Y.shape[0], k:k + 1] = X_rows
            y += Y[0: Y.shape[0], k:k + 1] * self.Wm[0, k]

        Y1 = Y - np.matmul(y, np.ones((1, L)))
        P = np.matmul(Y1, np.diag(self.Wc[0]))
        P = np.matmul(P, Y1.transpose())
        P += self.R

        return y, Y, P, Y1

    def update(self, measurements : list[float]):
        if not self.m and len(measurements):
            self.m = len(measurements)

            if not self.L:
                self.L = self.m

            self.init()

        z = np.zeros((self.m, 1))
        z[:, 0] = measurements

        X = self.get_sigma_points()
        x1, X1, P1, X2 = self.unscented_transform(X, self.L, self.Q)
        z1, Z1, P2, Z2 = self.unscented_transform(X1, self.m, self.R)
        P12 = np.matmul(np.matmul(X2, np.diag(self.Wc[0])), Z2.transpose())
        K = np.matmul(P12, np.linalg.inv(P2))

        self.x = x1 + np.matmul(K, z - z1)
        self.P = P1 - np.matmul(K, P12.transpose())

