import numpy as np
import time


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
        self.c = np.sqrt(self.c)

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


class FP_KalmanFilter:
    # state: x x_vel y y_vel w w_vel h h_vel  c1 ... cn

    def __init__(self, CLS : int):
        import filterpy.kalman as kal

        self.CLS = CLS
        self.f = kal.KalmanFilter(dim_x = 8 + CLS, dim_z = 4 + CLS)
        dt = .1
        var = .05
        noise = .1

        self.f.F = np.eye(8 + CLS)
        self.f.F[0, 1] = dt
        self.f.F[2, 3] = dt
        self.f.F[4, 5] = dt
        self.f.F[6, 7] = dt

        self.f.Q = np.array([
            [var/4, var/2,     0,     0,     0,     0,     0,     0] + ([0] * CLS),
            [var/2,   var,     0,     0,     0,     0,     0,     0] + ([0] * CLS),
            [    0,     0, var/4, var/2,     0,     0,     0,     0] + ([0] * CLS),
            [    0,     0, var/2,   var,     0,     0,     0,     0] + ([0] * CLS),
            [    0,     0,     0,     0, var/4, var/2,     0,     0] + ([0] * CLS),
            [    0,     0,     0,     0, var/2,   var,     0,     0] + ([0] * CLS),
            [    0,     0,     0,     0,     0,     0, var/4, var/2] + ([0] * CLS),
            [    0,     0,     0,     0,     0,     0, var/2,   var] + ([0] * CLS),
        ] + ([0, 0, 0, 0, 0, 0, 0, 0] + ([var/16] * CLS)) * CLS
        )

        self.f.H = np.zeros((4 + CLS, 8 + CLS))
        self.f.H[0, 0] = 1
        self.f.H[1, 2] = 1
        self.f.H[2, 4] = 1
        self.f.H[3, 6] = 1

        for i in range(CLS):
            self.f.H[4 + i, 8 + i] = 1

        self.f.R = np.eye(4 + CLS) * noise
        self.f.x = np.zeros((4 + CLS))
        self.f.P = np.eye(4) * 500.

    def update(self, measurement):
        self.f.update(measurement)

    def predict(self):
        self.f.predict()
        return self.f.get_prediction()


class SmoothingKalmanFilter:
    dimension : int
    Q : float
    timestamp : float
    velocity : np.ndarray
    position : np.ndarray
    min_accuracy : float = .1
    variance : np.ndarray


    def __init__(self, dim : int, q : float = 1.):
        self.dimension = dim
        self.variance = np.array([-1.] * dim)
        self.Q = q
        self.reset()

    def __repr__(self) -> str:
        return f'{"{"}p={self.position} v={self.velocity} v={self.variance} a={self.accuracy} min_a={self.min_accuracy} ts={self.timestamp} {"}"}'

    @property
    def accuracy(self) -> np.ndarray:
        if -1. in self.variance:
            raise Exception('The current smoothing kalman filter has not been initialized. Please call the "reset()"-function.')
        else:
            return np.sqrt(self.variance)

    def reset(self):
        self.force_state(
            np.zeros(self.dimension),
            np.zeros(self.dimension),
            np.zeros(self.dimension)
        )

    def force_state(self, position : np.ndarray, velocity : np.ndarray, accuracy : np.ndarray, timestamp : float | None = None) -> None:
        timestamp = timestamp or time.time()

        if position.shape != (self.dimension,):
            raise ValueError(f'The position vector must have a shape of ({self.dimension},).')
        elif velocity.shape != (self.dimension,):
            raise ValueError(f'The velocity vector must have a shape of ({self.dimension},).')
        elif accuracy.shape != (self.dimension,):
            raise ValueError(f'The accuracy vector must have a shape of ({self.dimension},).')
        else:
            self.position = position
            self.velocity = velocity
            self.variance = accuracy ** 2
            self.timestamp = timestamp

    def update(self, position : np.ndarray, accuracy : np.ndarray, timestamp : float | None = None):
        timestamp = timestamp or time.time()

        if position.shape != (self.dimension,):
            raise ValueError(f'The position vector must have a shape of ({self.dimension},).')
        elif accuracy.shape != (self.dimension,):
            raise ValueError(f'The accuracy vector must have a shape of ({self.dimension},).')
        elif timestamp < self.timestamp:
            raise ValueError(f'The given timestamp ({timestamp}) must be newer than the current timestamp ({self.timestamp}).')
        else:
            accuracy = np.array([max(x, self.min_accuracy) for x in accuracy])

            if any(x < 0. for x in self.variance):
                self.force_state(position, self.velocity, accuracy, timestamp)
            else:
                if (tdiff := timestamp - self.timestamp) > 0:
                    self.variance += tdiff * self.Q ** 2 / 1000.
                    self.timestamp = timestamp
                    self.velocity = position - self.position

                # kalman gain matrix K
                K = self.variance / (self.variance + self.accuracy ** 2)

                self.position += self.velocity * K
                self.variance *= 1 - K

    def estimate_position(self, timestamp : float | None = None) -> np.ndarray | None:
        timestamp = timestamp or time.time()

        if timestamp < self.timestamp:
            raise ValueError(f'The given timestamp ({timestamp}) must be newer than the current timestamp ({self.timestamp}).')
        else:
            return (timestamp - self.timestamp) * self.velocity * self.accuracy + self.position



import cv2
import random

skf = SmoothingKalmanFilter(1)
vals = []
x = 10

for i in range(1000):
    x += (random.random() - .5) * .4



for m in measurements:
    skf.update(m, np.array([.8]))
    time.sleep(.01)
    pos = skf.estimate_position()
    print(skf)
