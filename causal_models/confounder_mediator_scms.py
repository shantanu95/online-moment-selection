import numpy as np
import pandas as pd
from scipy.special import expit

from causal_models.scm import SCM


class LinearConfounderMediatorSCM(SCM):
    def __init__(
        self,
        a: float,
        b: float,
        d: float,
        w: float,
        x: float,
        m0: float,
        m1: float,
        y: float,
    ):
        assert m0 > 0 and m0 < 1
        assert m1 > 0 and m1 < 1

        self.a = a
        self.b = b
        self.d = d
        self.w = w
        self.x = x
        self.m0 = m0
        self.m1 = m1
        self.y = y
        self._true_ate = self.a * (self.m1 - self.m0)

    def get_true_ate(self) -> float:
        return self._true_ate

    def generate_data_samples(self, num_samples: int) -> pd.DataFrame:
        u_w = np.random.normal(scale=np.sqrt(self.w), size=(num_samples,))
        u_y = np.random.normal(scale=np.sqrt(self.y), size=(num_samples,))

        def get_W(u):
            return u

        def get_X(W):
            p_x_w = expit(self.d * W + self.x)
            return np.random.binomial(1, p_x_w), p_x_w

        def get_M(X):
            p_m_x = X * self.m1 + (1 - X) * self.m0
            return np.random.binomial(1, p_m_x), p_m_x

        def get_Y(M, W, u):
            return self.a * M + self.b * W + u

        W = get_W(u_w)
        X, p_x_w = get_X(W)
        M, p_m_x = get_M(X)
        Y = get_Y(M, W, u_y)

        return pd.DataFrame(
            {"W": W, "X": X, "M": M, "Y": Y, "P*(X=1|W)": p_x_w, "P*(M=1|X)": p_m_x}
        )

    def __str__(self) -> str:
        return f"a={self.a}, b={self.b}, d={self.d}, w={self.w}, x={self.x}, m0={self.m0}, m1={self.m1}, y={self.y}"
