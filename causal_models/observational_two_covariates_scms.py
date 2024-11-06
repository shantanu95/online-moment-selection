import numpy as np
import pandas as pd
from scipy.special import expit

from causal_models.scm import SCM
from utils.sigmoid_utils import sigmoid_expectation_monte_carlo


class UniformObservationalDataTwoCovariatesSCM(SCM):
    def __init__(
        self,
        beta: float,
        var_nu: float,
        var_nw: float,
        var_ny: float,
        ux: float,
        wx: float,
        uy: float,
        wy: float,
    ) -> None:
        self.beta = beta
        self.var_nu = var_nu
        self.var_nw = var_nw
        self.var_ny = var_ny
        self.ux = ux
        self.wx = wx
        self.uy = uy
        self.wy = wy

    def get_true_ate(self) -> float:
        return self.beta

    def generate_data_samples(self, num_samples: int) -> pd.DataFrame:
        noise_w = np.random.uniform(
            low=-self.var_nw, high=self.var_nw, size=(num_samples,)
        )
        noise_u = np.random.uniform(
            low=-self.var_nu, high=self.var_nu, size=(num_samples,)
        )
        U_monte_carlo = np.random.uniform(
            low=-self.var_nu, high=self.var_nu, size=(1000,)
        )
        noise_y = np.random.uniform(
            low=-self.var_ny, high=self.var_ny, size=(num_samples,)
        )

        def get_U(noise):
            return noise

        def get_W(noise):
            return noise

        def get_X(U, W, U_monte_carlo):
            p_x_uw = expit(self.wx * W + self.ux * U)
            p_x_w = sigmoid_expectation_monte_carlo(
                mu=self.wx * W, noise_samples=self.ux * U_monte_carlo
            )
            return np.random.binomial(1, p_x_uw), p_x_uw, p_x_w

        def get_Y(X, U, W, noise):
            return self.uy * U + self.wy * W + self.beta * X + noise

        U = get_U(noise=noise_u)
        W = get_W(noise=noise_w)
        X, p_x_uw, p_x_w = get_X(U=U, W=W, U_monte_carlo=U_monte_carlo)
        Y = get_Y(X, U, W, noise_y)

        return pd.DataFrame(
            {"U": U, "W": W, "X": X, "Y": Y, "P*(X=1|U,W)": p_x_uw, "P*(X=1|W)": p_x_w}
        )

    def __str__(self) -> str:
        return f"beta={self.beta}, ux={self.ux}, wx={self.wx}, uy={self.uy}, wy={self.wy}, var_nu={self.var_nu}, var_nw={self.var_nw}, var_ny={self.var_ny}"
