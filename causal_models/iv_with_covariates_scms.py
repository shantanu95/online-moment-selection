from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit

from causal_models.scm import SCM
from utils import rff_utils
from utils.sigmoid_utils import sigmoid_expectation_monte_carlo


class UniformIVCovariatesSCM(SCM):
    def __init__(
        self,
        wz: float,
        wx: float,
        wy: float,
        ux: float,
        uy: float,
        zx: float,
        xy: float,
        bias_x: float,
        var_nw: float,
        var_nu: float,
        var_nx: float,
        var_ny: float,
    ) -> None:
        super().__init__()
        self.wz = wz
        self.wx = wx
        self.wy = wy
        self.zx = zx
        self.wy = wy
        self.ux = ux
        self.uy = uy
        self.xy = xy
        self.bias_x = bias_x
        self.var_nw = var_nw
        self.var_nu = var_nu
        self.var_nx = var_nx
        self.var_ny = var_ny

    def get_true_ate(self) -> float:
        return self.xy

    def generate_data_samples(self, num_samples: int) -> pd.DataFrame:
        noise_w = np.random.uniform(
            low=-self.var_nw, high=self.var_nw, size=(num_samples,)
        )
        noise_u = np.random.uniform(
            low=-self.var_nu, high=self.var_nu, size=(num_samples,)
        )
        noise_x = np.random.uniform(
            low=-self.var_nx, high=self.var_nx, size=(num_samples,)
        )
        noise_y = np.random.uniform(
            low=-self.var_ny, high=self.var_ny, size=(num_samples,)
        )

        def get_W(noise):
            return noise

        def get_U(noise):
            return noise

        def get_Z(W):
            p_z_w = expit(self.wz * W)
            return np.random.binomial(1, p_z_w), p_z_w

        def get_X(W, U, Z, noise):
            p_x_uwz = expit(
                self.ux * U + self.wx * W + self.zx * Z + noise + self.bias_x
            )
            X = np.random.binomial(1, p_x_uwz)

            p_x_w_z0 = sigmoid_expectation_monte_carlo(
                mu=self.wx * W + self.bias_x, noise_samples=self.ux * U + noise
            )
            p_x_w_z1 = sigmoid_expectation_monte_carlo(
                mu=self.wx * W + self.zx + self.bias_x,
                noise_samples=self.ux * U + noise,
            )
            return X, p_x_w_z0, p_x_w_z1

        def get_Y(W, U, X, noise, p_x_w_z0, p_x_w_z1):
            Y = self.wy * W + self.uy * U + self.xy * X + noise
            e_y_w_z0 = self.wy * W + self.xy * p_x_w_z0
            e_y_w_z1 = self.wy * W + self.xy * p_x_w_z1
            return Y, e_y_w_z0, e_y_w_z1

        W = get_W(noise_w)
        U = get_U(noise_u)
        Z, p_z_w = get_Z(W=W)
        X, p_x_w_z0, p_x_w_z1 = get_X(W=W, U=U, Z=Z, noise=noise_x)
        Y, e_y_w_z0, e_y_w_z1 = get_Y(
            W=W, U=U, X=X, noise=noise_y, p_x_w_z0=p_x_w_z0, p_x_w_z1=p_x_w_z1
        )

        return pd.DataFrame(
            {
                "W": W,
                "Z": Z,
                "X": X,
                "Y": Y,
                "P*(Z=1|W)": p_z_w,
                "E*(X|Z=0,W)": p_x_w_z0,
                "E*(X|Z=1,W)": p_x_w_z1,
                "E*(Y|Z=0,W)": e_y_w_z0,
                "E*(Y|Z=1,W)": e_y_w_z1,
            }
        )

    def __str__(self) -> str:
        return f"wz={self.wz}, wx={self.wx}, wy={self.wy}, ux={self.ux}, uy={self.uy}, zx={self.zx}, xy={self.xy}"


class RFFIVCovariatesSCM(SCM):
    def __init__(
        self,
        w_z_num_features: int,
        wz_x_num_features: int,
        u_x_num_features: int,
        u_y_num_features: int,
        wx_y_num_features: int,
        var_nw: float,
        var_nu: float,
        var_ny: float,
        bias_z0_x: float,
        bias_z1_x: float,
        bias_x0_y: float,
        bias_x1_y: float,
    ) -> None:
        super().__init__()
        self.var_nw = var_nw
        self.var_nu = var_nu
        self.var_ny = var_ny

        self.w_z_coefs = rff_utils.sample_rff_coefs(
            num_functions=1, num_features=w_z_num_features
        )
        self.wz_x_coefs = rff_utils.sample_rff_coefs(
            num_functions=1, num_features=wz_x_num_features, x_dim=2
        )
        self.u_x_coefs = rff_utils.sample_rff_coefs(
            num_functions=1, num_features=u_x_num_features
        )

        self.u_y_coefs = rff_utils.sample_rff_coefs(
            num_functions=1, num_features=u_y_num_features
        )
        self.wx_y_coefs = rff_utils.sample_rff_coefs(
            num_functions=1, num_features=wx_y_num_features, x_dim=2
        )

        self.bias_z0_x = bias_z0_x
        self.bias_z1_x = bias_z1_x
        self.bias_x0_y = bias_x0_y
        self.bias_x1_y = bias_x1_y

        self.true_ate: Optional[float] = None
        self.optimal_kappa: Optional[float] = None

        self.logits_u_mc: Optional[np.ndarray] = None

    def get_true_ate(self) -> float:
        if self.true_ate is None:
            raise ValueError("True ATE has not been set")

        return self.true_ate

    def _init_logits_u_mc(self) -> None:
        MC_SAMPLES = 20000
        noise_u_copy = np.random.uniform(
            low=-self.var_nu, high=self.var_nu, size=(MC_SAMPLES,)
        )
        self.logits_u_mc = rff_utils.compute_rff_fn(
            x=noise_u_copy[:, None], rff_coefs=self.u_x_coefs
        )[0]

    def generate_data_samples(self, num_samples: int) -> pd.DataFrame:
        noise_w = np.random.uniform(
            low=-self.var_nw, high=self.var_nw, size=(num_samples,)
        )
        noise_u = np.random.uniform(
            low=-self.var_nu, high=self.var_nu, size=(num_samples,)
        )
        noise_y = np.random.uniform(
            low=-self.var_ny, high=self.var_ny, size=(num_samples,)
        )

        def get_W(noise):
            return noise

        def get_U(noise):
            return noise

        def get_Z(W):
            p_z_w = expit(
                rff_utils.compute_rff_fn(x=W[:, None], rff_coefs=self.w_z_coefs)[0]
            )
            return np.random.binomial(1, p_z_w), p_z_w

        def get_X(W, U, Z):
            logits_u = rff_utils.compute_rff_fn(
                x=U[:, None],
                rff_coefs=self.u_x_coefs,
            )[0]
            logits_wz = (
                rff_utils.compute_rff_fn(
                    x=np.vstack([W, Z]).T, rff_coefs=self.wz_x_coefs
                )[0]
                + Z * self.bias_z1_x
                + (1 - Z) * self.bias_z0_x
            )
            p_x_uwz = expit(logits_u + logits_wz)
            X = np.random.binomial(1, p_x_uwz)

            logits_wz0 = (
                rff_utils.compute_rff_fn(
                    x=np.vstack([W, np.zeros_like(W)]).T, rff_coefs=self.wz_x_coefs
                )[0]
                + self.bias_z0_x
            )
            p_x_w_z0 = sigmoid_expectation_monte_carlo(
                mu=logits_wz0, noise_samples=self.logits_u_mc
            )

            logits_wz1 = (
                rff_utils.compute_rff_fn(
                    x=np.vstack([W, np.ones_like(W)]).T, rff_coefs=self.wz_x_coefs
                )[0]
                + self.bias_z1_x
            )
            p_x_w_z1 = sigmoid_expectation_monte_carlo(
                mu=logits_wz1, noise_samples=self.logits_u_mc
            )
            return X, p_x_w_z0, p_x_w_z1

        def get_Y(W, U, X, noise, p_x_w_z0, p_x_w_z1):
            U_Y_effect = rff_utils.compute_rff_fn(
                x=U[:, None], rff_coefs=self.u_y_coefs
            )[0]
            WX_Y_effect = (
                rff_utils.compute_rff_fn(
                    x=np.vstack([W, X]).T, rff_coefs=self.wx_y_coefs
                )[0]
                + X * self.bias_x1_y
                + (1 - X) * self.bias_x0_y
            )
            Y = U_Y_effect + WX_Y_effect + noise

            WX0_Y_effect = (
                rff_utils.compute_rff_fn(
                    x=np.vstack([W, np.zeros_like(W)]).T, rff_coefs=self.wx_y_coefs
                )[0]
                + self.bias_x0_y
            )
            WX1_Y_effect = (
                rff_utils.compute_rff_fn(
                    x=np.vstack([W, np.ones_like(W)]).T, rff_coefs=self.wx_y_coefs
                )[0]
                + self.bias_x1_y
            )
            expectation_U_effect = np.mean(U_Y_effect)

            e_y_w_z0 = (
                expectation_U_effect
                + WX1_Y_effect * p_x_w_z0
                + WX0_Y_effect * (1 - p_x_w_z0)
            )
            e_y_w_z1 = (
                expectation_U_effect
                + WX1_Y_effect * p_x_w_z1
                + WX0_Y_effect * (1 - p_x_w_z1)
            )
            return Y, e_y_w_z0, e_y_w_z1

        W = get_W(noise_w)
        U = get_U(noise_u)
        Z, p_z_w = get_Z(W=W)
        X, p_x_w_z0, p_x_w_z1 = get_X(W=W, U=U, Z=Z)
        Y, e_y_w_z0, e_y_w_z1 = get_Y(
            W=W, U=U, X=X, noise=noise_y, p_x_w_z0=p_x_w_z0, p_x_w_z1=p_x_w_z1
        )

        return pd.DataFrame(
            {
                "W": W,
                "Z": Z,
                "X": X,
                "Y": Y,
                "P*(Z=1|W)": p_z_w,
                "E*(X|Z=0,W)": p_x_w_z0,
                "E*(X|Z=1,W)": p_x_w_z1,
                "E*(Y|Z=0,W)": e_y_w_z0,
                "E*(Y|Z=1,W)": e_y_w_z1,
            }
        )
