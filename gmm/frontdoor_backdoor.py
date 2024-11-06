import numpy as np
import pandas as pd

from causal_models.confounder_mediator_scms import LinearConfounderMediatorSCM
from gmm.gmm_equations import (
    GMMEquations,
    NuisanceEstimator,
    SequentialNuisanceEstimator,
)
from predictors import (
    Classifier,
    ConstantClassifier,
    LogisticClassifier,
    Regressor,
    RidgeRegressor,
)


class FrontdoorBackdoorGMMEqs(GMMEquations):
    @property
    def num_moments(self) -> int:
        return 2

    @property
    def num_params(self) -> int:
        return 1

    def get_moment_matrix(
        self,
        params: list[float],
        df: pd.DataFrame,
        nuisance: NuisanceEstimator,
    ) -> np.ndarray:
        X = df["X"].values
        M = df["M"].values
        Y = df["Y"].values
        S = df["SEL"].values

        t = params[0]

        p_x1_w = nuisance.get(df, "P(X=1|W)")
        e_y_x1_w = nuisance.get(df, "E[Y|X=1,W]")
        e_y_x0_w = nuisance.get(df, "E[Y|X=0,W]")
        e_y_x_w = X * e_y_x1_w + (1 - X) * e_y_x0_w

        m1 = S * (
            (X / p_x1_w - (1 - X) / (1 - p_x1_w)) * (Y - e_y_x_w)
            + (e_y_x1_w - e_y_x0_w)
            - t
        )

        p_m1_x1 = nuisance.get(df, "P(M=1|X=1)")
        p_m1_x0 = nuisance.get(df, "P(M=1|X=0)")
        p_m1_x = X * p_m1_x1 + (1 - X) * p_m1_x0  # P(M=1|X)
        p_m_x1 = M * p_m1_x1 + (1 - M) * (1 - p_m1_x1)  # P(M|X=1)
        p_m_x0 = M * p_m1_x0 + (1 - M) * ((1 - p_m1_x1))  # P(M|X=0)
        p_m_x = X * p_m_x1 + (1 - X) * p_m_x0  # P(M|X)

        e_y_x1_m1 = nuisance.get(df, "E[Y|X=1,M=1]")
        e_y_x0_m1 = nuisance.get(df, "E[Y|X=0,M=1]")
        e_y_x1_m0 = nuisance.get(df, "E[Y|X=1,M=0]")
        e_y_x0_m0 = nuisance.get(df, "E[Y|X=0,M=0]")

        e_y_x1_m = M * e_y_x1_m1 + (1 - M) * e_y_x1_m0
        e_y_x0_m = M * e_y_x0_m1 + (1 - M) * e_y_x0_m0
        e_y_x_m0 = X * e_y_x1_m0 + (1 - X) * e_y_x0_m0
        e_y_x_m1 = X * e_y_x1_m1 + (1 - X) * e_y_x0_m1
        # E(Y | X, M)
        e_y_x_m = X * (M * e_y_x1_m1 + (1 - M) * e_y_x1_m0) + (1 - X) * (
            M * e_y_x0_m1 + (1 - M) * e_y_x0_m0
        )

        p_x1 = nuisance.get(df, "P(X=1)")
        p_x0 = 1 - p_x1
        p_x = X * p_x1 + (1 - X) * p_x0

        m2 = (1 - S) * (
            ((p_m_x1 - p_m_x0) / p_m_x) * (Y - e_y_x_m)
            + ((X - (1 - X)) / p_x)
            * (
                e_y_x1_m * p_x1
                + e_y_x0_m * p_x0
                - (
                    e_y_x0_m0 * (1 - p_m1_x) * p_x0
                    + e_y_x0_m1 * p_m1_x * p_x0
                    + e_y_x1_m0 * (1 - p_m1_x) * p_x1
                    + e_y_x1_m1 * p_m1_x * p_x1
                )
            )
            + (
                (e_y_x_m1 * (p_m1_x1 - p_m1_x0))
                + (e_y_x_m0 * ((1 - p_m1_x1) - (1 - p_m1_x0)))
            )
            - t
        )
        return np.array([m1, m2])

    def get_moment_reweighting(self, weights: list[float]) -> np.ndarray:
        return np.diag(weights)

    def get_jacobian(
        self, params: list[float], weights: list[float], sel: np.ndarray
    ) -> np.ndarray:
        kappa = np.mean(sel)
        return np.array([[-weights[0] * kappa], [-weights[1] * (1 - kappa)]], float)


class FrontdoorBackdoorV2GMMEqs(GMMEquations):
    @property
    def num_moments(self) -> int:
        return 4

    @property
    def num_params(self) -> int:
        return 4

    def get_moment_matrix(
        self,
        params: list[float],
        df: pd.DataFrame,
        nuisance: NuisanceEstimator,
    ) -> np.ndarray:
        X = df["X"].values
        M = df["M"].values
        Y = df["Y"].values
        S = df["SEL"].values

        t = params[0]
        alpha = params[1]

        p_x1_w = nuisance.get(df, "P(X=1|W)")
        e_y_x1_w = nuisance.get(df, "E[Y|X=1,W]")
        e_y_x0_w = nuisance.get(df, "E[Y|X=0,W]")
        e_y_x_w = X * e_y_x1_w + (1 - X) * e_y_x0_w

        m1 = S * (
            (X / p_x1_w - (1 - X) / (1 - p_x1_w)) * (Y - e_y_x_w)
            + (e_y_x1_w - e_y_x0_w)
            - t
        )

        p_m1_x1 = nuisance.get(df, "P(M=1|X=1)")
        p_m1_x0 = nuisance.get(df, "P(M=1|X=0)")
        p_m1_x = X * p_m1_x1 + (1 - X) * p_m1_x0  # P(M=1|X)
        p_m_x1 = M * p_m1_x1 + (1 - M) * (1 - p_m1_x1)  # P(M|X=1)
        p_m_x0 = M * p_m1_x0 + (1 - M) * ((1 - p_m1_x1))  # P(M|X=0)
        p_m_x = X * p_m_x1 + (1 - X) * p_m_x0  # P(M|X)

        e_y_x1_m1 = nuisance.get(df, "E[Y|X=1,M=1]")
        e_y_x0_m1 = nuisance.get(df, "E[Y|X=0,M=1]")
        e_y_x1_m0 = nuisance.get(df, "E[Y|X=1,M=0]")
        e_y_x0_m0 = nuisance.get(df, "E[Y|X=0,M=0]")

        e_y_x1_m = M * e_y_x1_m1 + (1 - M) * e_y_x1_m0
        e_y_x0_m = M * e_y_x0_m1 + (1 - M) * e_y_x0_m0
        e_y_x_m0 = X * e_y_x1_m0 + (1 - X) * e_y_x0_m0
        e_y_x_m1 = X * e_y_x1_m1 + (1 - X) * e_y_x0_m1
        # E(Y | X, M)
        e_y_x_m = X * (M * e_y_x1_m1 + (1 - M) * e_y_x1_m0) + (1 - X) * (
            M * e_y_x0_m1 + (1 - M) * e_y_x0_m0
        )

        p_x1 = nuisance.get(df, "P(X=1)")
        p_x0 = 1 - p_x1
        p_x = X * p_x1 + (1 - X) * p_x0

        m2 = (1 - S) * (
            ((p_m_x1 - p_m_x0) / p_m_x) * (Y - e_y_x_m)
            + ((X - (1 - X)) / p_x)
            * (
                e_y_x1_m * p_x1
                + e_y_x0_m * p_x0
                - (
                    e_y_x0_m0 * (1 - p_m1_x) * p_x0
                    + e_y_x0_m1 * p_m1_x * p_x0
                    + e_y_x1_m0 * (1 - p_m1_x) * p_x1
                    + e_y_x1_m1 * p_m1_x * p_x1
                )
            )
            + (
                (e_y_x_m1 * (p_m1_x1 - p_m1_x0))
                + (e_y_x_m0 * ((1 - p_m1_x1) - (1 - p_m1_x0)))
            )
            - t
        )

        m3 = S * (Y - alpha)
        m4 = (1 - S) * (Y - alpha)
        return np.array([m1, m2, m3, m4])

    def get_moment_reweighting(self, weights: list[float]) -> np.ndarray:
        return np.diag(weights * 2)

    def get_jacobian(
        self, params: list[float], weights: list[float], sel: np.ndarray
    ) -> np.ndarray:
        kappa = np.mean(sel)
        return np.array(
            [
                [-weights[0] * kappa, 0],
                [-weights[1] * (1 - kappa), 0],
                [0, -weights[0] * kappa],
                [0, -weights[1] * (1 - kappa)],
            ],
            float,
        )


class BackdoorGMMEqs(GMMEquations):
    def __init__(self) -> None:
        super().__init__()

    @property
    def num_moments(self) -> int:
        return 1

    @property
    def num_params(self) -> int:
        return 1

    def get_moment_matrix(
        self,
        params: list[float],
        df: pd.DataFrame,
        nuisance: NuisanceEstimator,
    ) -> np.ndarray:
        X = df["X"].values
        Y = df["Y"].values
        S = df["SEL"].values

        assert np.mean(S) == 1

        t = params[0]

        p_x1_w = nuisance.get(df, "P(X=1|W)")
        e_y_x1_w = nuisance.get(df, "E[Y|X=1,W]")
        e_y_x0_w = nuisance.get(df, "E[Y|X=0,W]")
        e_y_x_w = X * e_y_x1_w + (1 - X) * e_y_x0_w

        m1 = S * (
            (X / p_x1_w - (1 - X) / (1 - p_x1_w)) * (Y - e_y_x_w)
            + (e_y_x1_w - e_y_x0_w)
            - t
        )
        return np.array([m1])

    def get_moment_reweighting(self, weights: list[float]) -> np.ndarray:
        raise NotImplementedError()

    def get_jacobian(self, weights: list[float], sel: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class FrontdoorGMMEqs(GMMEquations):
    def __init__(self) -> None:
        super().__init__()

    @property
    def num_moments(self) -> int:
        return 1

    @property
    def num_params(self) -> int:
        return 1

    def get_moment_matrix(
        self,
        params: list[float],
        df: pd.DataFrame,
        nuisance: NuisanceEstimator,
    ) -> np.ndarray:
        X = df["X"].values
        M = df["M"].values
        Y = df["Y"].values
        S = df["SEL"].values

        assert np.mean(S) == 0

        t = params[0]

        p_m1_x1 = nuisance.get(df, "P(M=1|X=1)")
        p_m1_x0 = nuisance.get(df, "P(M=1|X=0)")
        p_m1_x = X * p_m1_x1 + (1 - X) * p_m1_x0  # P(M=1|X)
        p_m_x1 = M * p_m1_x1 + (1 - M) * (1 - p_m1_x1)  # P(M|X=1)
        p_m_x0 = M * p_m1_x0 + (1 - M) * ((1 - p_m1_x1))  # P(M|X=0)
        p_m_x = X * p_m_x1 + (1 - X) * p_m_x0  # P(M|X)

        e_y_x1_m1 = nuisance.get(df, "E[Y|X=1,M=1]")
        e_y_x0_m1 = nuisance.get(df, "E[Y|X=0,M=1]")
        e_y_x1_m0 = nuisance.get(df, "E[Y|X=1,M=0]")
        e_y_x0_m0 = nuisance.get(df, "E[Y|X=0,M=0]")

        e_y_x1_m = M * e_y_x1_m1 + (1 - M) * e_y_x1_m0
        e_y_x0_m = M * e_y_x0_m1 + (1 - M) * e_y_x0_m0
        e_y_x_m0 = X * e_y_x1_m0 + (1 - X) * e_y_x0_m0
        e_y_x_m1 = X * e_y_x1_m1 + (1 - X) * e_y_x0_m1
        # E(Y | X, M)
        e_y_x_m = X * (M * e_y_x1_m1 + (1 - M) * e_y_x1_m0) + (1 - X) * (
            M * e_y_x0_m1 + (1 - M) * e_y_x0_m0
        )

        p_x1 = nuisance.get(df, "P(X=1)")
        p_x0 = 1 - p_x1
        p_x = X * p_x1 + (1 - X) * p_x0

        m2 = (1 - S) * (
            ((p_m_x1 - p_m_x0) / p_m_x) * (Y - e_y_x_m)
            + ((X - (1 - X)) / p_x)
            * (
                e_y_x1_m * p_x1
                + e_y_x0_m * p_x0
                - (
                    e_y_x0_m0 * (1 - p_m1_x) * p_x0
                    + e_y_x0_m1 * p_m1_x * p_x0
                    + e_y_x1_m0 * (1 - p_m1_x) * p_x1
                    + e_y_x1_m1 * p_m1_x * p_x1
                )
            )
            + (
                (e_y_x_m1 * (p_m1_x1 - p_m1_x0))
                + (e_y_x_m0 * ((1 - p_m1_x1) - (1 - p_m1_x0)))
            )
            - t
        )
        return np.array([m2])

    def get_moment_reweighting(self, weights: list[float]) -> np.ndarray:
        raise NotImplementedError()

    def get_jacobian(self, weights: list[float], sel: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


_NUISANCE_FUNCTION_NAMES = [
    "P(X=1|W)",
    "E[Y|X=1,W]",
    "E[Y|X=0,W]",
    "P(M=1|X=1)",
    "P(M=1|X=0)",
    "P(X=1)",
    "E[Y|X=1,M=1]",
    "E[Y|X=0,M=1]",
    "E[Y|X=1,M=0]",
    "E[Y|X=0,M=0]",
]


class OracleNuisanceEstimator(NuisanceEstimator):
    def __init__(
        self, true_scm: LinearConfounderMediatorSCM, num_oracle_samples: int = 1000000
    ) -> None:
        super().__init__()

        self.true_scm = true_scm

        # Find these in closed-form later. For now, just sample a large dataset
        # and compute this empirically.
        df_train = true_scm.generate_data_samples(num_samples=num_oracle_samples)

        self._nu_vals = {
            "E[Y|X=1,M=1]": np.mean(
                df_train[np.logical_and(df_train["X"] == 1, df_train["M"] == 1)]["Y"]
            ),
            "E[Y|X=0,M=1]": np.mean(
                df_train[np.logical_and(df_train["X"] == 0, df_train["M"] == 1)]["Y"]
            ),
            "E[Y|X=1,M=0]": np.mean(
                df_train[np.logical_and(df_train["X"] == 1, df_train["M"] == 0)]["Y"]
            ),
            "E[Y|X=0,M=0]": np.mean(
                df_train[np.logical_and(df_train["X"] == 0, df_train["M"] == 0)]["Y"]
            ),
            "P(X=1)": np.mean(df_train["X"]),
        }

    def _recompute_nuisances(self, df: pd.DataFrame) -> None:
        W = df["W"]

        self.df_nu = pd.DataFrame()

        self.df_nu["P(X=1|W)"] = df["P*(X=1|W)"]
        self.df_nu["E[Y|X=1,W]"] = self.true_scm.b * W + self.true_scm.get_true_ate()
        self.df_nu["E[Y|X=0,W]"] = self.true_scm.b * W

        self.df_nu["P(M=1|X=1)"] = self.true_scm.m1
        self.df_nu["P(M=1|X=0)"] = self.true_scm.m0

        for k, v in self._nu_vals.items():
            self.df_nu[k] = v

    @property
    def nuisance_function_names(self) -> list[str]:
        return _NUISANCE_FUNCTION_NAMES


class LearnedNuisanceEstimator(SequentialNuisanceEstimator):
    def __init__(
        self, horizon: int, batch_endpoints: list[int], initial_train_samples: int = 30
    ) -> None:
        super().__init__(horizon, batch_endpoints, initial_train_samples)

        self.e_y_x_w: Regressor
        self.e_y_x_m: Regressor
        self.p_x1_w: Classifier
        self.p_x1_w_trained: bool = False

        self.COLUMN_NAMES = [
            "P(X=1|W)",
            "E[Y|X=1,W]",
            "E[Y|X=0,W]",
            "P(M=1|X=1)",
            "P(M=1|X=0)",
            "P(X=1)",
            "E[Y|X=1,M=1]",
            "E[Y|X=0,M=1]",
            "E[Y|X=1,M=0]",
            "E[Y|X=0,M=0]",
            # used for debugging.
            "W",
            "P*(X=1|W)",
        ]

        self.df_nu = pd.DataFrame(
            index=range(horizon), columns=self.COLUMN_NAMES, dtype=float
        )
        self.backdoor_last_train_size = 0
        self.frontdoor_last_train_size = 0

    @property
    def nuisance_function_names(self) -> list[str]:
        return _NUISANCE_FUNCTION_NAMES

    def _train_backdoor_nuisance(self, df_train: pd.DataFrame) -> None:
        df_train_bd = df_train[df_train["SEL"] == 1]

        # print(
        #     f"len train_bd: {len(df_train_bd)}, last train: {self.backdoor_last_train_size}"
        # )

        if len(df_train_bd) == 0:
            df_train_bd = df_train[: self.INITIAL_TRAIN_SAMPLES]

        if len(df_train_bd) <= self.backdoor_last_train_size:
            return

        self.backdoor_last_train_size = len(df_train_bd)

        if np.unique(df_train_bd["X"]).shape[0] < 2:
            # This means the target label does not have both ones and zeros. Sklearn
            # throws an error. So for now, just output a probability of 0.5.
            # This should happen very rarely.
            self.p_x1_w_trained = False
        else:
            # \hat{P}(X=1|W).
            self.p_x1_w.train(X=df_train_bd[["W"]].values, y=df_train_bd["X"].values)
            self.p_x1_w_trained = True

        # \hat{E}(Y|X, W).
        self.e_y_x_w.train(X=df_train_bd[["X", "W"]].values, y=df_train_bd["Y"].values)

    def _train_frontdoor_nuisance(self, df_train: pd.DataFrame) -> None:
        def get_clipped(mean_val) -> float:
            if np.isnan(mean_val):
                return 0.5

            return np.clip(mean_val, 0.05, 0.95)

        df_train_fd = df_train[df_train["SEL"] == 0]

        if len(df_train_fd) == 0:
            df_train_fd = df_train[: self.INITIAL_TRAIN_SAMPLES]

        if len(df_train_fd) <= self.frontdoor_last_train_size:
            return

        self.frontdoor_last_train_size = len(df_train_fd)

        # \hat{P}(M=1|X).
        self.p_m1_x1 = get_clipped(np.mean(df_train_fd[df_train_fd["X"] == 1]["M"]))
        self.p_m1_x0 = get_clipped(np.mean(df_train_fd[df_train_fd["X"] == 0]["M"]))

        # \hat{E}(Y|X, M).
        self.e_y_x_m.train(X=df_train_fd[["X", "M"]].values, y=df_train_fd["Y"].values)

    def _train_nuisance_estimators(self, df: pd.DataFrame, train_endpoint: int) -> None:
        df_train = df[:train_endpoint]

        self._train_backdoor_nuisance(df_train)
        self._train_frontdoor_nuisance(df_train)

        # \hat{P}(X=1). We observe `X` in both data sources, so all samples can be
        # used for estimating P(X=1).
        self.p_x1 = min(max(np.mean(df_train["X"]), 0.05), 0.95)

    def _populate_nuisances(self, df: pd.DataFrame, start: int, end: int) -> None:
        df_nu = self.df_nu
        ps = start
        pe = end

        df_pred = df[ps:pe]

        # used for debugging
        df_nu["W"].values[ps:pe] = df_pred["W"].values
        df_nu["P*(X=1|W)"].values[ps:pe] = df_pred["P*(X=1|W)"].values

        p_x1_w = (
            self.p_x1_w if self.p_x1_w_trained else ConstantClassifier(prob_value=0.5)
        )
        df_nu["P(X=1|W)"].values[ps:pe] = p_x1_w.predict_proba(df_pred[["W"]].values)

        df_nu["E[Y|X=1,W]"].values[ps:pe] = self.e_y_x_w.predict(
            np.vstack([np.ones_like(df_pred["X"]), df_pred["W"]]).T
        )
        df_nu["E[Y|X=0,W]"].values[ps:pe] = self.e_y_x_w.predict(
            np.vstack([np.zeros_like(df_pred["X"]), df_pred["W"]]).T
        )

        df_nu["P(M=1|X=1)"].values[ps:pe] = self.p_m1_x1
        df_nu["P(M=1|X=0)"].values[ps:pe] = self.p_m1_x0

        df_nu["E[Y|X=1,M=1]"].values[ps:pe] = self.e_y_x_m.predict(
            np.vstack(
                [
                    np.ones_like(df_pred["X"]),
                    np.ones_like(df_pred["M"]),
                ]
            ).T
        )
        df_nu["E[Y|X=0,M=1]"].values[ps:pe] = self.e_y_x_m.predict(
            np.vstack(
                [
                    np.zeros_like(df_pred["X"]),
                    np.ones_like(df_pred["M"]),
                ]
            ).T
        )
        df_nu["E[Y|X=1,M=0]"].values[ps:pe] = self.e_y_x_m.predict(
            np.vstack(
                [
                    np.ones_like(df_pred["X"]),
                    np.zeros_like(df_pred["M"]),
                ]
            ).T
        )
        df_nu["E[Y|X=0,M=0]"].values[ps:pe] = self.e_y_x_m.predict(
            np.vstack(
                [
                    np.zeros_like(df_pred["X"]),
                    np.zeros_like(df_pred["M"]),
                ]
            ).T
        )

        df_nu["P(X=1)"].values[ps:pe] = self.p_x1


class LinearNuisanceEstimator(LearnedNuisanceEstimator):
    def __init__(
        self, horizon: int, batch_endpoints: list[int], initial_train_samples: int = 30
    ) -> None:
        super().__init__(horizon, batch_endpoints, initial_train_samples)

        self.e_y_x_w = RidgeRegressor()
        self.e_y_x_m = RidgeRegressor()
        self.p_x1_w = LogisticClassifier(clip_low=0.1, clip_high=0.9)
