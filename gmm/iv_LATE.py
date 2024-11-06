import numpy as np
import pandas as pd

from gmm.gmm_equations import (
    GMMEquations,
    NuisanceEstimator,
    SequentialNuisanceEstimator,
)
from predictors import (
    Classifier,
    LinearRegressor,
    LogisticClassifier,
    MLPSklearnClassifer,
    MLPSklearnRegressor,
    Regressor,
)


class LATETwoSampleGMMEqs(GMMEquations):
    @property
    def num_moments(self) -> int:
        return 2

    @property
    def num_params(self) -> int:
        return 2

    def get_moment_matrix(
        self, params: list[float], df: pd.DataFrame, nuisance: NuisanceEstimator
    ) -> np.ndarray:
        Z = df["Z"].values
        X = df["X"].values
        Y = df["Y"].values
        S = df["SEL"].values

        beta = params[0]
        alpha = params[1]

        p_z1_w = nuisance.get(df, "P(Z=1|W)")
        e_y_z1_w = nuisance.get(df, "E(Y|Z=1,W)")
        e_y_z0_w = nuisance.get(df, "E(Y|Z=0,W)")
        e_y_z_w = Z * e_y_z1_w + (1 - Z) * e_y_z0_w

        m1 = S * (
            (Z / p_z1_w - (1 - Z) / (1 - p_z1_w)) * (Y - e_y_z_w)
            + (e_y_z1_w - e_y_z0_w)
            - (beta * alpha)
        )

        e_x_z1_w = nuisance.get(df, "E(X|Z=1,W)")
        e_x_z0_w = nuisance.get(df, "E(X|Z=0,W)")
        e_x_z_w = Z * e_x_z1_w + (1 - Z) * e_x_z0_w

        m2 = (1 - S) * (
            (Z / p_z1_w - (1 - Z) / (1 - p_z1_w)) * (X - e_x_z_w)
            + (e_x_z1_w - e_x_z0_w)
            - alpha
        )
        return np.array([m1, m2])

    def get_moment_reweighting(self, weights: list[float]) -> np.ndarray:
        return np.diag(weights)

    def get_jacobian(
        self, params: list[float], weights: list[float], sel: np.ndarray
    ) -> np.ndarray:
        kappa = np.mean(sel)
        beta = params[0]
        alpha = params[1]
        return np.array(
            [
                [-weights[0] * kappa * alpha, -weights[0] * kappa * beta],
                [0, -weights[-1] * (1 - kappa)],
            ],
            float,
        )


_NUISANCE_FUNCTION_NAMES = [
    "P(Z=1|W)",
    "E(Y|Z=1,W)",
    "E(Y|Z=0,W)",
    "E(X|Z=1,W)",
    "E(X|Z=0,W)",
]


class OracleNuisanceEstimator(NuisanceEstimator):
    def _recompute_nuisances(self, df: pd.DataFrame) -> None:
        self.df_nu = pd.DataFrame()

        self.df_nu["P(Z=1|W)"] = df["P*(Z=1|W)"]

        self.df_nu["E(Y|Z=1,W)"] = df["E*(Y|Z=1,W)"]
        self.df_nu["E(Y|Z=0,W)"] = df["E*(Y|Z=0,W)"]

        self.df_nu["E(X|Z=1,W)"] = df["E*(X|Z=1,W)"]
        self.df_nu["E(X|Z=0,W)"] = df["E*(X|Z=0,W)"]

    @property
    def nuisance_function_names(self) -> list[str]:
        return _NUISANCE_FUNCTION_NAMES


class LearnedNuisanceEstimator(SequentialNuisanceEstimator):
    def __init__(
        self, horizon: int, batch_endpoints: list[int], initial_train_samples: int = 30
    ) -> None:
        super().__init__(horizon, batch_endpoints, initial_train_samples)

        self.p_z1_w: Classifier
        self.e_x_zw: Classifier
        self.e_y_zw: Regressor

        self.COLUMN_NAMES = [
            "P(Z=1|W)",
            "E(Y|Z=1,W)",
            "E(Y|Z=0,W)",
            "E(X|Z=1,W)",
            "E(X|Z=0,W)",
            # used for debugging.
            "P*(Z=1|W)",
            "E*(X|Z=1,W)",
            "E*(X|Z=0,W)",
            "E*(Y|Z=1,W)",
            "E*(Y|Z=0,W)",
        ]

        self.df_nu = pd.DataFrame(
            index=range(horizon), columns=self.COLUMN_NAMES, dtype=float
        )

        self.common_nuisance_train_last_size = 0
        self.treatment_nuisance_train_last_size = 0
        self.outcome_nuisance_train_last_size = 0

    @property
    def nuisance_function_names(self) -> list[str]:
        return _NUISANCE_FUNCTION_NAMES

    def _train_common_nuisance(self, df_train: pd.DataFrame) -> None:
        if len(df_train) < self.common_nuisance_train_last_size:
            return

        self.common_nuisance_train_last_size = len(df_train)
        self.p_z1_w.train(X=df_train[["W"]].values, y=df_train["Z"].values)

    def _train_treatment_nuisance(self, df_train: pd.DataFrame) -> None:
        df_train_tmt = df_train[df_train["SEL"] == 0]
        if len(df_train_tmt) < self.treatment_nuisance_train_last_size:
            return

        if len(df_train_tmt) < self.INITIAL_TRAIN_SAMPLES:
            df_train_tmt = df_train[: self.INITIAL_TRAIN_SAMPLES]

        self.treatment_nuisance_train_last_size = len(df_train_tmt)

        df_train = df_train_tmt
        self.e_x_zw.train(X=df_train[["Z", "W"]].values, y=df_train["X"].values)

    def _train_outcome_nuisance(self, df_train: pd.DataFrame) -> None:
        df_train_out = df_train[df_train["SEL"] == 1]

        if len(df_train_out) < self.outcome_nuisance_train_last_size:
            return

        if len(df_train_out) < self.INITIAL_TRAIN_SAMPLES:
            df_train_out = df_train[: self.INITIAL_TRAIN_SAMPLES]

        self.outcome_nuisance_train_last_size = len(df_train_out)

        df_train = df_train_out
        self.e_y_zw.train(X=df_train[["Z", "W"]].values, y=df_train["Y"].values)

    def _train_nuisance_estimators(self, df: pd.DataFrame, train_endpoint: int) -> None:
        df_train = df[:train_endpoint]

        self._train_common_nuisance(df_train)
        self._train_outcome_nuisance(df_train)
        self._train_treatment_nuisance(df_train)

    def _populate_nuisances(self, df: pd.DataFrame, start: int, end: int) -> None:
        df_nu = self.df_nu
        ps = start
        pe = end

        df_pred = df[ps:pe]

        # used for debugging.
        df_nu["P*(Z=1|W)"].values[ps:pe] = df_pred["P*(Z=1|W)"].values
        df_nu["E*(X|Z=1,W)"].values[ps:pe] = df_pred["E*(X|Z=1,W)"].values
        df_nu["E*(X|Z=0,W)"].values[ps:pe] = df_pred["E*(X|Z=0,W)"].values
        df_nu["E*(Y|Z=1,W)"].values[ps:pe] = df_pred["E*(Y|Z=1,W)"].values
        df_nu["E*(Y|Z=0,W)"].values[ps:pe] = df_pred["E*(Y|Z=0,W)"].values

        df_nu["P(Z=1|W)"].values[ps:pe] = self.p_z1_w.predict_proba(
            X=df_pred[["W"]].values
        )

        df_nu["E(Y|Z=0,W)"].values[ps:pe] = self.e_y_zw.predict(
            np.vstack([np.zeros_like(df_pred["Z"]), df_pred["W"]]).T
        )
        df_nu["E(Y|Z=1,W)"].values[ps:pe] = self.e_y_zw.predict(
            np.vstack([np.ones_like(df_pred["Z"]), df_pred["W"]]).T
        )

        df_nu["E(X|Z=0,W)"].values[ps:pe] = self.e_x_zw.predict_proba(
            np.vstack([np.zeros_like(df_pred["Z"]), df_pred["W"]]).T
        )
        df_nu["E(X|Z=1,W)"].values[ps:pe] = self.e_x_zw.predict_proba(
            np.vstack([np.ones_like(df_pred["Z"]), df_pred["W"]]).T
        )


class LogisticNuisanceEstimator(LearnedNuisanceEstimator):
    def __init__(
        self, horizon: int, batch_endpoints: list[int], initial_train_samples: int = 30
    ) -> None:
        super().__init__(horizon, batch_endpoints, initial_train_samples)

        self.p_z1_w = LogisticClassifier(clip_low=0.05, clip_high=0.95)
        self.e_y_zw = LinearRegressor()
        self.e_x_zw = LogisticClassifier(clip_low=0.05, clip_high=0.95)


class MLPNuisanceEstimator(LearnedNuisanceEstimator):
    def __init__(
        self, horizon: int, batch_endpoints: list[int], initial_train_samples: int = 30
    ) -> None:
        super().__init__(horizon, batch_endpoints, initial_train_samples)

        self.p_z1_w = MLPSklearnClassifer(
            hidden_layer_sizes=[64, 64], clip_low=0.05, clip_high=0.95
        )
        self.e_y_zw = MLPSklearnRegressor(hidden_layer_sizes=[64, 64])
        self.e_x_zw = MLPSklearnClassifer(
            hidden_layer_sizes=[64, 64], clip_low=0.05, clip_high=0.95
        )
