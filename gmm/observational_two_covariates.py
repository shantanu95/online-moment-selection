import numpy as np
import pandas as pd

from causal_models.observational_two_covariates_scms import (
    UniformObservationalDataTwoCovariatesSCM,
)
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


class ObservationalTwoCovariatesGMMEqs(GMMEquations):
    @property
    def num_moments(self) -> int:
        return 3

    @property
    def num_params(self) -> int:
        return 2

    def get_moment_matrix(
        self,
        params: list[float],
        df: pd.DataFrame,
        nuisance: NuisanceEstimator,
    ) -> np.ndarray:
        X = df["X"].values
        Y = df["Y"].values
        S = df["SEL"].values

        beta = params[0]
        alpha = params[1]

        p_x1_uw = nuisance.get(df, "P(X=1|U,W)")
        e_y_x1_uw = nuisance.get(df, "E[Y|X=1,U,W]")
        e_y_x0_uw = nuisance.get(df, "E[Y|X=0,U,W]")
        e_y_x_uw = X * e_y_x1_uw + (1 - X) * e_y_x0_uw

        m1 = S * (
            (X / p_x1_uw - (1 - X) / (1 - p_x1_uw)) * (Y - e_y_x_uw)
            + (e_y_x1_uw - e_y_x0_uw)
            - beta
        )

        p_x1_w = nuisance.get(df, "P(X=1|W)")
        e_y_x1_w = nuisance.get(df, "E[Y|X=1,W]")
        e_y_x0_w = nuisance.get(df, "E[Y|X=0,W]")
        e_y_x_w = X * e_y_x1_w + (1 - X) * e_y_x0_w

        m2 = S * (
            (X / p_x1_w - (1 - X) / (1 - p_x1_w)) * (Y - e_y_x_w)
            + (e_y_x1_w - e_y_x0_w)
            - alpha
        )
        m3 = (1 - S) * (
            (X / p_x1_w - (1 - X) / (1 - p_x1_w)) * (Y - e_y_x_w)
            + (e_y_x1_w - e_y_x0_w)
            - alpha
        )

        return np.array([m1, m2, m3])

    def get_moment_reweighting(self, weights: list[float]) -> np.ndarray:
        return np.array(
            [
                [weights[0], weights[0], 0],
                [weights[0], weights[0], 0],
                [0, 0, weights[-1]],
            ]
        )

    def get_jacobian(
        self, params: list[float], weights: list[float], sel: np.ndarray
    ) -> np.ndarray:
        kappa = np.mean(sel)
        return np.array(
            [
                [-weights[0] * kappa, 0],
                [0, -weights[0] * kappa],
                [0, -weights[-1] * (1 - kappa)],
            ],
            float,
        )


class ObservationalTwoCovariatesSingleMomentGMMEqs(GMMEquations):
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

        beta = params[0]

        p_x1_uw = nuisance.get(df, "P(X=1|U,W)")
        e_y_x1_uw = nuisance.get(df, "E[Y|X=1,U,W]")
        e_y_x0_uw = nuisance.get(df, "E[Y|X=0,U,W]")
        e_y_x_uw = X * e_y_x1_uw + (1 - X) * e_y_x0_uw

        m1 = S * (
            (X / p_x1_uw - (1 - X) / (1 - p_x1_uw)) * (Y - e_y_x_uw)
            + (e_y_x1_uw - e_y_x0_uw)
            - beta
        )

        return np.array([m1])

    def get_moment_reweighting(self, weights: list[float]) -> np.ndarray:
        return np.array([[-1]], float)

    def get_jacobian(
        self, params: list[float], weights: list[float], sel: np.ndarray
    ) -> np.ndarray:
        return np.array([[-1]], float)


_NUISANCE_FUNCTION_NAMES = [
    "P(X=1|U,W)",
    "E[Y|X=1,U,W]",
    "E[Y|X=0,U,W]",
    "P(X=1|W)",
    "E[Y|X=1,W]",
    "E[Y|X=0,W]",
]


class OracleNuisanceEstimator(NuisanceEstimator):
    def __init__(self, true_scm: UniformObservationalDataTwoCovariatesSCM) -> None:
        super().__init__()
        self.true_scm = true_scm

    def _recompute_nuisances(self, df: pd.DataFrame) -> None:
        U = df["U"]
        W = df["W"]

        self.df_nu = pd.DataFrame()

        self.df_nu["P(X=1|U,W)"] = df["P*(X=1|U,W)"]
        self.df_nu["E[Y|X=1,U,W]"] = (
            self.true_scm.uy * U + self.true_scm.wy * W + self.true_scm.beta
        )
        self.df_nu["E[Y|X=0,U,W]"] = self.true_scm.uy * U + self.true_scm.wy * W

        self.df_nu["P(X=1|W)"] = df["P*(X=1|W)"]
        self.df_nu["E[Y|X=1,W]"] = self.true_scm.wy * W + self.true_scm.beta
        self.df_nu["E[Y|X=0,W]"] = self.true_scm.wy * W

    @property
    def nuisance_function_names(self) -> list[str]:
        return _NUISANCE_FUNCTION_NAMES


class LearnedNuisanceEstimator(SequentialNuisanceEstimator):
    def __init__(
        self, horizon: int, batch_endpoints: list[int], initial_train_samples: int = 30
    ) -> None:
        super().__init__(horizon, batch_endpoints, initial_train_samples)

        self.e_y_x_uw: Regressor
        self.p_x1_uw: Classifier
        self.p_x1_uw_trained: bool = False

        self.e_y_x_w: Regressor
        self.p_x1_w: Classifier
        self.p_x1_w_trained: bool = False

        self.COLUMN_NAMES = [
            "P(X=1|U,W)",
            "E[Y|X=1,U,W]",
            "E[Y|X=0,U,W]",
            "P(X=1|W)",
            "E[Y|X=1,W]",
            "E[Y|X=0,W]",
            # used for debugging.
            "P*(X=1|U,W)",
            "P*(X=1|W)",
        ]
        self.df_nu = pd.DataFrame(
            index=range(horizon), columns=self.COLUMN_NAMES, dtype=float
        )
        self.source_1_last_train_size = 0
        self.source_2_last_train_size = 0

    @property
    def nuisance_function_names(self) -> list[str]:
        return _NUISANCE_FUNCTION_NAMES

    def _train_source_1_nuisance(self, df_train: pd.DataFrame) -> None:
        df_train_1 = df_train[df_train["SEL"] == 1]

        if len(df_train_1) == 0:
            df_train_1 = df_train[: self.INITIAL_TRAIN_SAMPLES]

        df_train = df_train_1
        if len(df_train) <= self.source_1_last_train_size:
            return

        self.source_1_last_train_size = len(df_train)

        if np.unique(df_train["X"]).shape[0] < 2:
            # This means the target label does not have both ones and zeros. Sklearn
            # throws an error. So for now, just output a probability of 0.5.
            # This should happen very rarely.
            self.p_x1_uw_trained = False
        else:
            self.p_x1_uw.train(X=df_train[["U", "W"]].values, y=df_train["X"].values)
            self.p_x1_uw_trained = True

        self.e_y_x_uw.train(X=df_train[["X", "U", "W"]].values, y=df_train["Y"].values)

    def _train_source_2_nuisance(self, df_train: pd.DataFrame) -> None:
        # The whole dataset can be used since source 1 returns a superset
        # of the observable variables in source 2.

        if len(df_train) <= self.source_2_last_train_size:
            return

        self.source_2_last_train_size = len(df_train)

        if np.unique(df_train["X"]).shape[0] < 2:
            # This means the target label does not have both ones and zeros. Sklearn
            # throws an error. So for now, just output a probability of 0.5.
            # This should happen very rarely.
            self.p_x1_w_trained = False
        else:
            self.p_x1_w.train(X=df_train[["W"]].values, y=df_train["X"].values)
            self.p_x1_w_trained = True

        self.e_y_x_w.train(X=df_train[["X", "W"]].values, y=df_train["Y"].values)

    def _train_nuisance_estimators(self, df: pd.DataFrame, train_endpoint: int) -> None:
        df_train = df[:train_endpoint]

        self._train_source_1_nuisance(df_train)
        self._train_source_2_nuisance(df_train)

    def _populate_nuisances(self, df: pd.DataFrame, start: int, end: int) -> None:
        df_nu = self.df_nu
        ps = start
        pe = end

        df_pred = df[ps:pe]

        # used for debugging
        df_nu["P*(X=1|U,W)"].values[ps:pe] = df_pred["P*(X=1|U,W)"].values
        df_nu["P*(X=1|W)"].values[ps:pe] = df_pred["P*(X=1|W)"].values

        p_x1_uw = (
            self.p_x1_uw if self.p_x1_uw_trained else ConstantClassifier(prob_value=0.5)
        )
        df_nu["P(X=1|U,W)"].values[ps:pe] = p_x1_uw.predict_proba(
            df_pred[["U", "W"]].values
        )
        df_nu["E[Y|X=1,U,W]"].values[ps:pe] = self.e_y_x_uw.predict(
            np.vstack([np.ones_like(df_pred["X"]), df_pred["U"], df_pred["W"]]).T
        )
        df_nu["E[Y|X=0,U,W]"].values[ps:pe] = self.e_y_x_uw.predict(
            np.vstack([np.zeros_like(df_pred["X"]), df_pred["U"], df_pred["W"]]).T
        )

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


class LinearNuisanceEstimator(LearnedNuisanceEstimator):
    def __init__(
        self, horizon: int, batch_endpoints: list[int], initial_train_samples: int = 30
    ) -> None:
        super().__init__(horizon, batch_endpoints, initial_train_samples)

        self.e_y_x_uw = RidgeRegressor()
        self.p_x1_uw = LogisticClassifier(clip_low=0.05, clip_high=0.95)

        self.e_y_x_w = RidgeRegressor()
        self.p_x1_w = LogisticClassifier(clip_low=0.05, clip_high=0.95)
