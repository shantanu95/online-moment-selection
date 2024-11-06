import numpy as np
import pandas as pd

from gmm.gmm_equations import (
    GMMEquations,
    NuisanceEstimator,
    SequentialNuisanceEstimator,
)
from predictors import (
    Classifier,
    LogisticClassifier,
    MLPSklearnClassifer,
    MLPSklearnRegressor,
    Regressor,
    RidgeRegressor,
)


class JTPADataLATEGMMEqs(GMMEquations):
    def __init__(self, use_complete_data: bool = False) -> None:
        super().__init__()
        self.use_complete_data = use_complete_data

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

        # ignore the selection variable since we want to use all the data.

        beta = params[0]
        alpha = params[1]

        p_z1_w = nuisance.get(df, "P(Z=1|W)")
        e_y_z1_w = nuisance.get(df, "E(Y|Z=1,W)")
        e_y_z0_w = nuisance.get(df, "E(Y|Z=0,W)")
        e_y_z_w = Z * e_y_z1_w + (1 - Z) * e_y_z0_w

        m1 = (
            (Z / p_z1_w - (1 - Z) / (1 - p_z1_w)) * (Y - e_y_z_w)
            + (e_y_z1_w - e_y_z0_w)
            - (beta * alpha)
        )

        e_x_z1_w = nuisance.get(df, "E(X|Z=1,W)")
        e_x_z0_w = nuisance.get(df, "E(X|Z=0,W)")
        e_x_z_w = Z * e_x_z1_w + (1 - Z) * e_x_z0_w

        m2 = (
            (Z / p_z1_w - (1 - Z) / (1 - p_z1_w)) * (X - e_x_z_w)
            + (e_x_z1_w - e_x_z0_w)
            - alpha
        )

        if not self.use_complete_data:
            S = df["SEL"].values
            m1 *= S
            m2 *= 1 - S

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

_COVARIATE_NAMES = [
    "single",
    "demog_1",
    "demog_2",
    "sex_1",
    "sex_2",
    "race_1",
    "race_2",
    "race_3",
    "race_4",
    "race_5",
    "site_CC",
    "site_IN",
    "site_JC",
    "site_PR",
    "age_standardized",
]


class CompleteDataCrossFitNuisanceEstimator(NuisanceEstimator):
    def __init__(self) -> None:
        super().__init__()
        # For the JTPA study, P(X=1|Z=0, W) = 0.
        self.e_x_z0_w: float = 0
        self.e_x_z1_w: Classifier
        self.e_y_zw: Regressor
        self.p_z1_w: float = np.nan

    def _recompute_nuisances(self, df: pd.DataFrame) -> None:
        self.df_nu = pd.DataFrame(columns=_NUISANCE_FUNCTION_NAMES, dtype=float)
        self.df_nu["E(X|Z=0,W)"] = np.zeros(shape=(len(df),))

        df_len_half = int(len(df) / 2)
        self._populate_nuisance(
            df, train_interval=[0, df_len_half], predict_interval=[df_len_half, len(df)]
        )
        self._populate_nuisance(
            df, train_interval=[df_len_half, len(df)], predict_interval=[0, df_len_half]
        )

    def _populate_nuisance(
        self,
        df: pd.DataFrame,
        train_interval: tuple[int, int],
        predict_interval: tuple[int, int],
    ) -> None:
        df_train = df[train_interval[0] : train_interval[1]]

        self.p_z1_w = np.mean(df_train["Z"])
        df_train_z1 = df_train[df_train["Z"] == 1]
        self.e_x_z1_w.train(
            X=df_train_z1[_COVARIATE_NAMES].values, y=df_train_z1["X"].values
        )
        self.e_y_zw.train(
            X=df_train[["Z"] + _COVARIATE_NAMES].values, y=df_train["Y"].values
        )

        ps = predict_interval[0]
        pe = predict_interval[1]

        df_pred = df[ps:pe]
        df_nu = self.df_nu

        df_nu["P(Z=1|W)"].values[ps:pe] = self.p_z1_w
        covariates_arr_list = [df_pred[c] for c in _COVARIATE_NAMES]
        df_nu["E(Y|Z=0,W)"].values[ps:pe] = self.e_y_zw.predict(
            np.vstack([np.zeros_like(df_pred["Z"])] + covariates_arr_list).T
        )
        df_nu["E(Y|Z=1,W)"].values[ps:pe] = self.e_y_zw.predict(
            np.vstack([np.ones_like(df_pred["Z"])] + covariates_arr_list).T
        )

        df_nu["E(X|Z=0,W)"].values[ps:pe] = self.e_x_z0_w
        df_nu["E(X|Z=1,W)"].values[ps:pe] = self.e_x_z1_w.predict_proba(
            np.vstack(covariates_arr_list).T
        )

    @property
    def nuisance_function_names(self) -> list[str]:
        return _NUISANCE_FUNCTION_NAMES


class CompleteDataLearnedNuisanceEstimator(SequentialNuisanceEstimator):
    def __init__(
        self, horizon: int, batch_endpoints: list[int], initial_train_samples: int = 30
    ) -> None:
        super().__init__(horizon, batch_endpoints, initial_train_samples)

        # For the JTPA study, P(X=1|Z=0, W) = 0.
        self.e_x_z0_w: float = 0
        self.e_x_z1_w: Classifier
        self.e_y_zw: Regressor
        self.p_z1_w: float = np.nan

        self.COLUMN_NAMES = [
            "P(Z=1|W)",
            "E(Y|Z=1,W)",
            "E(Y|Z=0,W)",
            "E(X|Z=1,W)",
            "E(X|Z=0,W)",
        ]

        self.df_nu = pd.DataFrame(
            index=range(horizon), columns=self.COLUMN_NAMES, dtype=float
        )
        self.nuisance_train_last_size = 0

    @property
    def nuisance_function_names(self) -> list[str]:
        return _NUISANCE_FUNCTION_NAMES

    def _train_nuisance_estimators(self, df: pd.DataFrame, train_endpoint: int) -> None:
        df_train = df[:train_endpoint]

        if len(df_train) < self.nuisance_train_last_size:
            return

        self.nuisance_train_last_size = len(df_train)

        # Z is randomly assigned so Z \indep W. Thus, we just compute the sample mean.
        self.p_z1_w = np.mean(df_train["Z"])
        df_train_z1 = df_train[df_train["Z"] == 1]
        self.e_x_z1_w.train(
            X=df_train_z1[_COVARIATE_NAMES].values, y=df_train_z1["X"].values
        )
        self.e_y_zw.train(
            X=df_train[["Z"] + _COVARIATE_NAMES].values, y=df_train["Y"].values
        )

    def _populate_nuisances(self, df: pd.DataFrame, start: int, end: int) -> None:
        df_nu = self.df_nu
        ps = start
        pe = end

        df_pred = df[ps:pe]
        df_nu["P(Z=1|W)"].values[ps:pe] = self.p_z1_w

        covariates_arr_list = [df_pred[c] for c in _COVARIATE_NAMES]
        df_nu["E(Y|Z=0,W)"].values[ps:pe] = self.e_y_zw.predict(
            np.vstack([np.zeros_like(df_pred["Z"])] + covariates_arr_list).T
        )
        df_nu["E(Y|Z=1,W)"].values[ps:pe] = self.e_y_zw.predict(
            np.vstack([np.ones_like(df_pred["Z"])] + covariates_arr_list).T
        )

        df_nu["E(X|Z=0,W)"].values[ps:pe] = self.e_x_z0_w
        df_nu["E(X|Z=1,W)"].values[ps:pe] = self.e_x_z1_w.predict_proba(
            np.vstack(covariates_arr_list).T
        )


class SplitDataLearnedNuisanceEstimator(SequentialNuisanceEstimator):
    def __init__(
        self, horizon: int, batch_endpoints: list[int], initial_train_samples: int = 30
    ) -> None:
        super().__init__(horizon, batch_endpoints, initial_train_samples)

        # For the JTPA study, P(X=1|Z=0, W) = 0.
        self.e_x_z0_w: float = 0
        self.e_x_z1_w: Classifier
        self.e_y_zw: Regressor
        self.p_z1_w: float = np.nan

        self.COLUMN_NAMES = [
            "P(Z=1|W)",
            "E(Y|Z=1,W)",
            "E(Y|Z=0,W)",
            "E(X|Z=1,W)",
            "E(X|Z=0,W)",
        ]

        self.df_nu = pd.DataFrame(
            index=range(horizon), columns=self.COLUMN_NAMES, dtype=float
        )
        self.nuisance_train_last_size = 0
        self.treatment_nuisance_train_last_size = 0
        self.outcome_nuisance_train_last_size = 0

    @property
    def nuisance_function_names(self) -> list[str]:
        return _NUISANCE_FUNCTION_NAMES

    def _train_treatment_nuisance(self, df_train: pd.DataFrame) -> None:
        df_train_tmt = df_train[df_train["SEL"] == 0]
        if len(df_train_tmt) < self.treatment_nuisance_train_last_size:
            return

        if len(df_train_tmt) < self.INITIAL_TRAIN_SAMPLES:
            df_train_tmt = df_train[: self.INITIAL_TRAIN_SAMPLES]

        self.treatment_nuisance_train_last_size = len(df_train_tmt)

        df_train = df_train_tmt
        df_train_z1 = df_train[df_train["Z"] == 1]
        self.e_x_z1_w.train(
            X=df_train_z1[_COVARIATE_NAMES].values, y=df_train_z1["X"].values
        )

    def _train_outcome_nuisance(self, df_train: pd.DataFrame) -> None:
        df_train_out = df_train[df_train["SEL"] == 1]
        if len(df_train_out) < self.outcome_nuisance_train_last_size:
            return

        if len(df_train_out) < self.INITIAL_TRAIN_SAMPLES:
            df_train_out = df_train[: self.INITIAL_TRAIN_SAMPLES]

        self.outcome_nuisance_train_last_size = len(df_train_out)

        df_train = df_train_out
        self.e_y_zw.train(
            X=df_train[["Z"] + _COVARIATE_NAMES].values, y=df_train["Y"].values
        )

    def _train_nuisance_estimators(self, df: pd.DataFrame, train_endpoint: int) -> None:
        df_train = df[:train_endpoint]

        if len(df_train) < self.nuisance_train_last_size:
            return

        self.nuisance_train_last_size = len(df_train)

        # print(f"Training till: {train_endpoint}")

        # Z is randomly assigned so Z \indep W. Thus, we just compute the sample mean.
        self.p_z1_w = np.mean(df_train["Z"])
        self._train_treatment_nuisance(df_train)
        self._train_outcome_nuisance(df_train)

    def _populate_nuisances(self, df: pd.DataFrame, start: int, end: int) -> None:
        df_nu = self.df_nu
        ps = start
        pe = end

        df_pred = df[ps:pe]
        df_nu["P(Z=1|W)"].values[ps:pe] = self.p_z1_w

        covariates_arr_list = [df_pred[c] for c in _COVARIATE_NAMES]
        df_nu["E(Y|Z=0,W)"].values[ps:pe] = self.e_y_zw.predict(
            np.vstack([np.zeros_like(df_pred["Z"])] + covariates_arr_list).T
        )
        df_nu["E(Y|Z=1,W)"].values[ps:pe] = self.e_y_zw.predict(
            np.vstack([np.ones_like(df_pred["Z"])] + covariates_arr_list).T
        )

        df_nu["E(X|Z=0,W)"].values[ps:pe] = self.e_x_z0_w
        df_nu["E(X|Z=1,W)"].values[ps:pe] = self.e_x_z1_w.predict_proba(
            np.vstack(covariates_arr_list).T
        )


class LinearCompleteDataCrossFitNuisanceEstimator(
    CompleteDataCrossFitNuisanceEstimator
):
    def __init__(self) -> None:
        super().__init__()

        self.e_x_z1_w = LogisticClassifier(clip_low=0.02, clip_high=0.98)
        self.e_y_zw = RidgeRegressor()


class MLPCompleteDataCrossFitNuisanceEstimator(CompleteDataCrossFitNuisanceEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.e_x_z1_w = MLPSklearnClassifer(
            hidden_layer_sizes=[64, 64], clip_low=0.02, clip_high=0.98
        )
        self.e_y_zw = MLPSklearnRegressor(hidden_layer_sizes=[64, 64])


class LinearCompleteDataLearnedNuisanceEstimator(CompleteDataLearnedNuisanceEstimator):
    def __init__(
        self, horizon: int, batch_endpoints: list[int], initial_train_samples: int = 30
    ) -> None:
        super().__init__(horizon, batch_endpoints, initial_train_samples)

        self.e_x_z1_w = LogisticClassifier(clip_low=0.02, clip_high=0.98)
        self.e_y_zw = RidgeRegressor()


class MLPCompleteDataLearnedNuisanceEstimator(CompleteDataLearnedNuisanceEstimator):
    def __init__(
        self, horizon: int, batch_endpoints: list[int], initial_train_samples: int = 30
    ) -> None:
        super().__init__(horizon, batch_endpoints, initial_train_samples)

        self.e_x_z1_w = MLPSklearnClassifer(
            hidden_layer_sizes=[64, 64], clip_low=0.02, clip_high=0.98
        )
        self.e_y_zw = MLPSklearnRegressor(hidden_layer_sizes=[64, 64])


class LinearSplitDataLearnedNuisanceEstimator(SplitDataLearnedNuisanceEstimator):
    def __init__(
        self, horizon: int, batch_endpoints: list[int], initial_train_samples: int = 30
    ) -> None:
        super().__init__(horizon, batch_endpoints, initial_train_samples)

        self.e_x_z1_w = LogisticClassifier(clip_low=0.02, clip_high=0.98)
        self.e_y_zw = RidgeRegressor()


class MLPSplitDataLearnedNuisanceEstimator(SplitDataLearnedNuisanceEstimator):
    def __init__(
        self, horizon: int, batch_endpoints: list[int], initial_train_samples: int = 30
    ) -> None:
        super().__init__(horizon, batch_endpoints, initial_train_samples)

        self.e_x_z1_w = MLPSklearnClassifer(
            hidden_layer_sizes=[64, 64], clip_low=0.02, clip_high=0.98
        )
        self.e_y_zw = MLPSklearnRegressor(hidden_layer_sizes=[64, 64])
