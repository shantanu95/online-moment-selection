import numpy as np
import pandas as pd

from gmm.gmm_equations import NuisanceEstimator, SequentialNuisanceEstimator

from predictors import LogisticClassifier, Classifier

from gmm.observational_two_covariates import _NUISANCE_FUNCTION_NAMES

_COVARIATE_NAMES_PRECISE = ["PS_P2"]
_COVARIATE_NAMES_CRUDE = ["PS_C2"]
_COVARIATE_NAMES_ALL = _COVARIATE_NAMES_PRECISE + _COVARIATE_NAMES_CRUDE


class ValidationCrossFitNuisanceEstimator(NuisanceEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.e_y_x_uw: Classifier
        self.e_x_uw: Classifier

    @property
    def nuisance_function_names(self) -> list[str]:
        return _NUISANCE_FUNCTION_NAMES

    def _recompute_nuisances(self, df: pd.DataFrame) -> None:
        self.df_nu = pd.DataFrame(
            index=range(len(df)), columns=_NUISANCE_FUNCTION_NAMES, dtype=float
        )

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

        self.e_y_x_uw.train(
            X=df_train[["X"] + _COVARIATE_NAMES_ALL].values,
            y=df_train["Y"].values,
        )
        self.e_x_uw.train(
            X=df_train[_COVARIATE_NAMES_ALL].values,
            y=df_train["X"].values,
        )

        ps = predict_interval[0]
        pe = predict_interval[1]

        df_pred = df[ps:pe]
        df_nu = self.df_nu

        cov_arr_list = [df_pred[c].values for c in _COVARIATE_NAMES_ALL]
        df_nu["E[Y|X=0,U,W]"].values[ps:pe] = self.e_y_x_uw.predict_proba(
            X=np.vstack([np.zeros_like(df_pred["X"])] + cov_arr_list).T
        )
        df_nu["E[Y|X=1,U,W]"].values[ps:pe] = self.e_y_x_uw.predict_proba(
            X=np.vstack([np.ones_like(df_pred["X"])] + cov_arr_list).T
        )
        df_nu["P(X=1|U,W)"].values[ps:pe] = self.e_x_uw.predict_proba(
            X=df_pred[_COVARIATE_NAMES_ALL].values
        )


class CrudeConfoundersCrossFitNuisanceEstimator(NuisanceEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.e_y_x_w: Classifier
        self.e_x_w: Classifier

    @property
    def nuisance_function_names(self) -> list[str]:
        return _NUISANCE_FUNCTION_NAMES

    def _recompute_nuisances(self, df: pd.DataFrame) -> None:
        self.df_nu = pd.DataFrame(
            index=range(len(df)), columns=_NUISANCE_FUNCTION_NAMES, dtype=float
        )

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

        self.e_y_x_w.train(
            X=df_train[["X"] + _COVARIATE_NAMES_CRUDE].values,
            y=df_train["Y"].values,
        )
        self.e_x_w.train(
            X=df_train[_COVARIATE_NAMES_CRUDE].values,
            y=df_train["X"].values,
        )

        ps = predict_interval[0]
        pe = predict_interval[1]

        df_pred = df[ps:pe]
        df_nu = self.df_nu

        cov_arr_list = [df_pred[c].values for c in _COVARIATE_NAMES_CRUDE]
        df_nu["E[Y|X=0,W]"].values[ps:pe] = self.e_y_x_w.predict_proba(
            X=np.vstack([np.zeros_like(df_pred["X"])] + cov_arr_list).T
        )
        df_nu["E[Y|X=1,W]"].values[ps:pe] = self.e_y_x_w.predict_proba(
            X=np.vstack([np.ones_like(df_pred["X"])] + cov_arr_list).T
        )
        df_nu["P(X=1|W)"].values[ps:pe] = self.e_x_w.predict_proba(
            X=df_pred[_COVARIATE_NAMES_CRUDE].values
        )


class CombinedDataCrossFitNuisanceEstimator(NuisanceEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.e_y_x_uw: Classifier
        self.e_x_uw: Classifier

        self.e_y_x_w_val: Classifier
        self.e_x_w_val: Classifier

        self.e_y_x_w_main: Classifier
        self.e_x_w_main: Classifier

    @property
    def nuisance_function_names(self) -> list[str]:
        return _NUISANCE_FUNCTION_NAMES

    def _recompute_nuisances(self, df: pd.DataFrame) -> None:
        self.df_nu = pd.DataFrame(
            index=range(len(df)), columns=_NUISANCE_FUNCTION_NAMES, dtype=float
        )
        self.df_nu.values[:] = 0.5

        sel = df["SEL"].values

        df_len_val_half = int(len(df[df["SEL"] == 1]) / 2)
        mask_1 = np.zeros_like(sel)
        mask_2 = np.zeros_like(sel)
        mask_1[np.where(sel == 1)[0][:df_len_val_half]] = 1
        mask_2[np.where(sel == 1)[0][df_len_val_half:]] = 1

        mask_1 = mask_1 == 1
        mask_2 = mask_2 == 1
        self._populate_nuisance_val(df=df, train_mask=mask_1, predict_mask=mask_2)
        self._populate_nuisance_val(df=df, train_mask=mask_2, predict_mask=mask_1)

        df_len_main_half = int(len(df[df["SEL"] == 0]) / 2)
        mask_1 = np.zeros_like(sel)
        mask_2 = np.zeros_like(sel)
        mask_1[np.where(sel == 0)[0][:df_len_main_half]] = 1
        mask_2[np.where(sel == 0)[0][df_len_main_half:]] = 1

        mask_1 = mask_1 == 1
        mask_2 = mask_2 == 1
        self._populate_nuisance_main(df=df, train_mask=mask_1, predict_mask=mask_2)
        self._populate_nuisance_main(df=df, train_mask=mask_2, predict_mask=mask_1)

    def _populate_nuisance_val(
        self, df: pd.DataFrame, train_mask: np.ndarray, predict_mask: np.ndarray
    ) -> None:
        df_train = df[train_mask]
        self.e_y_x_uw.train(
            X=df_train[["X"] + _COVARIATE_NAMES_ALL].values, y=df_train["Y"].values
        )
        self.e_x_uw.train(
            X=df_train[_COVARIATE_NAMES_ALL].values, y=df_train["X"].values
        )
        self.e_y_x_w_val.train(
            X=df_train[["X"] + _COVARIATE_NAMES_CRUDE].values,
            y=df_train["Y"].values,
        )
        self.e_x_w_val.train(
            X=df_train[_COVARIATE_NAMES_CRUDE].values,
            y=df_train["X"].values,
        )

        df_pred = df[predict_mask]
        df_nu = self.df_nu

        cov_arr_list = [df_pred[c].values for c in _COVARIATE_NAMES_CRUDE]
        df_nu["E[Y|X=0,W]"].values[predict_mask] = self.e_y_x_w_val.predict_proba(
            X=np.vstack([np.zeros_like(df_pred["X"])] + cov_arr_list).T
        )
        df_nu["E[Y|X=1,W]"].values[predict_mask] = self.e_y_x_w_val.predict_proba(
            X=np.vstack([np.ones_like(df_pred["X"])] + cov_arr_list).T
        )
        df_nu["P(X=1|W)"].values[predict_mask] = self.e_x_w_val.predict_proba(
            X=df_pred[_COVARIATE_NAMES_CRUDE].values
        )

        cov_arr_list = [df_pred[c].values for c in _COVARIATE_NAMES_ALL]
        df_nu["E[Y|X=0,U,W]"].values[predict_mask] = self.e_y_x_uw.predict_proba(
            X=np.vstack([np.zeros_like(df_pred["X"])] + cov_arr_list).T
        )
        df_nu["E[Y|X=1,U,W]"].values[predict_mask] = self.e_y_x_uw.predict_proba(
            X=np.vstack([np.ones_like(df_pred["X"])] + cov_arr_list).T
        )
        df_nu["P(X=1|U,W)"].values[predict_mask] = self.e_x_uw.predict_proba(
            X=df_pred[_COVARIATE_NAMES_ALL].values
        )

    def _populate_nuisance_main(
        self, df: pd.DataFrame, train_mask: np.ndarray, predict_mask: np.ndarray
    ) -> None:
        df_train = df[train_mask]

        self.e_y_x_w_main.train(
            X=df_train[["X"] + _COVARIATE_NAMES_CRUDE].values,
            y=df_train["Y"].values,
        )
        self.e_x_w_main.train(
            X=df_train[_COVARIATE_NAMES_CRUDE].values,
            y=df_train["X"].values,
        )

        df_pred = df[predict_mask]
        df_nu = self.df_nu

        cov_arr_list = [df_pred[c].values for c in _COVARIATE_NAMES_CRUDE]
        df_nu["E[Y|X=0,W]"].values[predict_mask] = self.e_y_x_w_main.predict_proba(
            X=np.vstack([np.zeros_like(df_pred["X"])] + cov_arr_list).T
        )
        df_nu["E[Y|X=1,W]"].values[predict_mask] = self.e_y_x_w_main.predict_proba(
            X=np.vstack([np.ones_like(df_pred["X"])] + cov_arr_list).T
        )
        df_nu["P(X=1|W)"].values[predict_mask] = self.e_x_w_main.predict_proba(
            X=df_pred[_COVARIATE_NAMES_CRUDE].values
        )


class LinearValidationCrossFitNuisanceEstimator(ValidationCrossFitNuisanceEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.e_y_x_uw = LogisticClassifier(penalty="l2")
        self.e_x_uw = LogisticClassifier(clip_low=0.05, clip_high=0.95)


class LinearCrudeConfoundersCrossFitNuisanceEstimator(
    CrudeConfoundersCrossFitNuisanceEstimator
):
    def __init__(self) -> None:
        super().__init__()
        self.e_y_x_w = LogisticClassifier(penalty="l2")
        self.e_x_w = LogisticClassifier(clip_low=0.05, clip_high=0.95)


class LinearCombinedDataCrossFitNuisanceEstimator(
    CombinedDataCrossFitNuisanceEstimator
):
    def __init__(self) -> None:
        super().__init__()
        self.e_x_uw = LogisticClassifier(clip_low=0.05, clip_high=0.95)
        self.e_y_x_uw = LogisticClassifier(penalty="l2")
        self.e_x_w_val = LogisticClassifier(clip_low=0.05, clip_high=0.95)
        self.e_y_x_w_val = LogisticClassifier(penalty="l2")
        self.e_x_w_main = LogisticClassifier(clip_low=0.05, clip_high=0.95)
        self.e_y_x_w_main = LogisticClassifier(penalty="l2")


class LearnedNuisanceEstimator(SequentialNuisanceEstimator):
    def __init__(
        self, horizon: int, batch_endpoints: list[int], initial_train_samples: int = 30
    ) -> None:
        super().__init__(horizon, batch_endpoints, initial_train_samples)

        self.e_y_x_uw: Classifier
        self.e_x_uw: Classifier

        self.e_y_x_w_val: Classifier
        self.e_x_w_val: Classifier

        self.e_y_x_w_main: Classifier
        self.e_x_w_main: Classifier

        self.COLUMN_NAMES = [
            "P(X=1|U,W)",
            "E[Y|X=1,U,W]",
            "E[Y|X=0,U,W]",
            "P(X=1|W)",
            "E[Y|X=1,W]",
            "E[Y|X=0,W]",
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
        df_train = df_train[df_train["SEL"] == 1]
        if len(df_train) <= self.source_1_last_train_size:
            return

        self.source_1_last_train_size = len(df_train)

        self.e_y_x_uw.train(
            X=df_train[["X"] + _COVARIATE_NAMES_ALL].values,
            y=df_train["Y"].values,
        )
        self.e_x_uw.train(
            X=df_train[_COVARIATE_NAMES_ALL].values,
            y=df_train["X"].values,
        )

        self.e_y_x_w_val.train(
            X=df_train[["X"] + _COVARIATE_NAMES_CRUDE].values,
            y=df_train["Y"].values,
        )
        self.e_x_w_val.train(
            X=df_train[_COVARIATE_NAMES_CRUDE].values,
            y=df_train["X"].values,
        )

    def _train_source_2_nuisance(self, df_train: pd.DataFrame) -> None:
        df_train = df_train[df_train["SEL"] == 0]
        if len(df_train) <= self.source_2_last_train_size:
            return

        self.source_2_last_train_size = len(df_train)

        self.e_y_x_w_main.train(
            X=df_train[["X"] + _COVARIATE_NAMES_CRUDE].values,
            y=df_train["Y"].values,
        )
        self.e_x_w_main.train(
            X=df_train[_COVARIATE_NAMES_CRUDE].values,
            y=df_train["X"].values,
        )

    def _train_nuisance_estimators(self, df: pd.DataFrame, train_endpoint: int) -> None:
        df_train = df[:train_endpoint]

        self._train_source_1_nuisance(df_train)
        self._train_source_2_nuisance(df_train)

    def _populate_nuisances(self, df: pd.DataFrame, start: int, end: int) -> None:
        df_nu = self.df_nu
        ps = start
        pe = end

        df_pred = df[ps:pe]
        sel = df_pred["SEL"]

        cov_arr_list = [df_pred[c].values for c in _COVARIATE_NAMES_CRUDE]
        df_nu["E[Y|X=0,W]"].values[ps:pe] = (
            sel
            * self.e_y_x_w_val.predict_proba(
                X=np.vstack([np.zeros_like(df_pred["X"])] + cov_arr_list).T
            )
        ) + (
            (1 - sel)
            * self.e_y_x_w_main.predict_proba(
                X=np.vstack([np.zeros_like(df_pred["X"])] + cov_arr_list).T
            )
        )
        df_nu["E[Y|X=1,W]"].values[ps:pe] = (
            sel
            * self.e_y_x_w_val.predict_proba(
                X=np.vstack([np.ones_like(df_pred["X"])] + cov_arr_list).T
            )
        ) + (
            (1 - sel)
            * self.e_y_x_w_main.predict_proba(
                X=np.vstack([np.ones_like(df_pred["X"])] + cov_arr_list).T
            )
        )
        df_nu["P(X=1|W)"].values[ps:pe] = (
            sel * self.e_x_w_val.predict_proba(X=df_pred[_COVARIATE_NAMES_CRUDE].values)
        ) + (
            (1 - sel)
            * self.e_x_w_main.predict_proba(X=df_pred[_COVARIATE_NAMES_CRUDE].values)
        )

        cov_arr_list = [df_pred[c].values for c in _COVARIATE_NAMES_ALL]
        df_nu["E[Y|X=0,U,W]"].values[ps:pe] = self.e_y_x_uw.predict_proba(
            X=np.vstack([np.zeros_like(df_pred["X"])] + cov_arr_list).T
        )
        df_nu["E[Y|X=1,U,W]"].values[ps:pe] = self.e_y_x_uw.predict_proba(
            X=np.vstack([np.ones_like(df_pred["X"])] + cov_arr_list).T
        )
        df_nu["P(X=1|U,W)"].values[ps:pe] = self.e_x_uw.predict_proba(
            X=df_pred[_COVARIATE_NAMES_ALL].values
        )


class LinearLearnedNuisanceEstimator(LearnedNuisanceEstimator):
    def __init__(
        self, horizon: int, batch_endpoints: list[int], initial_train_samples: int = 30
    ) -> None:
        super().__init__(horizon, batch_endpoints, initial_train_samples)

        self.e_x_uw = LogisticClassifier(clip_low=0.05, clip_high=0.95, penalty="l2")
        self.e_y_x_uw = LogisticClassifier(penalty="l2")
        self.e_x_w_val = LogisticClassifier(clip_low=0.05, clip_high=0.95, penalty="l2")
        self.e_y_x_w_val = LogisticClassifier(penalty="l2")
        self.e_x_w_main = LogisticClassifier(
            clip_low=0.05, clip_high=0.95, penalty="l2"
        )
        self.e_y_x_w_main = LogisticClassifier(penalty="l2")
