from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils.selection_utils import get_selection_cost


class NuisanceEstimator(ABC):
    def __init__(self) -> None:
        self.last_nuisance_df_len: int = 0
        self.df_nu: pd.DataFrame = None

    def recompute_nuisances(self, df: pd.DataFrame) -> None:
        if self.last_nuisance_df_len >= len(df):
            return

        self.last_nuisance_df_len = len(df)
        self._recompute_nuisances(df=df)

    @abstractmethod
    def _recompute_nuisances(self, df: pd.DataFrame) -> None:
        pass

    @property
    @abstractmethod
    def nuisance_function_names(self) -> list[str]:
        pass

    def get(self, df: pd.DataFrame, function_name: str) -> np.ndarray:
        if function_name not in self.nuisance_function_names:
            raise ValueError("Invalid function name: %s" % function_name)

        self.recompute_nuisances(df)
        return self.df_nu[: len(df)][function_name].values


class SequentialNuisanceEstimator(NuisanceEstimator):
    def __init__(
        self, horizon: int, batch_endpoints: list[int], initial_train_samples: int = 30
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.batch_endpoints = sorted(batch_endpoints)
        self.last_train_endpoint: int = 0
        self.last_predict_endpoint: int = 0

        self.INITIAL_TRAIN_SAMPLES = initial_train_samples

        assert batch_endpoints[0] > self.INITIAL_TRAIN_SAMPLES
        assert batch_endpoints[-1] >= horizon

    @abstractmethod
    def _populate_nuisances(self, df: pd.DataFrame, start: int, end: int) -> None:
        pass

    @abstractmethod
    def _train_nuisance_estimators(self, df: pd.DataFrame, train_endpoint: int) -> None:
        pass

    def _recompute_nuisances(self, df: pd.DataFrame) -> None:
        if self.last_train_endpoint == 0:
            # For the first self.INITIAL_TRAIN_SAMPLES, use the same data from train
            # and test.
            assert len(df) >= self.INITIAL_TRAIN_SAMPLES
            self._train_nuisance_estimators(df, self.INITIAL_TRAIN_SAMPLES)
            self._populate_nuisances(df, 0, self.INITIAL_TRAIN_SAMPLES)

            self.last_train_endpoint = self.INITIAL_TRAIN_SAMPLES
            self.last_predict_endpoint = self.INITIAL_TRAIN_SAMPLES

        current_df_len = len(df)
        for batch_end in self.batch_endpoints:
            if batch_end >= current_df_len:
                self._populate_nuisances(df, self.last_predict_endpoint, current_df_len)
                self.last_predict_endpoint = current_df_len
                return

            if batch_end > self.last_predict_endpoint:
                self._populate_nuisances(df, self.last_predict_endpoint, batch_end)
                self.last_predict_endpoint = batch_end

            if batch_end > self.last_train_endpoint:
                # And now we re-train the estimator.
                self._train_nuisance_estimators(df, batch_end)
                self.last_train_endpoint = batch_end


class GMMEquations(ABC):
    @property
    @abstractmethod
    def num_moments(self) -> int:
        pass

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass

    @abstractmethod
    def get_moment_matrix(
        self, params: list[float], df: pd.DataFrame, nuisance: NuisanceEstimator
    ) -> np.ndarray:
        pass

    @abstractmethod
    def get_moment_reweighting(self, weights: list[float]) -> np.ndarray:
        pass

    @abstractmethod
    def get_jacobian(
        self, params: list[float], weights: list[float], sel: np.ndarray
    ) -> np.ndarray:
        pass


class GMMEstimator:
    def __init__(
        self, df: pd.DataFrame, gmm_eqs: GMMEquations, nuisance: NuisanceEstimator
    ):
        self.df = df
        self.gmm_eqs = gmm_eqs
        self.nuisance = nuisance
        self.num_samples = len(df)

    def _momconds(self, params: list[float]) -> np.ndarray:
        moments_matrix = self.gmm_eqs.get_moment_matrix(
            params=params, df=self.df, nuisance=self.nuisance
        )
        return np.mean(moments_matrix, axis=-1)

    def _compute_moment_covariance(self, params: list[float]) -> np.ndarray:
        """Estimate the optimal weight matrix using the emprical moments."""
        moments_matrix = self.gmm_eqs.get_moment_matrix(
            params=params, df=self.df, nuisance=self.nuisance
        )
        moment_covariance = (moments_matrix @ moments_matrix.T) / self.num_samples
        return moment_covariance

    def _get_objective_fn(
        self, weight_matrix_inv: np.ndarray
    ) -> Callable[[list[float]], float]:
        """Returns the GMM objective function."""

        def objective(params: list[float]):
            moms = self._momconds(params)
            w_inv_mom = np.linalg.solve(weight_matrix_inv, moms)
            obj = moms.T @ w_inv_mom

            return obj

        return objective

    def _optimize_find_parameters(
        self, weight_matrix_inv: np.ndarray, initial_guess: list[float]
    ) -> list[float]:
        res = minimize(
            self._get_objective_fn(weight_matrix_inv),
            initial_guess,
            bounds=[(-np.inf, np.inf)] * self.gmm_eqs.num_params,
        )
        return res.x

    def find_parameters(
        self, num_iters: int = 2, weight_matrix_reg: float = 0
    ) -> tuple[list[float], np.ndarray]:
        weight_matrix_inv = np.eye(self.gmm_eqs.num_moments)
        for i in range(num_iters):
            params = self._optimize_find_parameters(
                weight_matrix_inv,
                initial_guess=(
                    params if i > 0 else [0.0 for _ in range(self.gmm_eqs.num_params)]
                ),
            )
            weight_matrix_inv = self._compute_moment_covariance(params)
            weight_matrix_inv += weight_matrix_reg * np.eye(weight_matrix_inv.shape[0])

        return params, weight_matrix_inv

    def _get_asymptotic_variance_matrix(
        self, k: float, moment_covariance: np.ndarray, params: list[float]
    ) -> np.ndarray:
        current_k = np.mean(self.df["SEL"])
        weights = [k / current_k, (1 - k) / current_k]
        jacobian = self.gmm_eqs.get_jacobian(
            params=params, weights=weights, sel=self.df["SEL"].values
        )

        moment_cov_reweighted = moment_covariance * self.gmm_eqs.get_moment_reweighting(
            weights=weights
        )
        mom_cov_inv_jac = np.linalg.solve(moment_cov_reweighted, jacobian)
        try:
            return np.linalg.inv(jacobian.T @ mom_cov_inv_jac)
        except np.linalg.LinAlgError:
            return np.diag(np.full(shape=(self.gmm_eqs.num_params), fill_value=np.inf))

    def _get_asymptotic_variance(
        self, k: float, moment_covariance: np.ndarray, params: list[float]
    ) -> float:
        return self._get_asymptotic_variance_matrix(k, moment_covariance, params)[0, 0]

    def _get_asymptotic_variance_with_cost(
        self,
        k: float,
        moment_covariance: np.ndarray,
        params: list[float],
        cost_per_source: list[float],
    ) -> float:
        return self._get_asymptotic_variance_matrix(k, moment_covariance, params)[
            0, 0
        ] * get_selection_cost(kappa=[k, 1 - k], cost_per_source=cost_per_source)

    def find_optimal_k(
        self,
        moment_covariance: np.ndarray,
        params: list[float],
        cost_per_source: list[float] = [1.0, 1.0],
    ) -> float:
        initial_guess = [0.5]
        lower_bound = 0.05
        upper_bound = 0.99
        objective = lambda x: self._get_asymptotic_variance_with_cost(
            x[0], moment_covariance, params, cost_per_source=cost_per_source
        )
        # We use `lower_bound` and `upper_bound` to make the optimization stable since
        # the objective can go to infinity at the boundary.
        res = minimize(objective, initial_guess, bounds=[(lower_bound, upper_bound)])

        if res.x[0] == lower_bound:
            return 0

        if res.x[0] == upper_bound:
            return 1

        return res.x[0]
