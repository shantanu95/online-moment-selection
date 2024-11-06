from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import pandas as pd

from causal_models.scm import SCM
from gmm.gmm_equations import GMMEquations, GMMEstimator, NuisanceEstimator
from sample_revealer import SampleRevealer
from utils.selection_utils import compute_reveal_k_with_budget


class StrategyRunResult:
    def __init__(self):
        self.budgets_used: list[int] = []
        self.squared_errors: list[float] = []
        self.optimal_ks: list[float] = []
        self.current_ks: list[float] = []
        self.ate_hats: list[float] = []
        self.var_hats: list[float] = []
        self.nuisance_df: Optional[pd.DataFrame] = None

    def append(
        self,
        budget_used: int,
        squared_error: float,
        optimal_k: float,
        current_k: float,
        ate_hat: float,
        var_hat: float,
    ) -> None:
        self.budgets_used.append(budget_used)
        self.squared_errors.append(squared_error)
        self.optimal_ks.append(optimal_k)
        self.current_ks.append(current_k)
        self.ate_hats.append(ate_hat)
        self.var_hats.append(var_hat)

    def __str__(self) -> str:
        return f"Estimate: {self.ate_hats[-1]}, Optimal kappa: {self.optimal_ks[-1]}, Achieved kappa: {self.current_ks[-1]}"


class Strategy(ABC):
    def __init__(
        self,
        sample_revealer: SampleRevealer,
        true_scm: SCM,
        gmm_eqs: GMMEquations,
        nuisance: NuisanceEstimator,
        horizon: int,
        cost_per_source: list[float] = [1.0, 1.0],
        store_nuisance_df: bool = False,
    ):
        self.batch_size: list[int]
        self.latest_estimation_result: tuple[list[float], np.ndarray]

        self.latest_optimal_kappa: Optional[float] = None

        self.sample_revealer = sample_revealer
        # This is only used to compute the squared errors.
        self.true_scm = true_scm
        self.gmm_eqs = gmm_eqs
        self.nuisance = nuisance
        self.horizon = horizon
        self.cost_per_source = cost_per_source
        self.save_nuisance_df = store_nuisance_df

    @abstractmethod
    def get_reveal_kappa(self, batch_num: int, batch_size: int) -> float:
        pass

    def estimate_and_store_parameters(
        self, df: pd.DataFrame
    ) -> tuple[list[float], float]:
        gmm_estimator = GMMEstimator(
            df=df, gmm_eqs=self.gmm_eqs, nuisance=self.nuisance
        )
        self.latest_estimation_result = gmm_estimator.find_parameters()
        params, moment_covariance = self.latest_estimation_result
        current_k = np.mean(df["SEL"])
        var_hat = gmm_estimator._get_asymptotic_variance(
            k=current_k, moment_covariance=moment_covariance, params=params
        )
        return params, var_hat

    def get_squared_error(self, estimated_ate: float) -> float:
        return (self.true_scm.get_true_ate() - estimated_ate) ** 2

    def execute_run(self) -> StrategyRunResult:
        result = StrategyRunResult()

        for batch_num, batch_size in enumerate(self.batch_sizes):
            reveal_kappa = self.get_reveal_kappa(batch_num, batch_size)
            df = self.sample_revealer.reveal_with_cost(
                reveal_kappa=reveal_kappa,
                budget_to_use=batch_size,
                cost_per_source=self.cost_per_source,
            )
            params, var_hat = self.estimate_and_store_parameters(df)
            estimated_ate = params[0]
            current_k = np.mean(df["SEL"])

            result.append(
                budget_used=(
                    self.sample_revealer.initial_budget - self.sample_revealer.budget
                ),
                squared_error=self.get_squared_error(estimated_ate),
                optimal_k=self.latest_optimal_kappa,
                current_k=current_k,
                ate_hat=estimated_ate,
                var_hat=var_hat,
            )

        if self.save_nuisance_df:
            result.nuisance_df = self.nuisance.df_nu

        return result


class OracleStrategy(Strategy):
    def __init__(
        self,
        sample_revealer: SampleRevealer,
        true_scm: SCM,
        gmm_eqs: GMMEquations,
        nuisance: NuisanceEstimator,
        optimal_kappa: float,
        horizon: int,
        cost_per_source: list[float] = [1.0, 1.0],
        store_nuisance_df: bool = False,
    ):
        super().__init__(
            sample_revealer,
            true_scm,
            gmm_eqs,
            nuisance,
            horizon,
            cost_per_source,
            store_nuisance_df,
        )
        self.latest_optimal_kappa = optimal_kappa
        self.batch_sizes = [self.horizon]

    def get_reveal_kappa(self, batch_num: int, batch_size: int) -> float:
        assert batch_num == 0
        assert batch_size == self.horizon

        return self.latest_optimal_kappa


def _batch_fractions_to_sizes(horizon: int, batch_fractions: list[float]) -> list[int]:
    batch_sizes: list[int] = []

    for i in range(len(batch_fractions) + 1):
        prev_batch_end = 0 if i == 0 else int(horizon * batch_fractions[i - 1])
        curr_batch_end = (
            horizon if i == len(batch_fractions) else int(horizon * batch_fractions[i])
        )

        batch_sizes.append(curr_batch_end - prev_batch_end)

    return batch_sizes


class ExploreThenEpsilonGreedyStrategy(Strategy):
    def __init__(
        self,
        eplison_fn: Callable[[int], float],
        batch_fractions: list[float],
        sample_revealer: SampleRevealer,
        true_scm: SCM,
        gmm_eqs: GMMEquations,
        nuisance: NuisanceEstimator,
        horizon: int,
        cost_per_source: list[float] = [1.0, 1.0],
        store_nuisance_df: bool = False,
    ):
        super().__init__(
            sample_revealer,
            true_scm,
            gmm_eqs,
            nuisance,
            horizon,
            cost_per_source,
            store_nuisance_df,
        )
        self.epsilon_fn = eplison_fn
        # We assume that the first batch is exploration.
        self.batch_sizes = _batch_fractions_to_sizes(
            horizon=horizon, batch_fractions=batch_fractions
        )

    def get_reveal_kappa(self, batch_num: int, batch_size: int) -> float:
        # The first batch is assumed to be exploration.
        if batch_num == 0:
            return 0.5

        # w.p. epsilon, do exploration.
        if np.random.binomial(n=1, p=self.eplison_fn(batch_num)) == 1:
            return 0.5

        # first find the optimal_k using the latest estimated parameters.
        params, moment_covariance = self.latest_estimation_result
        df = self.sample_revealer.get_dataset()
        gmm_estimator = GMMEstimator(
            df=df, gmm_eqs=self.gmm_eqs, nuisance=self.nuisance
        )
        self.latest_optimal_kappa = gmm_estimator.find_optimal_k(
            params=params,
            moment_covariance=moment_covariance,
            cost_per_source=self.cost_per_source,
        )

        return compute_reveal_k_with_budget(
            current_samples=len(df),
            current_kappa=np.mean(df["SEL"]),
            budget_to_use=batch_size,
            target_kappa=self.latest_optimal_kappa,
            cost_per_source=self.cost_per_source,
        )


class ExploreThenGreedyStrategy(Strategy):
    def __init__(
        self,
        batch_fractions: list[float],
        sample_revealer: SampleRevealer,
        true_scm: SCM,
        gmm_eqs: GMMEquations,
        nuisance: NuisanceEstimator,
        horizon: int,
        cost_per_source: list[float] = [1.0, 1.0],
        store_nuisance_df: bool = False,
    ):
        super().__init__(
            sample_revealer,
            true_scm,
            gmm_eqs,
            nuisance,
            horizon,
            cost_per_source,
            store_nuisance_df,
        )
        # We assume that the first batch is exploration.
        self.batch_sizes = _batch_fractions_to_sizes(
            horizon=horizon, batch_fractions=batch_fractions
        )

    def get_reveal_kappa(self, batch_num: int, batch_size: int) -> float:
        # The first batch is assumed to be exploration.
        if batch_num == 0:
            return 0.5

        # first find the optimal_k using the latest estimated parameters.
        params, moment_covariance = self.latest_estimation_result
        df = self.sample_revealer.get_dataset()
        gmm_estimator = GMMEstimator(
            df=df, gmm_eqs=self.gmm_eqs, nuisance=self.nuisance
        )
        self.latest_optimal_kappa = gmm_estimator.find_optimal_k(
            params=params,
            moment_covariance=moment_covariance,
            cost_per_source=self.cost_per_source,
        )

        return compute_reveal_k_with_budget(
            current_samples=len(df),
            current_kappa=np.mean(df["SEL"]),
            budget_to_use=batch_size,
            target_kappa=self.latest_optimal_kappa,
            cost_per_source=self.cost_per_source,
        )


class ExploreThenCommitStrategy(ExploreThenGreedyStrategy):
    def __init__(
        self,
        exploration: float,
        sample_revealer: SampleRevealer,
        true_scm: SCM,
        gmm_eqs: GMMEquations,
        nuisance: NuisanceEstimator,
        horizon: int,
        cost_per_source: list[float] = [1.0, 1.0],
        store_nuisance_df: bool = False,
    ):
        super().__init__(
            batch_fractions=[exploration],
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=gmm_eqs,
            nuisance=nuisance,
            horizon=horizon,
            cost_per_source=cost_per_source,
            store_nuisance_df=store_nuisance_df,
        )


class EpsilonGreedyStrategy(Strategy):
    def __init__(
        self,
        eplison_fn: Callable[[int], float],
        sample_revealer: SampleRevealer,
        true_scm: SCM,
        gmm_eqs: GMMEquations,
        nuisance: NuisanceEstimator,
        horizon: int,
        cost_per_source: list[float] = [1.0, 1.0],
        store_nuisance_df: bool = False,
    ):
        super().__init__(
            sample_revealer,
            true_scm,
            gmm_eqs,
            nuisance,
            horizon,
            cost_per_source,
            store_nuisance_df,
        )
        self.eplison_fn = eplison_fn
        # We assume that the first batch is exploration.
        self.batch_sizes = list(range(1, horizon + 1))

    def get_reveal_kappa(self, batch_num: int, batch_size: int) -> float:
        # The first batch is assumed to be exploration.
        if batch_num == 0:
            return 0.5

        # w.p. epsilon, do exploration.
        if np.random.binomial(n=1, p=self.eplison_fn(batch_num)) == 1:
            return 0.5

        # first find the optimal_k using the latest estimated parameters.
        params, moment_covariance = self.latest_estimation_result
        df = self.sample_revealer.get_dataset()
        gmm_estimator = GMMEstimator(
            df=df, gmm_eqs=self.gmm_eqs, nuisance=self.nuisance
        )
        self.latest_optimal_kappa = gmm_estimator.find_optimal_k(
            params=params,
            moment_covariance=moment_covariance,
            cost_per_source=self.cost_per_source,
        )

        return compute_reveal_k_with_budget(
            current_samples=len(df),
            current_kappa=np.mean(df["SEL"]),
            budget_to_use=batch_size,
            target_kappa=self.latest_optimal_kappa,
            cost_per_source=self.cost_per_source,
        )
