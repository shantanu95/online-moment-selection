from datetime import datetime
from typing import Optional

import numpy as np

from causal_models.scm import SCM
from gmm.copd_data import (
    LinearCombinedDataCrossFitNuisanceEstimator,
    LinearLearnedNuisanceEstimator,
)
from gmm.observational_two_covariates import (
    ObservationalTwoCovariatesGMMEqs,
    ObservationalTwoCovariatesSingleMomentGMMEqs,
)
from sample_revealer import SampleRevealerSelectionBias
from strategies import (
    ExploreThenCommitStrategy,
    ExploreThenGreedyStrategy,
    OracleStrategy,
    StrategyRunResult,
)
from utils.print_utils import print_and_log


def execute_strategy_iteration(
    true_scm_val: SCM,
    true_scm_main: SCM,
    strategy_name: str,
    iteration_num: int,
    horizon: int,
    cost_per_source: list[float],
    optimal_kappa: Optional[float] = None,
    logging_path: Optional[str] = None,
    store_nuisance_df: bool = False,
    # keep this because parallel_utils expects this arg.
    true_scm: Optional[SCM] = None,
) -> StrategyRunResult:
    if logging_path:
        print_and_log(
            "%s: Start %s, iter: %d, horizon %d"
            % (datetime.now(), strategy_name, iteration_num, horizon),
            filepath=f"{logging_path}_{strategy_name}_{horizon}",
        )

    random_seed = 232281293 + iteration_num
    np.random.seed(random_seed)
    num_samples = horizon

    df_val = true_scm_val.generate_data_samples(
        num_samples=int(num_samples / max(cost_per_source) + 1)
    )
    df_main = true_scm_main.generate_data_samples(num_samples=int(num_samples))

    sample_revealer = SampleRevealerSelectionBias(
        budget=num_samples, df_obs=df_val, df_bias=df_main
    )
    nuisance_endpoints = np.arange(35, num_samples * 2, step=15)

    if strategy_name == "oracle":
        assert optimal_kappa is not None, "`optimal_kappa` must be set."
        nuisance = LinearCombinedDataCrossFitNuisanceEstimator()
        strategy = OracleStrategy(
            sample_revealer=sample_revealer,
            true_scm=true_scm_val,
            gmm_eqs=ObservationalTwoCovariatesGMMEqs(),
            nuisance=nuisance,
            optimal_kappa=optimal_kappa,
            horizon=num_samples,
            cost_per_source=cost_per_source,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "single_source":
        nuisance = LinearCombinedDataCrossFitNuisanceEstimator()
        strategy = OracleStrategy(
            sample_revealer=sample_revealer,
            true_scm=true_scm_val,
            gmm_eqs=ObservationalTwoCovariatesSingleMomentGMMEqs(),
            nuisance=nuisance,
            optimal_kappa=1,
            horizon=num_samples,
            cost_per_source=cost_per_source,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etc_0.1":
        nuisance = LinearLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenCommitStrategy(
            exploration=0.1,
            sample_revealer=sample_revealer,
            true_scm=true_scm_val,
            gmm_eqs=ObservationalTwoCovariatesGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            cost_per_source=cost_per_source,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etc_0.2":
        nuisance = LinearLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenCommitStrategy(
            exploration=0.2,
            sample_revealer=sample_revealer,
            true_scm=true_scm_val,
            gmm_eqs=ObservationalTwoCovariatesGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            cost_per_source=cost_per_source,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etc_0.4":
        nuisance = LinearLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenCommitStrategy(
            exploration=0.4,
            sample_revealer=sample_revealer,
            true_scm=true_scm_val,
            gmm_eqs=ObservationalTwoCovariatesGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            cost_per_source=cost_per_source,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etg_0.1":
        nuisance = LinearLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenGreedyStrategy(
            batch_fractions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9],
            sample_revealer=sample_revealer,
            true_scm=true_scm_val,
            gmm_eqs=ObservationalTwoCovariatesGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            cost_per_source=cost_per_source,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etg_0.2":
        nuisance = LinearLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenGreedyStrategy(
            batch_fractions=[0.2, 0.4, 0.6, 0.8],
            sample_revealer=sample_revealer,
            true_scm=true_scm_val,
            gmm_eqs=ObservationalTwoCovariatesGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            cost_per_source=cost_per_source,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etg_0.4":
        nuisance = LinearLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenGreedyStrategy(
            batch_fractions=[0.4, 0.5, 0.6, 0.7, 0.8],
            sample_revealer=sample_revealer,
            true_scm=true_scm_val,
            gmm_eqs=ObservationalTwoCovariatesGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            cost_per_source=cost_per_source,
            store_nuisance_df=store_nuisance_df,
        )
    else:
        raise ValueError(f"invalid strategy_name: {strategy_name}")

    result = strategy.execute_run()
    np.random.seed(None)

    if logging_path:
        print_and_log(
            "%s: End %s, iter: %d, horizon %d"
            % (datetime.now(), strategy_name, iteration_num, horizon),
            filepath=f"{logging_path}_{strategy_name}_{horizon}",
        )

    return result
