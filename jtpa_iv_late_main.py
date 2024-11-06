from datetime import datetime
from typing import Optional

import numpy as np

from causal_models.scm import SCM
from gmm.jtpa import (
    JTPADataLATEGMMEqs,
    LinearCompleteDataCrossFitNuisanceEstimator,
    LinearSplitDataLearnedNuisanceEstimator,
    MLPCompleteDataCrossFitNuisanceEstimator,
    MLPSplitDataLearnedNuisanceEstimator,
)
from sample_revealer import SampleRevealer
from strategies import (
    ExploreThenCommitStrategy,
    ExploreThenGreedyStrategy,
    OracleStrategy,
)
from utils.print_utils import print_and_log


def execute_strategy_iteration(
    true_scm: SCM,
    strategy_name: str,
    iteration_num: int,
    horizon: int,
    logging_path: Optional[str] = None,
    store_nuisance_df: bool = False,
):
    if logging_path:
        print_and_log(
            "%s: Start %s, iter: %d, horizon %d"
            % (datetime.now(), strategy_name, iteration_num, horizon),
            filepath=f"{logging_path}_{strategy_name}_{horizon}",
        )

    random_seed = 232281293 + iteration_num
    np.random.seed(random_seed)
    num_samples = horizon
    df = true_scm.generate_data_samples(num_samples=num_samples)

    sample_revealer = SampleRevealer(budget=num_samples, df=df)
    nuisance_endpoints = np.arange(35, num_samples * 2, step=int(np.sqrt(num_samples)))

    if strategy_name == "complete_data_cross_fit":
        nuisance = LinearCompleteDataCrossFitNuisanceEstimator()
        strategy = OracleStrategy(
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=JTPADataLATEGMMEqs(use_complete_data=True),
            nuisance=nuisance,
            # this parameter is ignored since we use the complete data.
            optimal_kappa=0.5,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "complete_data_cross_fit_mlp":
        nuisance = MLPCompleteDataCrossFitNuisanceEstimator()
        strategy = OracleStrategy(
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=JTPADataLATEGMMEqs(use_complete_data=True),
            nuisance=nuisance,
            # this parameter is ignored since we use the complete data.
            optimal_kappa=0.5,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "fixed_equal":
        nuisance = LinearSplitDataLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = OracleStrategy(
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=JTPADataLATEGMMEqs(),
            nuisance=nuisance,
            optimal_kappa=0.5,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "fixed_equal_mlp":
        nuisance = MLPSplitDataLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = OracleStrategy(
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=JTPADataLATEGMMEqs(),
            nuisance=nuisance,
            optimal_kappa=0.5,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "oracle":
        nuisance = LinearSplitDataLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = OracleStrategy(
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=JTPADataLATEGMMEqs(),
            nuisance=nuisance,
            optimal_kappa=0.93,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "oracle_mlp":
        nuisance = MLPSplitDataLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = OracleStrategy(
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=JTPADataLATEGMMEqs(),
            nuisance=nuisance,
            optimal_kappa=0.91,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etc_0.1":
        nuisance = LinearSplitDataLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenCommitStrategy(
            exploration=0.1,
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=JTPADataLATEGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etc_0.1_mlp":
        nuisance = MLPSplitDataLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenCommitStrategy(
            exploration=0.1,
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=JTPADataLATEGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etc_0.2":
        nuisance = LinearSplitDataLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenCommitStrategy(
            exploration=0.2,
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=JTPADataLATEGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etc_0.2_mlp":
        nuisance = MLPSplitDataLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenCommitStrategy(
            exploration=0.2,
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=JTPADataLATEGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etc_0.4":
        nuisance = LinearSplitDataLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenCommitStrategy(
            exploration=0.4,
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=JTPADataLATEGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etc_0.4_mlp":
        nuisance = MLPSplitDataLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenCommitStrategy(
            exploration=0.4,
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=JTPADataLATEGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etg_0.1":
        nuisance = LinearSplitDataLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenGreedyStrategy(
            batch_fractions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9],
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=JTPADataLATEGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etg_0.1_mlp":
        nuisance = MLPSplitDataLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenGreedyStrategy(
            batch_fractions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9],
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=JTPADataLATEGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etg_0.2":
        nuisance = LinearSplitDataLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenGreedyStrategy(
            batch_fractions=[0.2, 0.4, 0.6, 0.8],
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=JTPADataLATEGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etg_0.2_mlp":
        nuisance = MLPSplitDataLearnedNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenGreedyStrategy(
            batch_fractions=[0.2, 0.4, 0.6, 0.8],
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=JTPADataLATEGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
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
