from datetime import datetime
from typing import Optional

import numpy as np
from utils.print_utils import print_and_log

from causal_models.scm import SCM
from gmm.iv_LATE import (
    LATETwoSampleGMMEqs,
    LogisticNuisanceEstimator,
    OracleNuisanceEstimator,
)
from sample_revealer import SampleRevealer
from strategies import (
    ExploreThenCommitStrategy,
    ExploreThenEpsilonGreedyStrategy,
    ExploreThenGreedyStrategy,
    OracleStrategy,
)


def execute_strategy_iteration(
    true_scm: SCM,
    strategy_name: str,
    iteration_num: int,
    horizon: int,
    logging_path: Optional[str] = None,
    store_nuisance_df: bool = False,
    oracle_kappa: Optional[float] = None,
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
    nuisance_endpoints = np.arange(40, num_samples * 2, step=int(np.sqrt(num_samples)))

    if "oracle" in strategy_name:
        assert oracle_kappa is not None

    if strategy_name == "oracle_with_true_nu":
        nuisance = OracleNuisanceEstimator()
        strategy = OracleStrategy(
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=LATETwoSampleGMMEqs(),
            nuisance=nuisance,
            optimal_kappa=oracle_kappa,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "oracle_with_est_nu":
        nuisance = LogisticNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = OracleStrategy(
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=LATETwoSampleGMMEqs(),
            nuisance=nuisance,
            optimal_kappa=oracle_kappa,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "fixed_equal":
        nuisance = LogisticNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = OracleStrategy(
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=LATETwoSampleGMMEqs(),
            nuisance=nuisance,
            optimal_kappa=0.5,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etc_0.1":
        nuisance = LogisticNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenCommitStrategy(
            exploration=0.1,
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=LATETwoSampleGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etc_0.2":
        nuisance = LogisticNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenCommitStrategy(
            exploration=0.2,
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=LATETwoSampleGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etc_0.4":
        nuisance = LogisticNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenCommitStrategy(
            exploration=0.4,
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=LATETwoSampleGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etg_0.1":
        nuisance = LogisticNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenGreedyStrategy(
            batch_fractions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9],
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=LATETwoSampleGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etg_0.2":
        nuisance = LogisticNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenGreedyStrategy(
            batch_fractions=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=LATETwoSampleGMMEqs(),
            nuisance=nuisance,
            horizon=num_samples,
            store_nuisance_df=store_nuisance_df,
        )
    elif strategy_name == "etg_0.4":
        nuisance = LogisticNuisanceEstimator(
            horizon=num_samples, batch_endpoints=nuisance_endpoints
        )
        strategy = ExploreThenGreedyStrategy(
            batch_fractions=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            sample_revealer=sample_revealer,
            true_scm=true_scm,
            gmm_eqs=LATETwoSampleGMMEqs(),
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
