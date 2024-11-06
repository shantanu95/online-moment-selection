from datetime import datetime
from typing import Callable, Optional

import ipyparallel as ipp

from causal_models.scm import SCM
from strategies import StrategyRunResult
from utils.print_utils import print_and_log


def get_parallel_direct_view() -> ipp.DirectView:
    parallel_client = ipp.Client(debug=False)
    dview = parallel_client[:]

    # run something to make sure ipcluster is running.
    ar = dview.map(lambda x: x, (i for i in range(0, 10)))
    assert ar.get()[0] == 0

    return dview


def execute_strategy_in_parallel(
    true_scm: SCM,
    strategy_name: str,
    horizon: int,
    iterations: int,
    execute_fn: Callable,
    logging_path: Optional[str] = None,
):
    dview = get_parallel_direct_view()

    def execute_iteration(iteration_num: int):
        return execute_fn(
            true_scm=true_scm,
            strategy_name=strategy_name,
            iteration_num=iteration_num,
            horizon=horizon,
            logging_path=logging_path,
        )

    dview["logging_path"] = logging_path
    dview["execute_fn"] = execute_fn
    return dview.map(execute_iteration, range(iterations))


def combine_parallel_results(async_result) -> list[StrategyRunResult]:
    run_results: list[StrategyRunResult] = [
        res for res in async_result.get() if res is not None
    ]
    return run_results


def get_timeseries_for(
    true_scm, strategy_names, horizons, iterations, results_dict, execute_fn, log_path
):
    results_dict["true_scm"] = true_scm

    for strategy_name in strategy_names:
        for horizon in horizons:
            print_and_log(
                "Timestamp start: %s, Strategy: %s, Horizon: %d, Iters: %d"
                % (datetime.now(), strategy_name, horizon, iterations),
                log_path,
            )
            async_result = execute_strategy_in_parallel(
                true_scm=true_scm,
                strategy_name=strategy_name,
                horizon=horizon,
                iterations=iterations,
                execute_fn=execute_fn,
                logging_path=log_path,
            )
            run_results = combine_parallel_results(async_result)
            print_and_log("Timestamp end: %s" % (datetime.now()), log_path)

            if strategy_name not in results_dict:
                results_dict[strategy_name] = {}

            results_dict[strategy_name][horizon] = run_results
