import numpy as np
from scipy.optimize import LinearConstraint, minimize


def get_selection_cost(
    kappa: list[float], cost_per_source: list[float], num_samples: int = 1
) -> float:
    assert len(kappa) == len(cost_per_source)
    return sum(k * c for k, c in zip(kappa, cost_per_source)) * num_samples


def compute_reveal_k(
    current_samples: int,
    samples_to_reveal: int,
    current_kappa: float,
    target_kappa: float,
) -> float:
    reveal_k = (
        target_kappa * (current_samples + samples_to_reveal)
        - current_kappa * current_samples
    ) / (samples_to_reveal)
    return min(1, (max(0, reveal_k)))


def compute_reveal_k_with_budget(
    current_samples: int,
    budget_to_use: int,
    current_kappa: float,
    target_kappa: float,
    cost_per_source: list[float],
):
    if cost_per_source == [1.0, 1.0]:
        return compute_reveal_k(
            current_samples=current_samples,
            samples_to_reveal=budget_to_use,
            current_kappa=current_kappa,
            target_kappa=target_kappa,
        )

    def objective(k: float):
        samples_to_reveal = int(
            budget_to_use
            / get_selection_cost(kappa=[k, 1 - k], cost_per_source=cost_per_source)
        )
        k_final = (current_kappa * current_samples + k * samples_to_reveal) / (
            current_samples + samples_to_reveal
        )
        return (k_final - target_kappa) ** 2

    lower_bound = 0.01
    upper_bound = 0.99
    res = minimize(
        objective,
        current_kappa,
        constraints=[LinearConstraint(1, lb=lower_bound, ub=upper_bound)],
    )

    return res.x
