import numpy as np
import pandas as pd

from utils.selection_utils import get_selection_cost


class SampleRevealer:
    def __init__(self, budget: int, df: pd.DataFrame):
        self.initial_budget = budget
        self.budget = budget
        self.buffer_size = len(df)
        self.df = df.copy()
        self.df["SEL"] = -1
        self.counter = 0

    def reset(self) -> None:
        self.counter = 0
        self.budget = self.initial_budget

    def reveal(self, reveal_kappa: float, samples_to_reveal: int) -> pd.DataFrame:
        assert (
            self.counter + samples_to_reveal <= self.buffer_size
        ), f"no buffer, counter={self.counter}, to_reveal={samples_to_reveal}, buffer={self.buffer_size}"

        selection_cost = get_selection_cost(
            kappa=[reveal_kappa, 1 - reveal_kappa],
            cost_per_source=self.cost_per_source,
            num_samples=samples_to_reveal,
        )
        assert self.budget >= selection_cost, "No budget left"
        self.budget -= selection_cost

        observe_count = int(reveal_kappa * samples_to_reveal)
        sel = np.zeros((samples_to_reveal))
        sel[:observe_count] = 1
        np.random.shuffle(sel)
        self.df["SEL"].values[self.counter : self.counter + samples_to_reveal] = sel
        self.counter += samples_to_reveal

        return self.get_dataset()

    def reveal_with_cost(
        self,
        reveal_kappa: float,
        budget_to_use: int,
        cost_per_source: list[float] = [1.0, 1.0],
    ) -> pd.DataFrame:
        assert self.budget >= budget_to_use, "No budget left"
        self.budget -= budget_to_use

        samples_to_reveal = int(
            budget_to_use
            / get_selection_cost(
                kappa=[reveal_kappa, 1 - reveal_kappa], cost_per_source=cost_per_source
            )
        )
        assert samples_to_reveal > 0

        observe_count = int(reveal_kappa * samples_to_reveal)
        sel = np.zeros((samples_to_reveal))
        sel[:observe_count] = 1
        np.random.shuffle(sel)
        self.df["SEL"].values[self.counter : self.counter + samples_to_reveal] = sel
        self.counter += samples_to_reveal

        return self.get_dataset()

    def get_dataset(self) -> pd.DataFrame:
        return self.df[: self.counter]


class SampleRevealerSelectionBias:
    def __init__(self, budget: int, df_obs: pd.DataFrame, df_bias: pd.DataFrame):
        assert set(df_obs.columns) == set(df_bias.columns)

        self.df_obs = df_obs
        self.df_bias = df_bias

        self.initial_budget = budget
        self.budget = budget
        self.df = pd.DataFrame(
            index=range(len(df_obs) + len(df_bias)),
            columns=list(df_obs.columns),
            dtype=float,
        )
        self.df["SEL"] = -1
        self.counter = 0

        self.obs_sample_counter = 0
        self.bias_sample_counter = 0

    def _get_observational_samples(self, num_samples: int) -> pd.DataFrame:
        assert self.obs_sample_counter + num_samples <= len(self.df_obs)

        self.obs_sample_counter += num_samples
        return self.df_obs[
            self.obs_sample_counter - num_samples : self.obs_sample_counter
        ]

    def _get_biased_samples(self, num_samples: int) -> pd.DataFrame:
        assert self.bias_sample_counter + num_samples <= len(self.df_bias)

        self.bias_sample_counter += num_samples
        return self.df_bias[
            self.bias_sample_counter - num_samples : self.bias_sample_counter
        ]

    def reveal_with_cost(
        self,
        reveal_kappa: float,
        budget_to_use: int,
        cost_per_source: list[float] = [1.0, 1.0],
    ) -> pd.DataFrame:
        assert self.budget >= budget_to_use, "No budget left"
        self.budget -= budget_to_use

        samples_to_reveal = int(
            budget_to_use
            / get_selection_cost(
                kappa=[reveal_kappa, 1 - reveal_kappa], cost_per_source=cost_per_source
            )
        )
        assert samples_to_reveal > 0

        observe_count = int(reveal_kappa * samples_to_reveal)
        df_obs = self._get_observational_samples(num_samples=observe_count)
        df_bias = self._get_biased_samples(
            num_samples=(samples_to_reveal - observe_count)
        )
        df_reveal = pd.concat([df_obs, df_bias]).reset_index(drop=True)

        sel = np.zeros((samples_to_reveal))
        sel[:observe_count] = 1
        df_reveal["SEL"] = sel
        df_reveal = df_reveal.sample(frac=1, replace=False)

        self.df[self.counter : self.counter + samples_to_reveal] = df_reveal.values
        self.counter += samples_to_reveal

        return self.get_dataset()

    def get_dataset(self) -> pd.DataFrame:
        return self.df[: self.counter]
