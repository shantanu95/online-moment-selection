import pandas as pd

from causal_models.scm import SCM

# This true ATE was computed in "notebooks/jtpa_IV_true_LATE.ipynb".
MLP_JTPA_TRUE_LATE = 0.13658784058673604


class JTPADataSCM(SCM):
    def __init__(self, data_filepath: str) -> None:
        super().__init__()
        self.data_df: pd.DataFrame = pd.read_pickle(data_filepath)

    def get_max_size(self) -> int:
        return len(self.data_df)

    def get_true_ate(self) -> float:
        return MLP_JTPA_TRUE_LATE

    def generate_data_samples(self, num_samples: int) -> pd.DataFrame:
        assert num_samples <= self.get_max_size(), "Trying to get too many samples"
        return self.data_df.sample(n=num_samples, replace=False).reset_index(drop=True)


class JTPADataBootstrapSCM(SCM):
    def __init__(self, data_filepath: str) -> None:
        super().__init__()
        self.data_df: pd.DataFrame = pd.read_pickle(data_filepath)

    def get_original_size(self) -> int:
        return len(self.data_df)

    def get_true_ate(self) -> float:
        return MLP_JTPA_TRUE_LATE

    def generate_data_samples(self, num_samples: int) -> pd.DataFrame:
        return self.data_df.sample(n=num_samples, replace=True).reset_index(drop=True)
