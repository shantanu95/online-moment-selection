from itertools import product

import pandas as pd

from causal_models.scm import SCM
from utils.data_utils import standardize_column


class CopdDataSCM(SCM):
    def __init__(self, data_filepath: str, standardize_covariates: bool = True) -> None:
        super().__init__()
        self.data_df: pd.DataFrame = pd.read_csv(data_filepath)
        self.is_validation_data = "PS_P2" in list(self.data_df.columns)

        self.data_df = self.data_df.rename(columns={"COPD": "X", "s1": "Y"})

        if standardize_covariates:
            prefixs = ["C", "P"] if self.is_validation_data else ["C"]
            for p, i in product(prefixs, range(1, 4)):
                col = f"PS_{p}{i}"
                self.data_df = standardize_column(
                    df=self.data_df, column=col, rename=False
                )

        if not self.is_validation_data:
            for i in range(1, 4):
                self.data_df[f"PS_P{i}"] = 0.5

        self.data_df = self.data_df[
            ["X", "Y"]
            + [f"PS_P{i}" for i in range(1, 4)]
            + [f"PS_C{i}" for i in range(1, 4)]
        ]

    def get_original_dataset(self) -> pd.DataFrame:
        return self.data_df.sample(frac=1, replace=False).reset_index(drop=True)

    def get_true_ate(self) -> float:
        # The true ATE is computed in `notebooks/copd_data_true_ATE.ipynb`.
        return 0.015609412706598623

    def generate_data_samples(self, num_samples: int) -> pd.DataFrame:
        return self.data_df.sample(n=num_samples, replace=True).reset_index(drop=True)
