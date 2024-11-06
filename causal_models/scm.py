from abc import ABC, abstractmethod

import pandas as pd


class SCM(ABC):
    @abstractmethod
    def get_true_ate(self) -> float:
        pass

    @abstractmethod
    def generate_data_samples(self, num_samples: int) -> pd.DataFrame:
        pass
