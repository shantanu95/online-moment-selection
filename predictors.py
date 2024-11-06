from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor


class Regressor(ABC):
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class Classifier(ABC):
    def __init__(self, clip_low: float = 0.0, clip_high: float = 1.0):
        self.clip_low = clip_low
        self.clip_high = clip_high

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass


class LinearRegressor(Regressor):
    def __init__(self) -> None:
        super().__init__()
        self.linear_regression = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.linear_regression = LinearRegression(fit_intercept=True).fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.linear_regression is not None
        return self.linear_regression.predict(X)


class RidgeRegressor(Regressor):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.ridge_regression = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.ridge_regression = Ridge(alpha=self.alpha, fit_intercept=True).fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.ridge_regression is not None
        return self.ridge_regression.predict(X)


class ConstantClassifier(Classifier):
    def __init__(self, prob_value: float, clip_low: float = 0, clip_high: float = 1):
        super().__init__(clip_low, clip_high)
        self.prob_value = prob_value

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        return

    def predict_proba(self, X: np.ndarray) -> float:
        return self.prob_value


class MeanValueClassifier(Classifier):
    def __init__(self, clip_low: float = 0, clip_high: float = 1):
        super().__init__(clip_low, clip_high)
        self.mean_value = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean_value = np.mean(y)

    def predict_proba(self, X: Optional[np.ndarray] = None) -> float:
        assert self.mean_value is not None
        return self.mean_value


class ConstantRegressor(Regressor):
    def __init__(self, reg_value: float):
        super().__init__()
        self.reg_value = reg_value

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        return

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.reg_value


class LogisticClassifier(Classifier):
    def __init__(
        self, clip_low: float = 0, clip_high: float = 1, penalty: Optional[str] = None
    ):
        super().__init__(clip_low, clip_high)
        self.logistic_regression = None
        self.is_constant_classifier = None
        self.penalty = penalty

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        if np.unique(y).shape[0] < 2:
            # This means the target label does not have both ones and zeros. Sklearn
            # throws an error. So for now, just output a probability of 0.5.
            # This should happen very rarely.
            self.logistic_regression = ConstantClassifier(prob_value=0.5)
            self.is_constant_classifier = True
        else:
            self.logistic_regression = LogisticRegression(penalty=self.penalty).fit(
                X, y
            )
            self.is_constant_classifier = False

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.logistic_regression is not None
        if self.is_constant_classifier:
            return np.full(
                shape=(X.shape[0]), fill_value=self.logistic_regression.predict_proba(X)
            )

        return np.clip(
            self.logistic_regression.predict_proba(X)[:, 1],
            self.clip_low,
            self.clip_high,
        )


class MLPSklearnRegressor(Regressor):

    def __init__(self, hidden_layer_sizes: list[int]) -> None:
        super().__init__()
        self.mlp: MLPRegressor = None
        self.hidden_layer_sizes = hidden_layer_sizes

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.mlp is None:
            self.mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                early_stopping=True,
                validation_fraction=0.2,
                warm_start=True,
            )
        self.mlp.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.mlp is not None
        return self.mlp.predict(X)


class MLPSklearnClassifer(Classifier):

    def __init__(
        self,
        hidden_layer_sizes: list[int],
        clip_low: float = 0.0,
        clip_high: float = 1.0,
    ) -> None:
        super().__init__(clip_low, clip_high)
        self.mlp: MLPClassifier = None
        self.hidden_layer_sizes = hidden_layer_sizes

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.mlp is None:
            self.mlp = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                early_stopping=True,
                validation_fraction=0.2,
                warm_start=True,
            )

        try:
            self.mlp.fit(X, y)
        except ValueError:
            # this means that the train/val split did not have unique labels.
            pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.mlp is not None
        return np.clip(
            self.mlp.predict_proba(X)[:, 1],
            self.clip_low,
            self.clip_high,
        )
