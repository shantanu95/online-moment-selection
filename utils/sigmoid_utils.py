import numpy as np
from scipy.special import expit


def sigmoid_expectation_monte_carlo(
    mu: np.ndarray, noise_samples: np.ndarray
) -> np.ndarray:
    # shape: (mu, noise_samples)
    samples_with_noise = noise_samples + mu[:, None]
    return np.mean(expit(samples_with_noise), axis=-1)


def sigmoid_normal_expectation_monte_carlo(
    mu: np.ndarray, sigma: np.ndarray, num_samples: int = 1000
) -> np.ndarray:
    """
    Approximate E[sigmoid(x)] where x is Gaussian with N(mu, sigma^2)
    via Monte Carlo sampling of `x`.
    """
    assert len(mu) == len(sigma)

    samples_x = np.random.normal(loc=mu, scale=sigma, size=(num_samples, len(mu)))
    return np.mean(expit(samples_x), axis=0)
