from dataclasses import dataclass

import numpy as np

# Code taken from https://random-walks.org/book/papers/rff/rff.html#implementation.


@dataclass
class RFFCoefficients:
    omega: np.ndarray
    weights: np.ndarray
    phi: np.ndarray


def sample_rff_coefs(
    num_functions: int, num_features: int, x_dim: int = 1, kernel: str = "eq"
) -> RFFCoefficients:
    omega_shape = (num_functions, num_features, x_dim)
    # Handle each of three possible kernels separately
    if kernel == "eq":
        omega = np.random.normal(size=omega_shape)
    elif kernel == "laplace":
        omega = np.random.standard_cauchy(size=omega_shape)
    elif kernel == "cauchy":
        omega = np.random.laplace(size=omega_shape)
    else:
        raise ValueError(f"Kernel {kernel} not supported")

    weights = np.random.normal(loc=0.0, scale=1.0, size=(num_functions, num_features))
    phi = np.random.uniform(
        low=0.0, high=(2 * np.pi), size=(num_functions, num_features, 1)
    )
    return RFFCoefficients(omega=omega, weights=weights, phi=phi)


def compute_rff_fn(
    x: np.ndarray,
    rff_coefs: RFFCoefficients,
    lengthscale: float = 1,
    coefficient: float = 1,
) -> np.ndarray:
    # Dimension of data space
    x_dim = x.shape[-1]
    assert rff_coefs.omega.shape[-1] == x_dim

    omega, weights, phi = (rff_coefs.omega, rff_coefs.weights, rff_coefs.phi)

    # Scale omegas by lengthscale
    omega = omega / lengthscale

    num_features = rff_coefs.omega.shape[1]
    features = np.cos(np.einsum("sfd, nd -> sfn", omega, x) + phi)
    features = (2 / num_features) ** 0.5 * features * coefficient

    functions = np.einsum("sf, sfn -> sn", weights, features)

    return functions
