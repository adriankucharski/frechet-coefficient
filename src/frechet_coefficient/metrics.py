import logging
from typing import List, Literal, Tuple

import numpy as np

from frechet_coefficient.models import PretrainedModelWrapper


def calculate_mean_cov(features: np.ndarray, dtype=np.float64) -> Tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2:
        logging.error(f"Features must be 2D array, but got {features.ndim}D array")
        raise ValueError("Features must be 2D array")

    if features.shape[0] < features.shape[1]:
        logging.warning(
            f"Number of samples is less than number of features ({features.shape[0]} < {features.shape[1]}). Covariance matrix may be singular. Consider increasing the number of samples."
        )

    features = np.array(features, dtype=dtype)
    mean = np.mean(features, axis=0, dtype=dtype)
    cov = np.cov(features, rowvar=False, dtype=dtype)

    return mean, cov


def frechet_distance(mean1: np.ndarray, cov1: np.ndarray, mean2: np.ndarray, cov2: np.ndarray) -> float:
    if not (mean1.shape == mean2.shape and cov1.shape == cov2.shape):
        logging.error(f"Shape mismatch: mean1={mean1.shape}, mean2={mean2.shape}, cov1={cov1.shape}, cov2={cov2.shape}")
        raise ValueError("Shape mismatch")

    mean1, mean2 = np.array(mean1, dtype=np.float64), np.array(mean2, dtype=np.float64)
    cov1, cov2 = (
        np.array(cov1, dtype=np.complex128),
        np.array(cov2, dtype=np.complex128),
    )

    a = np.linalg.norm(mean1 - mean2) ** 2

    eig = np.linalg.eigvals(cov1 @ cov2)
    c = np.trace(cov1 + cov2) - 2 * np.sum(np.sqrt(eig))

    return a + np.real(c)


def frechet_coefficient(mean1: np.ndarray, cov1: np.ndarray, mean2: np.ndarray, cov2: np.ndarray) -> float:
    if not (mean1.shape == mean2.shape and cov1.shape == cov2.shape):
        logging.error(f"Shape mismatch: mean1={mean1.shape}, mean2={mean2.shape}, cov1={cov1.shape}, cov2={cov2.shape}")
        raise ValueError("Shape mismatch")

    mean1, mean2 = np.array(mean1, dtype=np.float64), np.array(mean2, dtype=np.float64)
    cov1, cov2 = (
        np.array(cov1, dtype=np.complex128),
        np.array(cov2, dtype=np.complex128),
    )

    k = mean1.size
    diff_of_mu = mean1 - mean2
    sum_of_sigma = cov1 + cov2
    d = (diff_of_mu @ np.linalg.pinv(sum_of_sigma / 2.0) @ diff_of_mu) / (2 * k)
    a = np.exp(-np.real(d))

    eig = np.linalg.eigvals(cov1 @ cov2)
    c = 2.0 * np.sum(np.sqrt(eig)) / np.trace(sum_of_sigma)

    return a * np.real(c)


def hellinger_distance(mean1: np.ndarray, cov1: np.ndarray, mean2: np.ndarray, cov2: np.ndarray) -> float:
    if not (mean1.shape == mean2.shape and cov1.shape == cov2.shape):
        logging.error(f"Shape mismatch: mean1={mean1.shape}, mean2={mean2.shape}, cov1={cov1.shape}, cov2={cov2.shape}")
        raise ValueError("Shape mismatch")

    mean1, mean2 = np.array(mean1, dtype=np.float64), np.array(mean2, dtype=np.float64)
    cov1, cov2 = (
        np.array(cov1, dtype=np.complex128),
        np.array(cov2, dtype=np.complex128),
    )

    sum_of_sigma = cov1 + cov2
    det1 = np.linalg.det(cov1)
    det2 = np.linalg.det(cov2)
    det3 = np.linalg.det(sum_of_sigma / 2)

    term1 = (det1**0.25 * det2**0.25) / (det3**0.5)
    term2 = (mean1 - mean2) @ np.linalg.pinv(sum_of_sigma / 2) @ (mean1 - mean2) / 0.125

    return 1 - np.real(term1) * np.real(np.exp(-term2))


class ImageSimilarityMetrics(PretrainedModelWrapper):
    def __init__(
        self,
        model: Literal[
            "inceptionv3",
            "resnet50v2",
            "xception",
            "densenet201",
            "convnexttiny",
            "efficientnetv2s",
        ] = "inceptionv3",
        verbose: int = 1,
    ):
        PretrainedModelWrapper.__init__(self, model)
        self.verbose = verbose

    def derive_mean_cov(self, images: List[np.ndarray] | np.ndarray, batch_size: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        images = self._preprocess(images, batch_size)
        features = self.model.predict(images, batch_size=batch_size, verbose=self.verbose)
        return calculate_mean_cov(features)

    def calculate_frechet_distance(
        self, images_a: List[np.ndarray] | np.ndarray, images_b: List[np.ndarray] | np.ndarray, batch_size: int = 4
    ) -> float:
        mean1, cov1 = self.derive_mean_cov(images_a, batch_size)
        mean2, cov2 = self.derive_mean_cov(images_b, batch_size)
        fd = frechet_distance(mean1, cov1, mean2, cov2)
        return fd

    def calculate_frechet_coefficient(
        self, images_a: List[np.ndarray] | np.ndarray, images_b: List[np.ndarray] | np.ndarray, batch_size: int = 4
    ) -> float:
        mean1, cov1 = self.derive_mean_cov(images_a, batch_size)
        mean2, cov2 = self.derive_mean_cov(images_b, batch_size)
        fc = frechet_coefficient(mean1, cov1, mean2, cov2)
        return fc

    def calculate_hellinger_distance(
        self, images_a: List[np.ndarray] | np.ndarray, images_b: List[np.ndarray] | np.ndarray, batch_size: int = 4
    ) -> float:
        mean1, cov1 = self.derive_mean_cov(images_a, batch_size)
        mean2, cov2 = self.derive_mean_cov(images_b, batch_size)
        hd = hellinger_distance(mean1, cov1, mean2, cov2)
        return hd

    def calculate_fd_with_mu_sigma(self, mean1: np.ndarray, cov1: np.ndarray, mean2: np.ndarray, cov2: np.ndarray) -> float:
        return frechet_distance(mean1, cov1, mean2, cov2)

    def calculate_fc_with_mu_sigma(self, mean1: np.ndarray, cov1: np.ndarray, mean2: np.ndarray, cov2: np.ndarray) -> float:
        return frechet_coefficient(mean1, cov1, mean2, cov2)

    def calculate_hd_with_mu_sigma(self, mean1: np.ndarray, cov1: np.ndarray, mean2: np.ndarray, cov2: np.ndarray) -> float:
        return hellinger_distance(mean1, cov1, mean2, cov2)
