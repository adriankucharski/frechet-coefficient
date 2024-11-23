import logging
from typing import List, Literal, Tuple, Union

import numpy as np

try:
    from .models_tensorflow import PretrainedModelWrapper
except ModuleNotFoundError:
    try:
        from .models_torch import PretrainedModelWrapper
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Could not import models from either TensorFlow or PyTorch")


def calculate_mean_cov(features: np.ndarray, dtype=np.float64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and covariance of the given features.

    Args:
        features (np.ndarray): The input features as a 2D array.
        dtype (type, optional): The data type of the output arrays. Defaults to np.float64.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the mean and covariance arrays.
    """
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
    """
    Calculates the Frechet distance between two multivariate Gaussian distributions.

    ## Note:
    To improve the efficiency of the function, it calculates sum(sqrt(eigenvalues)) instead of calculating tr((cov1 @ cov2)^0.5) directly.

    In other words: $\sum_{i=1}^{k} \sqrt{\lambda_i} = \text{tr}(A^{1/2})$

    Parameters:
        mean1 (np.ndarray): Mean vector of the first Gaussian distribution.
        cov1 (np.ndarray): Covariance matrix of the first Gaussian distribution.
        mean2 (np.ndarray): Mean vector of the second Gaussian distribution.
        cov2 (np.ndarray): Covariance matrix of the second Gaussian distribution.

    Returns:
        float: The Frechet distance between the two Gaussian distributions.

    Raises:
        ValueError: If the shapes of mean1, mean2, cov1, and cov2 do not match.
    """
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
    """
    Calculates the Frechet coefficient between two multivariate Gaussian distributions.
    To improve numerical stability pdeudo-inverse is used instead of the inverse of the covariance matrix.

    Parameters:
        mean1 (np.ndarray): Mean vector of the first Gaussian distribution.
        cov1 (np.ndarray): Covariance matrix of the first Gaussian distribution.
        mean2 (np.ndarray): Mean vector of the second Gaussian distribution.
        cov2 (np.ndarray): Covariance matrix of the second Gaussian distribution.

    Returns:
        float: The Frechet coefficient between the two Gaussian distributions.

    Raises:
        ValueError: If the shapes of mean1, mean2, cov1, and cov2 do not match.
    """
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
    """
    Calculates the Hellinger distance between two multivariate Gaussian distributions.

    ## Warning:
    The Hellinger distance is numerically unstable when the covariance matrices are singular or poorly estimated. In such cases, the function may return NaN.

    Parameters:
        mean1 (np.ndarray): Mean vector of the first Gaussian distribution.
        cov1 (np.ndarray): Covariance matrix of the first Gaussian distribution.
        mean2 (np.ndarray): Mean vector of the second Gaussian distribution.
        cov2 (np.ndarray): Covariance matrix of the second Gaussian distribution.

    Returns:
        float: The Hellinger distance between the two Gaussian distributions.

    Raises:
        ValueError: If the shapes of mean1, mean2, cov1, and cov2 do not match.
    """
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
        """
        Initializes an instance of the Metrics class.
        Args:
            model (Literal["inceptionv3", "resnet50v2", "xception", "densenet201", "convnexttiny", "efficientnetv2s"], optional):
                The name of the pre-trained model to use. Defaults to "inceptionv3".
            verbose (int, optional):
                Verbosity level. Defaults to 1.
        Returns:
            None
        """
        PretrainedModelWrapper.__init__(self, model)
        self.verbose = verbose

    def derive_mean_cov(self, images: Union[List[np.ndarray], np.ndarray], batch_size: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the mean and covariance of the given images.

        Args:
            images (List[np.ndarray] | np.ndarray): A list of images or a single image.
            batch_size (int, optional): The batch size for processing the images. Defaults to 4.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the mean and covariance of the features extracted from the images.
        """
        features = self.predict(images, batch_size=batch_size, verbose=self.verbose)
        return calculate_mean_cov(features)

    def calculate_frechet_distance(
        self, images_a: Union[List[np.ndarray], np.ndarray], images_b: Union[List[np.ndarray], np.ndarray], batch_size: int = 4
    ) -> float:
        """
        Calculates the Frechet distance between two sets of images.

        Args:
            images_a (List[np.ndarray] | np.ndarray): The first set of images.
            images_b (List[np.ndarray] | np.ndarray): The second set of images.
            batch_size (int, optional): The batch size for computing mean and covariance. Defaults to 4.

        Returns:
            float: The calculated Frechet distance.

        """
        mean1, cov1 = self.derive_mean_cov(images_a, batch_size)
        mean2, cov2 = self.derive_mean_cov(images_b, batch_size)
        fd = frechet_distance(mean1, cov1, mean2, cov2)
        return fd

    def calculate_frechet_coefficient(
        self, images_a: Union[List[np.ndarray], np.ndarray], images_b: Union[List[np.ndarray], np.ndarray], batch_size: int = 4
    ) -> float:
        """
        Calculates the Frechet coefficient between two sets of images.

        Args:
            images_a (List[np.ndarray] | np.ndarray): The first set of images.
            images_b (List[np.ndarray] | np.ndarray): The second set of images.
            batch_size (int, optional): The batch size for computing mean and covariance. Defaults to 4.

        Returns:
            float: The calculated Frechet coefficient.

        """
        mean1, cov1 = self.derive_mean_cov(images_a, batch_size)
        mean2, cov2 = self.derive_mean_cov(images_b, batch_size)
        fc = frechet_coefficient(mean1, cov1, mean2, cov2)
        return fc

    def calculate_hellinger_distance(
        self, images_a: Union[List[np.ndarray], np.ndarray], images_b: Union[List[np.ndarray], np.ndarray], batch_size: int = 4
    ) -> float:
        """
        Calculates the Frechet coefficient between two sets of images.

        ### Warning:
            The Hellinger distance is numerically unstable when the covariance matrices are singular or poorly estimated. In such cases, the function may return NaN.


        Args:
            images_a (List[np.ndarray] | np.ndarray): The first set of images.
            images_b (List[np.ndarray] | np.ndarray): The second set of images.
            batch_size (int, optional): The batch size for computing mean and covariance. Defaults to 4.

        Returns:
            float: The calculated Hellinger distance.
        """
        mean1, cov1 = self.derive_mean_cov(images_a, batch_size)
        mean2, cov2 = self.derive_mean_cov(images_b, batch_size)
        hd = hellinger_distance(mean1, cov1, mean2, cov2)
        return hd

    def calculate_fd_with_mean_cov(self, mean1: np.ndarray, cov1: np.ndarray, mean2: np.ndarray, cov2: np.ndarray) -> float:
        """
        Calculates the Fréchet Distance between two multivariate Gaussian distributions.

        Parameters:
            mean1 (np.ndarray): Mean vector of the first Gaussian distribution.
            cov1 (np.ndarray): Covariance matrix of the first Gaussian distribution.
            mean2 (np.ndarray): Mean vector of the second Gaussian distribution.
            cov2 (np.ndarray): Covariance matrix of the second Gaussian distribution.

        Returns:
            float: The Fréchet Distance between the two distributions.
        """
        return frechet_distance(mean1, cov1, mean2, cov2)

    def calculate_fc_with_mean_cov(self, mean1: np.ndarray, cov1: np.ndarray, mean2: np.ndarray, cov2: np.ndarray) -> float:
        """
        Calculates the Frechet coefficient using the given mean and covariance matrices.

        Parameters:
            mean1 (np.ndarray): The mean vector of the first distribution.
            cov1 (np.ndarray): The covariance matrix of the first distribution.
            mean2 (np.ndarray): The mean vector of the second distribution.
            cov2 (np.ndarray): The covariance matrix of the second distribution.

        Returns:
            float: The calculated Frechet coefficient.

        """
        return frechet_coefficient(mean1, cov1, mean2, cov2)

    def calculate_hd_with_mean_cov(self, mean1: np.ndarray, cov1: np.ndarray, mean2: np.ndarray, cov2: np.ndarray) -> float:
        """
        Calculates the Hellinger distance between two multivariate Gaussian distributions using their means and covariances.

        ### Warning:
            The Hellinger distance is numerically unstable when the covariance matrices are singular or poorly estimated. In such cases, the function may return NaN.

        Parameters:
            mean1 (np.ndarray): The mean of the first Gaussian distribution.
            cov1 (np.ndarray): The covariance matrix of the first Gaussian distribution.
            mean2 (np.ndarray): The mean of the second Gaussian distribution.
            cov2 (np.ndarray): The covariance matrix of the second Gaussian distribution.

        Returns:
            float: The Hellinger distance between the two Gaussian distributions.
        """
        return hellinger_distance(mean1, cov1, mean2, cov2)
