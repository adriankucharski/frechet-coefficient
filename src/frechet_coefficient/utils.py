import os
import logging
from glob import glob
from typing import List, Tuple, Union

import imageio.v3 as iio
import numpy as np


def rgb2gray(image: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to grayscale.

    Parameters:
        image (np.ndarray): The input RGB image.

    Returns:
        np.ndarray: The grayscale image.

    Examples:
        >>> image = np.array(
        ...     [
        ...         [255, 0, 0],
        ...         [0, 255, 0],
        ...         [0, 0, 255],
        ...     ],
        ...     dtype=np.uint8,
        ... )
        >>> gray_image = rgb2gray(image)
        >>> print(gray_image)
        [[ 76]
         [150]
         [ 29]]
    """
    if image.ndim == 3 and image.shape[2] == 3:
        # CRT grayscale conversion
        return image @ np.array([0.2125, 0.7154, 0.0721], dtype=image.dtype)
    return image


def crop_random_patches(
    images: List[np.ndarray],
    size: Tuple[int, int],
    num_of_patch: int,
    seed: Union[int, None] = None,
) -> np.ndarray:
    """
    Crop random patches from a list of images.
    Args:
        images (List[np.ndarray]): A list of input images.
        size (Tuple[int, int]): The size of the patches to be cropped.
        num_of_patch (int): The number of patches to be cropped.
        seed (int | None, optional): The seed value for random number generation. Defaults to None.

    Returns:
        np.ndarray: An array containing the cropped patches.

    Raises:
        ValueError: If any image has an invalid shape.
    """
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        if not (img.ndim == 3 and h >= size[0] and w >= size[1]):
            logging.error(f"Image {i} has invalid shape {img.shape}")
            raise ValueError(f"Image {i} has invalid shape {img.shape}, must be at least {size}")

    nprng = np.random.Generator(np.random.PCG64(seed))
    new_images = np.empty((num_of_patch, *size, images[0].shape[-1]), dtype=images[0].dtype)

    num_of_images = len(images)
    for p in range(num_of_patch):
        i = p % num_of_images
        y = nprng.integers(0, h - size[0] + 1)
        x = nprng.integers(0, w - size[1] + 1)
        new_images[p] = images[i][y : y + size[0], x : x + size[1]]

    return new_images


def load_images(path: str, num_of_images: int = None, as_gray: bool = False) -> List[np.ndarray]:
    """
    Load images from the specified path.
    Args:
        path (str): The path to the directory containing the images.
        num_of_images (int, optional): The maximum number of images to load. Defaults to None.
        as_gray (bool, optional): Convert the images to grayscale. Defaults to False.
    Returns:
        List[np.ndarray]: A list of loaded images as numpy arrays.
    """
    if not os.path.exists(path):
        logging.error(f"Path {path} does not exist")
        raise FileNotFoundError(f"Path {path} does not exist")

    images: List[np.ndarray] = []
    for i, img in enumerate(glob(os.path.join(path, "*"))):
        if num_of_images is not None and i >= num_of_images:
            break
        img = iio.imread(img)
        if as_gray and img.ndim == 3:
            img = rgb2gray(img) * 255
        img = img / 255.0
        img = np.atleast_3d(img)
        images.append(img)
    return images
