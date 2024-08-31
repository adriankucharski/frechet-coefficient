import logging
from glob import glob
from typing import List, Tuple

import imageio.v3 as iio
import numpy as np


def rgb2gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        return image @ np.array([0.2125, 0.7154, 0.0721], dtype=image.dtype)
    return image


def crop_random_patches(
    images: List[np.ndarray],
    size: Tuple[int, int],
    num_of_patch: int,
    seed: int | None = None,
) -> np.ndarray:
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
    images = []
    for i, img in enumerate(glob(f"{path}/*")):
        if num_of_images is not None and i >= num_of_images:
            break
        img = iio.imread(img)
        if as_gray and img.ndim == 3:
            img = rgb2gray(img) * 255
        img = img / 255.0
        img = np.atleast_3d(img)
        images.append(img)
    return images
