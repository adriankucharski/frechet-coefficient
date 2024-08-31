# Frechet Coefficient

Frechet Coefficient is a Python package for calculating various similarity metrics between images, including Frechet Distance, Frechet Coefficient, and Hellinger Distance. It leverages pre-trained models from TensorFlow's Keras applications to extract features from images.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Citation](#citation)
- [License](#license)

## Installation

To install the package, use the following command:

```sh
pip install frechet-coefficient
```

Requirements:
- Python 3.11
- TensorFlow 2.16.1
- numpy 1.26.2
- imageio 2.33.1

## Usage

You can use the command-line interface (CLI) to calculate similarity metrics between two directories of images.

```sh
frechet-coefficient <path_to_directory1> <path_to_directory2> --metric <metric> [options]
```

Remember to use enough images to get a meaningful result. If your datasets are small, consider using `--random_patches` argument to calculate metrics on random patches of images.

### Positional Arguments
- `dir1`: Path to the first directory of images.
- `dir2`: Path to the second directory of images.

### Options

- `--metric`: Metric to calculate (fd, fc, hd).
- `--batch_size`: Batch size for processing images.
- `--num_of_images`: Number of images to load from each directory.
- `--as_gray`: Load images as grayscale.
- `--random_patches`: Calculate metrics on random patches of images.
- `--patch_size`: Size of the random patches.
- `--num_of_patch`: Number of random patches to extract.
- `--model`: Pre-trained model to use as feature extractor (inceptionv3, resnet50v2, xception, densenet201, convnexttiny, efficientnetv2s).
- `--verbose`: Enable verbose output.

### Example CLI Commands

To calculate the Frechet Distance between two sets of images, use the following command:
```sh
frechet-coefficient images/set1 images/set2 --metric fd
```

To calculate the Frechet Coefficient between two sets of images using the InceptionV3 model, use the following command:
```sh
frechet-coefficient images/set1 images/set2 --metric fc --model inceptionv3
```

To calculate the Hellinger Distance between two sets of images using random patches, use the following command:
```sh
frechet-coefficient images/set1 images/set2 --metric hd --random_patches --patch_size 128 --num_of_patch 10000
```
S
### Python Code

You can also use python code to calculate similarity metrics between two sets of images.

```python
import numpy as np
from typing import List
from frechet_coefficient import ImageSimilarityMetrics

# Initialize the ImageSimilarityMetrics class
ism = ImageSimilarityMetrics(model='inceptionv3', verbose=0)

images_1: List[np.ndarray] | np.ndarray = ... # shape: (num_of_images, height, width, channels)
images_2: List[np.ndarray] | np.ndarray = ... # shape: (num_of_images, height, width, channels)

# Calculate Frechet Distance
fd = ism.calculate_frechet_distance(images_1, images_2, batch_size=4)
# Calculate Frechet Coefficient
fc = ism.calculate_frechet_coefficient(images_1, images_2, batch_size=4)
# Calculate Hellinger Distance
hd = ism.calculate_hellinger_distance(images_1, images_2, batch_size=4)

# Calculate means vectors and covariance matrices
mean_1, cov_1 = ism.derive_mean_cov(images_1, batch_size=4)
mean_2, cov_2 = ism.derive_mean_cov(images_2, batch_size=4)

# Calculate metrics using mean vectors and covariance matrices
fd = ism.calculate_fd_with_mean_cov(mean_1, cov_1, mean_2, cov_2)
fc = ism.calculate_fc_with_mean_cov(mean_1, cov_1, mean_2, cov_2)
hd = ism.calculate_hd_with_mean_cov(mean_1, cov_1, mean_2, cov_2)

```

You can also calculate similarity metrics between two sets of images using random patches.

```python
import numpy as np
from typing import List
from frechet_coefficient import ImageSimilarityMetrics
from frechet_coefficient import crop_random_patches

# Initialize the ImageSimilarityMetrics class
ism = ImageSimilarityMetrics(model='inceptionv3', verbose=0)

images_1: List[np.ndarray] | np.ndarray = ... # shape: (num_of_images, height, width, channels)
images_2: List[np.ndarray] | np.ndarray = ... # shape: (num_of_images, height, width, channels)

# Crop random patches from images
images_1_patches = crop_random_patches(images_1, size=(128, 128), num_of_patch=10000)
images_2_patches = crop_random_patches(images_2, size=(128, 128), num_of_patch=10000)

# Calculate Frechet Distance
fd = ism.calculate_frechet_distance(images_1_patches, images_2_patches, batch_size=4)
# Calculate Frechet Coefficient
fc = ism.calculate_frechet_coefficient(images_1_patches, images_2_patches, batch_size=4)
# Calculate Hellinger Distance
hd = ism.calculate_hellinger_distance(images_1_patches, images_2_patches, batch_size=4)
```


### Metrics

- `fd`: Frechet Distance (with InceptionV3 model is equivalent to FID)
- `fc`: Frechet Coefficient
- `hd`: Hellinger Distance

The Hellinger Distance is numerically unstable for small datasets. The main reason is poorly estimated covariance matrices. To mitigate this issue, you can use the `--random_patches` argument to calculate metrics on random patches of images with a very high number of patches (e.g., 50000).

### Models

- `inceptionv3` - Input size: 299x299, Output size: 2048 - https://keras.io/api/applications/inceptionv3/
- `resnet50v2` - Input size: 224x224, Output size: 2048 - https://keras.io/api/applications/resnet/
- `xception` - Input size: 224x224, Output size: 2048 - https://keras.io/api/applications/xception/
- `densenet201` - Input size: 224x224, Output size: 1920 - https://keras.io/api/applications/densenet/
- `convnexttiny` - Input size: 224x224, Output size: 768 - https://keras.io/api/applications/convnext/
- `efficientnetv2s` - Input size: 384x384, Output size: 1280 - https://keras.io/api/applications/efficientnet/


## Features

- Calculate Frechet Distance, Frechet Coefficient, and Hellinger Distance between two sets of images.
- Support for multiple pre-trained models.
- Option to calculate metrics on random patches of images. 

## Citation

If you use this package in your research, please consider citing the following paper:

- not available yet

## License

This project is licensed under the MIT License. See the [`LICENSE`] file for details.
