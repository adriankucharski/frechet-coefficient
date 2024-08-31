```markdown
# Frechet Coefficient

Frechet Coefficient is a Python package for calculating various similarity metrics between images, including Frechet Distance, Frechet Coefficient, and Hellinger Distance. It leverages pre-trained models from TensorFlow's Keras applications to extract features from images.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [License](#license)

## Installation

To install the package, use the following command:

```sh
pip install frechet-coefficient
```

## Usage

You can use the command-line interface (CLI) to calculate similarity metrics between two directories of images.

```sh
frechet-coefficient --dir1 <path_to_directory1> --dir2 <path_to_directory2> --metric <metric> [options]
```

Remember to use enough images to get a meaningful result. If your datasets are small, consider using `--random_patches` argument to calculate metrics on random patches of images.

### Example

To calculate the Frechet Distance between two sets of images, use the following command:
```sh
frechet-coefficient --dir1 images/set1 --dir2 images/set2 --metric fd
```

To calculate the Frechet Coefficient between two sets of images using the InceptionV3 model, use the following command:
```sh
frechet-coefficient --dir1 images/set1 --dir2 images/set2 --metric fc --model inceptionv3
```

To calculate the Hellinger Distance between two sets of images using random patches, use the following command:
```sh
frechet-coefficient --dir1 images/set1 --dir2 images/set2 --metric hd --random_patches --patch_size 128 --num_of_patch 10000
```

### Options

- `--dir1`: Path to the first directory of images.
- `--dir2`: Path to the second directory of images.
- `--metric`: Metric to calculate (fd, fc, hd).
- `--batch_size`: Batch size for processing images.
- `--num_of_images`: Number of images to load from each directory.
- `--as_gray`: Load images as grayscale.
- `--random_patches`: Calculate metrics on random patches of images.
- `--patch_size`: Size of the random patches.
- `--num_of_patch`: Number of random patches to extract.
- `--model`: Pre-trained model to use as feature extractor (inceptionv3, resnet50v2, xception, densenet201, convnexttiny, efficientnetv2s).
- `--verbose`: Enable verbose output.

### Metrics

- `fd`: Frechet Distance (with InceptionV3 model is equivalent to FID)
- `fc`: Frechet Coefficient
- `hd`: Hellinger Distance

The Hellinger Distance is numerically unstable for small datasets. The main reason is poorly estimated covariance matrices. To mitigate this issue, you can use the `--random_patches` argument to calculate metrics on random patches of images with very high number of patches (e.g., 50000).

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

## License

This project is licensed under the MIT License. See the [`LICENSE`] file for details.
```