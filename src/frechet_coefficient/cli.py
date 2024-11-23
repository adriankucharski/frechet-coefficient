"""
This module contains the CLI for the Frechet Coefficient. We can calculate the Frechet Coefficient between two directories of images using this CLI.
"""

import argparse
import os

from .metrics import ImageSimilarityMetrics
from .utils import crop_random_patches, load_images

parser = argparse.ArgumentParser(
    prog="Image Similarity Metrcis CLI", description="Frechet Coefficient", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("dir1", type=str, help="First directory containing images")
parser.add_argument("dir2", type=str, help="Second directory containing images")
parser.add_argument(
    "--model",
    type=str,
    default="inceptionv3",
    help="Pretrained model to use",
    choices=["inceptionv3", "resnet50v2", "xception", "densenet201", "convnexttiny", "efficientnetv2s"],
)
parser.add_argument(
    "--metric",
    type=str,
    default="fc",
    help="Metric to use, one of ['fd', 'fc', 'hd']. fd: Frechet Distance, fc: Frechet Coefficient, hd: Hellinger Distance",
    choices=["fd", "fc", "hd"],
)

parser.add_argument("--batch_size", type=int, default=4, help="Batch size for prediction")
parser.add_argument("--verbose", type=int, default=0, help="Verbosity level for prediction", choices=[0, 1, 2])
parser.add_argument("--num_of_images", type=int, default=None, help="Number of images to use from each directory. If None, all images will be used")
parser.add_argument("--as_gray", action="store_true", help="Convert images to grayscale")

parser.add_argument("--random_patches", action="store_true", help="Use random patches for calculation")
parser.add_argument("--patch_size", type=int, default=128, help="Size of the patch. Only used if random_patches is True")
parser.add_argument("--num_of_patch", type=int, default=2048, help="Number of patches to use. Only used if random_patches is True")


def main() -> None:
    args = parser.parse_args()

    print("Arguments: ", args)

    assert os.path.exists(args.dir1), f"Directory '{args.dir1}' not found"
    assert os.path.exists(args.dir2), f"Directory '{args.dir2}' not found"
    assert args.batch_size > 0, "Batch size must be greater than 0"
    assert args.num_of_images is None or args.num_of_images > 0, "Number of images must be greater than 0"
    if args.random_patches:
        assert args.patch_size > 0, "Patch size must be greater than 0"
        assert args.num_of_patch > 0, "Number of patches must be greater than 0"

    images1 = load_images(args.dir1, args.num_of_images, args.as_gray)
    images2 = load_images(args.dir2, args.num_of_images, args.as_gray)

    assert len(images1) > 1, "At least 2 images are required"
    assert len(images2) > 1, "At least 2 images are required"

    ism = ImageSimilarityMetrics(args.model, args.verbose)

    if args.random_patches:
        images1 = crop_random_patches(images1, (args.patch_size, args.patch_size), args.num_of_patch)
        images2 = crop_random_patches(images2, (args.patch_size, args.patch_size), args.num_of_patch)

    if args.metric == "fd":
        metric = ism.calculate_frechet_distance(images1, images2, args.batch_size)
    elif args.metric == "fc":
        metric = ism.calculate_frechet_coefficient(images1, images2, args.batch_size)
    elif args.metric == "hd":
        metric = ism.calculate_hellinger_distance(images1, images2, args.batch_size)
    else:
        raise ValueError(f"Invalid metric {args.metric}")

    print(f"Calculated {args.metric}: {metric}")


if __name__ == "__main__":
    main()
