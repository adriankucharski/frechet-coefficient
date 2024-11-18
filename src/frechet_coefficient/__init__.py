__version__ = "2.0.0"

try:
    import tensorflow
except ImportError:
    try:
        import torch
    except ImportError:
        raise ImportError(
            "Neither TensorFlow nor PyTorch is installed. Please install the package with `pip install frechet-coefficient[tensorflow]`, or `pip install frechet-coefficient[torch]`."
        )

# allow to import only the necessary functions
from .metrics import frechet_coefficient, hellinger_distance, frechet_distance, calculate_mean_cov, ImageSimilarityMetrics
from .utils import crop_random_patches, load_images
