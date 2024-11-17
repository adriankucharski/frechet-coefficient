__version__ = "1.0.4"
# allow to import only the necessary functions
from .metrics import frechet_coefficient, hellinger_distance, frechet_distance, calculate_mean_cov, ImageSimilarityMetrics
from .utils import crop_random_patches, load_images
