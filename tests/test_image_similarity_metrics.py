import unittest
import numpy as np
from frechet_coefficient.metrics import ImageSimilarityMetrics


class TestImageSimilarityMetrics(unittest.TestCase):
    def setUp(self):
        self.model_name = "inceptionv3"
        self.verbose = 0
        self.ism = ImageSimilarityMetrics(self.model_name, self.verbose)

        # Create a random number generator
        rng = np.random.Generator(np.random.PCG64(12345))

        # Create dummy data for testing
        self.images1 = rng.uniform(size=(10, 299, 299, 3))  # 10 random images of size 299x299 with 3 channels
        self.images2 = rng.uniform(size=(10, 299, 299, 3))  # Another set of 10 random images

        # Create dummy data
        self.images3 = [rng.uniform(size=(rng.integers(50, 299), rng.integers(50, 299), 3)) for _ in range(10)]
        self.images4 = [rng.uniform(size=(rng.integers(50, 299), rng.integers(50, 299), 3)) for _ in range(10)]

    def test_derive_mean_cov_ndarray(self):
        mean, cov = self.ism.derive_mean_cov(self.images1)
        self.assertEqual(mean.shape, (self.ism.feature_vector_size,))
        self.assertEqual(
            cov.shape,
            (
                self.ism.feature_vector_size,
                self.ism.feature_vector_size,
            ),
        )

    def test_derive_mean_cov_list(self):
        mean, cov = self.ism.derive_mean_cov(self.images3)
        self.assertEqual(mean.shape, (self.ism.feature_vector_size,))
        self.assertEqual(
            cov.shape,
            (
                self.ism.feature_vector_size,
                self.ism.feature_vector_size,
            ),
        )

    def test_calculate_frechet_distance(self):
        fd = self.ism.calculate_frechet_distance(self.images1, self.images2)
        self.assertIsInstance(fd, float)

    def test_calculate_frechet_coefficient(self):
        fc = self.ism.calculate_frechet_coefficient(self.images1, self.images2)
        self.assertIsInstance(fc, float)

    def test_calculate_hellinger_distance(self):
        hd = self.ism.calculate_hellinger_distance(self.images1, self.images2)
        self.assertIsInstance(hd, float)


if __name__ == "__main__":
    unittest.main()
