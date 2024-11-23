import unittest


class TestImports(unittest.TestCase):
    def test_tensorflow_fail_import(self):
        # This test will fail if tensorflow is imported
        with self.assertRaises(ModuleNotFoundError):
            import tensorflow

    def test_torch_fail_import(self):
        # This test will fail if torch is imported
        with self.assertRaises(ModuleNotFoundError):
            import torch

    def test_import_imageio(self):
        try:
            import imageio.v3
        except ModuleNotFoundError:
            self.fail("imageio not imported")
