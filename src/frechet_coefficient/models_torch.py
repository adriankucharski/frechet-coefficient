import os

import torch.utils.data
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms, models
import torch
import numpy as np
import tqdm

from typing import List, Literal, Union


def not_implemented(name: str):
    raise NotImplementedError(f"{name} is not implemented")


MODELS_SPEC = {
    "inceptionv3": {
        "model": lambda: models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT),
        "layer": "avgpool",
        "preprocess": transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(size=(299, 299), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    },
    "convnexttiny": {
        "model": lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT),
        "layer": "avgpool",
        "preprocess": transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    },
    "efficientnetv2s": {
        "model": lambda: models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT),
        "layer": "avgpool",
        "preprocess": transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(size=(384, 384), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    },
    "densenet201": {
        "model": lambda: models.densenet201(weights=models.DenseNet201_Weights.DEFAULT),
        "layer": "adaptive_avg_pool2d",
        "preprocess": transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(size=(256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    },
    "resnet50v2": {
        "model": lambda: not_implemented("resnet50v2"),
    },
    "xception": {
        "model": lambda: not_implemented("xception"),
    },
}


class PretrainedModelWrapper:
    def __init__(
        self,
        model: Literal["inceptionv3", "convnexttiny", "efficientnetv2s", "densenet201", "resnet50v2", "xception"] = "inceptionv3",
    ) -> None:
        """
        Initializes the model object.

        Args:
            model (str, optional): The name of the model to use. Defaults to "inceptionv3".

        Raises:
            ValueError: If the specified model is not found.
        """
        if not isinstance(model, str) or model.lower() not in MODELS_SPEC:
            raise ValueError(f"Model {model} not found")

        # Set the device
        self.device = os.environ.get("FRECHET_COEFFICIENT_DEVICE_TORCH", "cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = model
        data = MODELS_SPEC[model.lower()]
        model: torch.nn.Module = data["model"]().to(self.device)
        self.layer: str = data["layer"]

        # Create the feature extractor model
        self.model: torch.nn.Module = create_feature_extractor(model, [self.layer])
        self.model.eval()

        # Set the preprocessing function
        self.preprocess = data["preprocess"]

    @property
    def feature_vector_size(self) -> int:
        """
        Returns the size of the feature vector.

        Returns:
            int: The size of the feature vector.
        """
        temp = torch.zeros(1, 3, 299, 299, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            feature = self.model(temp)
        return feature[self.layer].shape[1]

    def _preprocess(self, images: Union[List[np.ndarray], np.ndarray], verbose: int = 0) -> np.ndarray:
        """
        Preprocesses the input images.

        Args:
            images (List[np.ndarray] | np.ndarray): The input images to preprocess.
            verbose (int, optional): The verbosity level for the preprocessing. Defaults to 0.

        Returns:
            np.ndarray: The preprocessed images.
        """
        if isinstance(images, list):
            _min, _max = np.min([x.min() for x in images]), np.max([x.max() for x in images])
        else:
            _min, _max = images.min(), images.max()
        if _min < 0 or _max > 1:
            raise ValueError(f"Input data must be in range [0, 1] but got [{_min}, {_max}]")

        xn: List[np.ndarray] = []
        for i in tqdm.tqdm(range(len(images)), desc="Preprocessing images") if verbose else range(len(images)):
            image = (images[i] * 255).astype(np.uint8)
            xn.append(self.preprocess(image).numpy())

        return np.array(xn, dtype=np.float32)

    def predict(self, images: Union[List[np.ndarray], np.ndarray], batch_size: int = 4, verbose: int = 0) -> np.ndarray:
        """
        Predicts the feature vectors for the input images.

        Args:
            images (List[np.ndarray] | np.ndarray): The input images for which to predict the feature vectors.
            batch_size (int, optional): The batch size for the prediction. Defaults to 4.
            verbose (int, optional): The verbosity level for the prediction. Defaults to 0.

        Returns:
            np.ndarray: The feature vectors for the input images.
        """
        xn: np.ndarray = self._preprocess(images, verbose=verbose)
        with torch.no_grad():
            features: List[np.ndarray] = []
            range_fn = tqdm.tqdm(range(0, len(xn), batch_size)) if verbose else range(0, len(xn), batch_size)
            for i in range_fn:
                x = torch.from_numpy(xn[i : i + batch_size])
                x = x.to(self.device)
                y = self.model(x)
                features.append(y[self.layer].cpu().numpy())
            features = np.concatenate(features, axis=0).squeeze()

        return features
