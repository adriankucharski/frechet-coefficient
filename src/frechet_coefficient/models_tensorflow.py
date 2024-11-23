import os

# Disable TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import List, Literal, Union

import numpy as np
import tensorflow as tf
from keras import applications, models

MODELS_SPEC = {
    "inceptionv3": {
        "model": applications.InceptionV3,
        "input_shape": (299, 299, 3),
        "layer": "avg_pool",
        "preprocess": applications.inception_v3.preprocess_input,
    },
    "resnet50v2": {
        "model": applications.ResNet50V2,
        "input_shape": (224, 224, 3),
        "layer": "avg_pool",
        "preprocess": applications.resnet_v2.preprocess_input,
    },
    "xception": {
        "model": applications.Xception,
        "input_shape": (299, 299, 3),
        "layer": "avg_pool",
        "preprocess": applications.xception.preprocess_input,
    },
    "densenet201": {
        "model": applications.DenseNet201,
        "input_shape": (224, 224, 3),
        "layer": "avg_pool",
        "preprocess": applications.densenet.preprocess_input,
    },
    "convnexttiny": {
        "model": applications.ConvNeXtTiny,
        "input_shape": (224, 224, 3),
        "layer": "convnext_tiny_head_layernorm",
        "preprocess": applications.convnext.preprocess_input,
    },
    "efficientnetv2s": {
        "model": applications.EfficientNetV2S,
        "input_shape": (384, 384, 3),
        "layer": "avg_pool",
        "preprocess": applications.efficientnet.preprocess_input,
    },
}


class PretrainedModelWrapper:
    def __init__(
        self,
        model: Literal[
            "inceptionv3",
            "resnet50v2",
            "xception",
            "densenet201",
            "convnexttiny",
            "efficientnetv2s",
        ] = "inceptionv3",
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

        self.model_path = model
        data = MODELS_SPEC[model.lower()]
        model: models.Model = data["model"](include_top=True, input_shape=data["input_shape"])
        latent = model.get_layer(data["layer"])
        self.model = models.Model(inputs=model.input, outputs=latent.output)

    def _preprocess(self, images: Union[List[np.ndarray], np.ndarray], batch_size: int = 4) -> tf.data.Dataset:
        """
        Preprocesses the input images for the model.

        Args:
            images (List[np.ndarray] | np.ndarray): The input images to be preprocessed.
            batch_size (int, optional): The batch size for the dataset. Defaults to 4.

        Returns:
            tf.data.Dataset: The preprocessed dataset.

        Raises:
            ValueError: If the model path is not a string or not found in MODELS_SPEC.
            ValueError: If the input data is not in the range [0, 1].
        """
        if not isinstance(self.model_path, str) or self.model_path.lower() not in MODELS_SPEC:
            raise ValueError("Model not found")

        if isinstance(images, list):
            _min, _max = np.min([x.min() for x in images]), np.max([x.max() for x in images])
        else:
            _min, _max = images.min(), images.max()
        if _min < 0 or _max > 1:
            raise ValueError(f"Input data must be in range [0, 1] but got [{_min}, {_max}]")

        data = MODELS_SPEC[self.model_path.lower()]

        def resize(x: tf.Tensor) -> tf.Tensor:
            size = data["input_shape"][:2]
            if x.ndim == 3 and x.shape[:2] == size or x.ndim == 4 and x.shape[1:3] == size:
                return x
            return tf.image.resize(x, size, antialias=True)

        def apply(x: tf.Tensor) -> tf.Tensor:
            x = tf.clip_by_value(x * 255, 0, 255)
            x = tf.image.grayscale_to_rgb(x) if x.shape[-1] != 3 else x
            x = resize(x)
            x = data["preprocess"](x)
            return x

        if len(set([x.shape for x in images])) != 1:
            # if some images have different shapes resize them
            images = [resize(x) for x in images]

        xn = tf.data.Dataset.from_tensor_slices(images).batch(batch_size).map(apply)
        return xn

    @property
    def feature_vector_size(self) -> int:
        return self.model.output.shape[1]

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
        xn = self._preprocess(images)
        features = self.model.predict(xn, batch_size=batch_size, verbose=verbose)
        return features
