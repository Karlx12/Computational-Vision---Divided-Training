from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras.losses import (  # type: ignore
    SparseCategoricalCrossentropy,
)


class BaseModel(ABC):
    def __init__(self, input_shape, num_classes):
        self.model = self._build_model(input_shape, num_classes)

    @abstractmethod
    def _build_model(self, input_shape, num_classes): ...

    def compile(self, learning_rate=0.001):
        if self.model is None:
            raise ValueError("Model has not been built.")
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(  # type: ignore
                learning_rate=learning_rate
            ),
            loss=SparseCategoricalCrossentropy,
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),  # type: ignore
                tf.keras.metrics.Recall(name="recall"),  # type: ignore
                tf.keras.metrics.AUC(name="auc"),  # type: ignore
            ],
        )

    def get_model(self):
        return self.model
