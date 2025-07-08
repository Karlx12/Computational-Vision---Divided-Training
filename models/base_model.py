from typing import Protocol
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import AdamW


class ModelProtocol(Protocol):
    def compile(self, learning_rate=0.001): ...
    def get_model(self): ...


class BaseModel:
    def __init__(self, input_shape, num_classes, learning_rate=0.001):
        self.model = self._build_model(input_shape, num_classes)
        self.compile(learning_rate)

    def _build_model(self, input_shape, num_classes):
        raise NotImplementedError(
            "Debes implementar _build_model en la subclase."
        )

    def compile(self, learning_rate=0.001):
        if self.model is None:
            raise ValueError("Model has not been built.")
        self.model.compile(
            optimizer=AdamW(learning_rate=learning_rate),
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def get_model(self):
        return self.model
