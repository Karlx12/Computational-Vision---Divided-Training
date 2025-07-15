from typing import Protocol
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import AdamW
from keras.applications import EfficientNetB0


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


class EfficientNetB0FinetuneModel(BaseModel):
    def __init__(
        self,
        input_shape,
        num_classes,
        trainable_layers=30,
        freeze_batchnorm=True,
    ):
        self.trainable_layers = trainable_layers
        self.freeze_batchnorm = freeze_batchnorm
        super().__init__(input_shape, num_classes)

    def _build_model(self, input_shape, num_classes):
        base_model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
        # Congelar todas las capas
        for layer in base_model.layers:
            layer.trainable = False
        # Descongelar las últimas N capas
        if self.trainable_layers > 0:
            for layer in base_model.layers[-self.trainable_layers :]:
                layer.trainable = True
        # Opcional: mantener BatchNorm congeladas
        from keras.layers import (
            BatchNormalization,
            GlobalAveragePooling2D,
            Dense,
            Dropout,
        )

        if self.freeze_batchnorm:
            for layer in base_model.layers:
                if isinstance(layer, BatchNormalization):
                    layer.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation="softmax")(x)
        model = base_model.__class__.model(inputs=base_model.input, outputs=x)
        return model
