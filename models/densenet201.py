from tensorflow.keras.applications import DenseNet201  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model  # type: ignore

from .base_model import BaseModel


class DenseNet201FinetuneModel(BaseModel):
    def __init__(
        self,
        input_shape,
        num_classes,
        trainable_layers=30,
        freeze_batchnorm=True,
    ) -> None:
        self.trainable_layers = trainable_layers
        self.freeze_batchnorm = freeze_batchnorm
        super().__init__(input_shape, num_classes)

    def _build_model(self, input_shape, num_classes):
        base_model = DenseNet201(
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
        if self.freeze_batchnorm:
            for layer in base_model.layers:
                if isinstance(layer, BatchNormalization):
                    layer.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=x)
        return model
