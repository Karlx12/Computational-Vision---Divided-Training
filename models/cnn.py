from .base_model import BaseModel
from keras.layers import (
    BatchNormalization,
    Activation,
    MaxPooling2D,
    SpatialDropout2D,
    GlobalAveragePooling2D,
    Dense,
    Conv2D,
)
from keras import models


class CNNModel(BaseModel):
    def _build_model(self, input_shape, num_classes):
        model = models.Sequential(
            [
                # Bloque 1 - Capa de entrada
                Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
                BatchNormalization(),
                Activation("relu"),
                MaxPooling2D((2, 2)),
                SpatialDropout2D(0.1),
                # Bloque 2
                Conv2D(64, (3, 3), padding="same"),
                BatchNormalization(),
                Activation("relu"),
                MaxPooling2D((2, 2)),
                SpatialDropout2D(0.2),
                # Bloque 3
                Conv2D(128, (3, 3), padding="same"),
                BatchNormalization(),
                Activation("relu"),
                MaxPooling2D((2, 2)),
                SpatialDropout2D(0.3),
                # Bloque 4
                Conv2D(256, (3, 3), padding="same"),
                BatchNormalization(),
                Activation("relu"),
                MaxPooling2D((2, 2)),
                SpatialDropout2D(0.4),
                # Capas finales
                Conv2D(512, (3, 3), padding="same", activation="relu"),
                GlobalAveragePooling2D(),
                Dense(128, activation="relu"),
                Dense(num_classes, activation="softmax"),
            ]
        )

        return model
