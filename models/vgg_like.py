from keras import layers
from keras.models import Sequential
from .base_model import BaseModel


class VGGLikeModel(BaseModel):
    def _build_model(self, input_shape, num_classes) -> Sequential:
        model = Sequential(
            [
                layers.Input(shape=input_shape),
                # Bloque 1 - 2 capas convolucionales
                layers.Conv2D(
                    32,
                    (3, 3),
                    padding="same",
                    activation="relu",
                ),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Bloque 2 - 2 capas convolucionales
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Bloque 3 - 3 capas convolucionales
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.35),
                # Bloque 4 - 3 capas convolucionales
                layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.4),
                # Clasificador
                layers.Flatten(),
                layers.Dense(512, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        return model
