from .base_model import BaseModel
from tensorflow.keras.layers import ( # type: ignore
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    SpatialDropout2D,
    GlobalAveragePooling2D,
    Dense,
)  # type: ignore
from tensorflow.keras import models  # type: ignore


class CNNModel(BaseModel):
    def _build_model(self, input_shape, num_classes):
        model = models.Sequential()

        # Bloque 1 - Capa de entrada
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(SpatialDropout2D(0.1))

        # Bloque 2
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(SpatialDropout2D(0.2))

        # Bloque 3
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(SpatialDropout2D(0.3))

        # Bloque 4
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(SpatialDropout2D(0.4))

        # Capas finales
        model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(num_classes, activation="softmax"))
        return model
