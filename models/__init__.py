"""
Inteligencia de Negocios - Models Package

Este paquete contiene las arquitecturas de modelos disponibles para
entrenamiento y fine-tuning:
- CNNModel
- VGGModel
- DenseNet201FinetuneModel

Asegúrate de que cada modelo herede de BaseModel y
exponga la propiedad .model correctamente.
"""

from .base_model import BaseModel
from .cnn import CNNModel
from .vgg_like import VGGLikeModel
from .densenet201 import DenseNet201FinetuneModel
from .resnet import ResNet50FinetuneModel
from .efficientnetb0 import EfficientNetB0FinetuneModel
from .vgg16 import VGG16Model
from .vgg19 import VGG19Model
from .densenet121 import DenseNet121FinetuneModel
from .efficientnetb1 import EfficientNetB1FinetuneModel
from .inceptionv3 import InceptionV3FinetuneModel
from .nasnet import NASNetMobileFinetuneModel, NASNetLargeFinetuneModel

__all__ = [
    "BaseModel",
    "CNNModel",
    "VGGLikeModel",
    "DenseNet201FinetuneModel",
    "ResNet50FinetuneModel",
    "EfficientNetB0FinetuneModel",
    "VGG16Model",
    "VGG19Model",
    "DenseNet121FinetuneModel",
    "EfficientNetB1FinetuneModel",
    "InceptionV3FinetuneModel",
    "NASNetMobileFinetuneModel",
    "NASNetLargeFinetuneModel",
]
