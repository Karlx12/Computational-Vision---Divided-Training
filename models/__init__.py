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
from .efficient_net import EfficientNetB0FinetuneModel
from .vgg16 import VGG16Model
from .vgg19 import VGG19Model

__all__ = [
    "BaseModel",
    "CNNModel",
    "VGGLikeModel",
    "DenseNet201FinetuneModel",
    "ResNet50FinetuneModel",
    "EfficientNetB0FinetuneModel",
    "VGG16Model",
    "VGG19Model",
]
