"""
Inteligencia de Negocios - Models Package

Este paquete contiene las arquitecturas de modelos disponibles para entrenamiento y fine-tuning:
- CNNModel
- VGGModel
- DenseNet201FinetuneModel

Asegúrate de que cada modelo herede de BaseModel y exponga la propiedad .model correctamente.
"""

from .base_model import BaseModel
from .cnn import CNNModel
from .vgg import VGGModel
from .densenet201 import DenseNet201FinetuneModel

__all__ = [
    "BaseModel",
    "CNNModel",
    "VGGModel",
    "DenseNet201FinetuneModel",
]
