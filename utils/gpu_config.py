import os
import tensorflow as tf
from tensorflow.keras import mixed_precision  # pyright: ignore[reportMissingImports]


def configure_environment():
    """Configura el entorno de ejecución de TensorFlow y la GPU."""
    # Configuración de variables de entorno (solo aquí)
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    gpus = tf.config.list_physical_devices("GPU")
    if gpus and gpus[0].device_type == "GPU":
        policy = mixed_precision.Policy("mixed_bfloat16")
    else:
        policy = mixed_precision.Policy("float32")
    mixed_precision.set_global_policy(policy)

    # Optimizaciones específicas para arquitectura Ada
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.optimizer.set_experimental_options({
                "layout_optimizer": True,
                "constant_folding": True,
                "shape_optimization": True,
                "remapping": True,
                "arithmetic_optimization": True,
                "dependency_optimization": True,
                "loop_optimization": True,
                "function_optimization": True,
                "debug_stripper": True,
                "disable_model_pruning": False,
                "scoped_allocator_optimization": True,
                "pin_to_host_optimization": True,
                "implementation_selector": True,
                "auto_mixed_precision": True,
                "disable_meta_optimizer": False,
            })
            print("✅ Configuración de GPU completada con éxito.")
        except RuntimeError as e:
            print(f"⚠️ No se pudo configurar GPU: {e}")
