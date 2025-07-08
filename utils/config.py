from pathlib import Path
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde .env
load_dotenv()


# --- Configuración centralizada de variables de entorno ---
def get_tf_cpp_min_log_level():
    return os.getenv("TF_CPP_MIN_LOG_LEVEL", "2")


def get_img_size():
    return int(os.getenv("IMG_SIZE", "640"))


def get_batch_size():
    return int(os.getenv("BATCH_SIZE", 48))


def get_epochs():
    return int(os.getenv("EPOCHS", 100))


def get_model():
    return os.getenv("MODEL", "cnn")


def get_finetune_trainable_layers():
    return int(os.getenv("FINETUNE_TRAINABLE_LAYERS", 30))


def get_finetune_freeze_batchnorm():
    return os.getenv("FINETUNE_FREEZE_BATCHNORM", "true").lower() == "true"


# --- Directorios ---
def get_directories() -> dict[str, Path]:
    """Retorna los Path de los directorios principales del proyecto."""
    return {
        "DATASET_DIR": Path("dataset/preprocessed"),
        "MODELS_DIR": Path("models_trained"),
        "CHECKPOINTS_DIR": Path("training_checkpoints"),
        "LOG_DIR": Path("logs"),
        "SYNC_DIR": Path("sync"),
    }


def setup_directories() -> tuple[Path, ...]:
    """Crea los directorios principales si no existen."""
    dirs = get_directories()
    for key in ["MODELS_DIR", "CHECKPOINTS_DIR", "LOG_DIR", "SYNC_DIR"]:
        dirs[key].mkdir(exist_ok=True)
    return tuple(dirs.values())
