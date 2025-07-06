from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()


def get_directories() -> dict[str, Path]:
    """Retorna los Path de los directorios principales del proyecto."""
    return {
        "DATASET_DIR": Path("dataset/Training"),
        "MODELS_DIR": Path("models"),
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
