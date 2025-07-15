import argparse
from pathlib import Path
import numpy as np
from typing import Optional
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import classification_report


def generate_classification_report(
    model_path: Path,
    data_dir: Path,
    img_size: int = 256,
    batch_size: int = 32,
    output_dir: "Optional[Path]" = None,
):
    # Cargar modelo
    model = load_model(model_path, safe_mode=True)
    if model is None:
        raise ValueError(f"No se pudo cargar el modelo desde {model_path}")
    # Preparar generador de datos
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_gen = datagen.flow_from_directory(
        str(data_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False,
    )
    # Predicciones
    y_true = val_gen.classes
    if y_true is None:
        raise ValueError(
            "No se encontraron etiquetas verdaderas en el generador."
        )
    y_pred = model.predict(val_gen, verbose=1)  # type: ignore
    print(type(model))
    if y_pred is None:
        raise ValueError("El modelo no devolvió predicciones.")
    y_pred_classes = np.argmax(y_pred, axis=1)
    # Obtener nombres de clases
    class_names = list(val_gen.class_indices.keys())
    # Reporte de clasificación con nombres de clase
    report = classification_report(
        y_true, y_pred_classes, digits=4, target_names=class_names
    )
    if isinstance(report, dict):
        report = str(report)
    # Guardar solo el reporte de clasificación
    if output_dir is None:
        output_dir = model_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(
            "Matriz de reporte de clasificación (Classification Report):\n"
            "- Precision: de las veces que el modelo predijo una clase, "
            "cuántas fueron correctas.\n"
            "- Recall: de las veces que una clase realmente apareció, "
            "cuántas veces el modelo la detectó.\n"
            "- F1-Score: media armónica de precisión y recall.\n"
            "- Support: número real de muestras por clase.\n\n"
        )
        f.write(report)
    print(f"✅ Reporte de clasificación guardado en {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Genera reporte de clasificación para un modelo Keras guardado."
        )
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        default="./models_trained/densenet201_final_20250708-152753/densenet201_final_20250708-152753.keras",
        help="Ruta al modelo .keras o .h5",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=False,
        default="./dataset/preprocessed",
        help=(
            "Directorio con las imágenes de validación/test "
            "(subcarpetas por clase)"
        ),
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Tamaño de imagen (default: 256)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directorio de salida (opcional)",
    )
    args = parser.parse_args()
    generate_classification_report(
        Path(args.model_path),
        Path(args.data_dir),
        args.img_size,
        args.batch_size,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
