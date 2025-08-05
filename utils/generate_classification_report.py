import argparse
from pathlib import Path
import numpy as np
from typing import Optional
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import classification_report
import json

PREFS_FILENAME = "model_prefs.json"


def save_model_prefs(model_dir, img_size):
    prefs_path = model_dir / PREFS_FILENAME
    prefs = {"img_size": img_size}
    with open(prefs_path, "w") as f:
        json.dump(prefs, f)


def load_model_prefs(model_dir):
    prefs_path = model_dir / PREFS_FILENAME
    if prefs_path.exists():
        try:
            with open(prefs_path) as f:
                prefs = json.load(f)
                return prefs.get("img_size")
        except Exception:
            pass
    return None


def get_model_image_size(model_path, fallback_size=256):
    """
    Obtiene el tamaño de imagen preferido para el modelo:
    1. Busca en models_trained/<modelo>/image_size.txt
    2. Si no existe, intenta inferirlo del modelo y lo guarda
    3. Si falla, usa fallback_size
    """
    model_dir = model_path.parent
    # 1. JSON prefs
    img_size = load_model_prefs(model_dir)
    if img_size:
        return img_size
    # 2. TXT legacy
    pref_file = model_dir / "image_size.txt"
    if pref_file.exists():
        try:
            with open(pref_file) as f:
                size = int(f.read().strip())
                save_model_prefs(model_dir, size)
                return size
        except Exception:
            pass
    # 3. Inferir del modelo
    try:
        from keras.models import Model

        loaded_model = load_model(model_path, safe_mode=True)
        # Defensive: check for None and keras Model type
        if loaded_model is not None and isinstance(loaded_model, Model):
            input_shape = None
            # Try input_shape attribute
            if (
                hasattr(loaded_model, "input_shape")
                and loaded_model.input_shape is not None
            ):
                input_shape = loaded_model.input_shape
            # Try layers[0].input_shape for Sequential/Functional
            elif (
                hasattr(loaded_model, "layers")
                and isinstance(loaded_model.layers, (list, tuple))
                and len(loaded_model.layers) > 0
            ):
                first_layer = loaded_model.layers[0]
                if (
                    hasattr(first_layer, "input_shape")
                    and first_layer.input_shape is not None
                ):
                    input_shape = first_layer.input_shape
            if input_shape and len(input_shape) == 4:
                size = int(input_shape[1])
                save_model_prefs(model_dir, size)
                with open(pref_file, "w") as f:
                    f.write(str(size))
                return size
    except Exception:
        pass
    # 4. Fallback
    save_model_prefs(model_dir, fallback_size)
    with open(pref_file, "w") as f:
        f.write(str(fallback_size))
    return fallback_size


def load_and_prepare_model(model_path: Path):
    """Carga el modelo y obtiene el tamaño de imagen preferido."""
    model = load_model(model_path, safe_mode=True)
    if model is None:
        raise ValueError(f"No se pudo cargar el modelo desde {model_path}")
    img_size = get_model_image_size(model_path)
    return model, img_size


def prepare_data_generators(data_dir: Path, img_size: int, batch_size: int):
    """Prepara los generadores de datos para validación y entrenamiento."""
    validation_dir = data_dir / "validation"
    training_dir = data_dir / "training"

    if not validation_dir.exists() or not training_dir.exists():
        raise ValueError(
            "El directorio proporcionado debe contener subcarpetas "
            "'training' y 'validation'."
        )

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_gen = datagen.flow_from_directory(
        str(validation_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False,
    )
    train_gen = datagen.flow_from_directory(
        str(training_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False,
    )
    return val_gen, train_gen


def generate_classification_reports(model, val_gen, train_gen=None):
    """Genera los reportes de clasificación para validación y entrenamiento."""
    y_true = val_gen.classes
    y_pred = model.predict(val_gen, verbose=1).astype(np.float32)
    y_pred_classes = np.argmax(y_pred, axis=1)
    class_names_es = list(val_gen.class_indices.keys())
    class_translation = {
        "meningioma": "meningioma",
        "no_tumor": "no tumor",
        "pituitaria": "pituitary tumor",
    }
    class_names_en = [class_translation.get(c, c) for c in class_names_es]
    report_es = classification_report(
        y_true,
        y_pred_classes,
        digits=4,
        target_names=class_names_es,
        output_dict=True,
    )
    report_en = classification_report(
        y_true,
        y_pred_classes,
        digits=4,
        target_names=[str(c) for c in class_names_en],
        output_dict=True,
    )
    if train_gen:
        y_train_true = train_gen.classes
        y_train_pred = model.predict(train_gen, verbose=0).astype(np.float32)
        y_train_pred_classes = np.argmax(y_train_pred, axis=1)
        train_class_names = list(train_gen.class_indices.keys())
        train_metrics = classification_report(
            y_train_true,
            y_train_pred_classes,
            digits=4,
            target_names=[str(c) for c in train_class_names],
            output_dict=True,
        )
    else:
        train_metrics = None
    return report_es, report_en, train_metrics


def save_reports(output_dir: Path, report_es, report_en, train_metrics=None):
    """Guarda los reportes de clasificación en formato markdown y texto."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def dict_to_combined_markdown(report_dict, train_dict, class_names, title):
        lines = [f"### **{title}**\n"]
        lines.append(
            "| Classification  | Precision (Val) | Recall (Val) | "
            "F1-score (Val) | Support (Val) | "
        )
        lines.append(
            "|-----------------|-----------------|---------------|----------------|---------------|"
            "------------------|----------------|----------------|----------------|"
        )
        for cname in class_names:
            val_row = report_dict.get(cname, {})
            if not isinstance(val_row, dict):
                val_row = {
                    "precision": 0,
                    "recall": 0,
                    "f1-score": 0,
                    "support": 0,
                }
            train_row = train_dict.get(cname, {}) if train_dict else {}
            if not isinstance(train_row, dict):
                train_row = {
                    "precision": 0,
                    "recall": 0,
                    "f1-score": 0,
                    "support": 0,
                }
            lines.append(
                f"| {cname} | {val_row.get('precision', 0) * 100:.2f} | "
                f"{val_row.get('recall', 0) * 100:.2f} | "
                f"{val_row.get('f1-score', 0) * 100:.2f} | "
                f"{int(val_row.get('support', 0))} | "
                f"{train_row.get('precision', 0) * 100:.2f} | "
                f"{train_row.get('recall', 0) * 100:.2f} | "
                f"{train_row.get('f1-score', 0) * 100:.2f} | "
                f"{int(train_row.get('support', 0))} |"
            )
        return "\n".join(lines)

    with open(output_dir / "classification_report.md", "w") as f:
        f.write(
            dict_to_combined_markdown(
                report_en,
                train_metrics,
                list(report_en.keys()),
                "Combined Metrics",
            )
        )
    with open(output_dir / "classification_report_es.txt", "w") as f:
        f.write(str(report_es))
    with open(output_dir / "classification_report_en.txt", "w") as f:
        f.write(str(report_en))


def generate_classification_report(
    model_path: Path,
    data_dir: Path,
    img_size: Optional[int] = None,
    batch_size: int = 32,
    output_dir: "Optional[Path]" = None,
    _retry: bool = False,
):
    model_dir = model_path.parent
    # Set the output directory to a 'reports' subfolder
    # within the model's directory
    output_dir = model_dir / "reports"
    try:
        model, img_size = load_and_prepare_model(model_path)
        val_gen, train_gen = prepare_data_generators(
            data_dir,
            img_size if isinstance(img_size, int) else 256,
            batch_size,
        )
        report_es, report_en, train_metrics = generate_classification_reports(
            model,
            val_gen,
            train_gen,
        )
        save_reports(
            output_dir,
            report_es,
            report_en,
            train_metrics,
        )
    except ValueError as ve:
        msg = str(ve)
        if "expected shape" in msg and not _retry:
            # Extraer tamaño esperado del error
            import re

            m = re.search(
                r"expected shape=\\(None, (\\d+), (\\d+), (\\d+)\\)",
                msg,
            )
            if m:
                size = int(m.group(1))
                print(
                    f"⚠️ Tamaño de imagen incorrecto. Reintentando con {size}..."
                )
                save_model_prefs(model_dir, size)
                return generate_classification_report(
                    model_path,
                    data_dir,
                    img_size=size,
                    batch_size=batch_size,
                    output_dir=output_dir,
                    _retry=True,
                )
        print(f"❌ Error: {ve}")
        # Guardar error en prefs
        prefs_path = model_dir / PREFS_FILENAME
        try:
            with open(prefs_path, "r") as f:
                prefs = json.load(f)
        except Exception:
            prefs = {}
        prefs["last_error"] = msg
        with open(prefs_path, "w") as f:
            json.dump(prefs, f)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        # Guardar error en prefs
        prefs_path = model_dir / PREFS_FILENAME
        try:
            with open(prefs_path, "r") as f:
                prefs = json.load(f)
        except Exception:
            prefs = {}
        prefs["last_error"] = str(e)
        with open(prefs_path, "w") as f:
            json.dump(prefs, f)


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
        # default="./dataset/preprocessed_mri/",
        default="./dataset/real_training/",
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
