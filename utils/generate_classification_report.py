import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize
import json
import scipy.sparse

import visualkeras

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


def generate_classification_report(
    model_path: Path,
    data_dir: Path,
    img_size: Optional[int] = None,
    batch_size: int = 32,
    output_dir: "Optional[Path]" = None,
    _retry: bool = False,
):
    model_dir = model_path.parent
    try:
        # Cargar modelo
        model = load_model(model_path, safe_mode=True)
        if model is None:
            raise ValueError(f"No se pudo cargar el modelo desde {model_path}")
        # Detectar tamaño de imagen si no se especifica
        if img_size is None:
            img_size = get_model_image_size(model_path)
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
        if y_pred is None:
            raise ValueError("El modelo no devolvió predicciones.")
        y_pred = y_pred.astype(
            np.float32
        )  # Asegura dtype compatible con sklearn
        y_pred_classes = np.argmax(y_pred, axis=1)
        # Obtener nombres de clases
        class_names = list(val_gen.class_indices.keys())
        n_classes = len(class_names)
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

        # Visualización del modelo con visualkeras
        if _visualkeras_available:
            try:
                visualkeras.layered_view(
                    model, to_file=str(output_dir / "model_visualization.png")
                )
                print(
                    "🖼️ Visualización del modelo guardada en "
                    f"{output_dir / 'model_visualization.png'}"
                )
            except Exception as e:
                print(
                    "⚠️ No se pudo generar la visualización del modelo con "
                    "visualkeras: "
                    f"{e}"
                )
        else:
            print(
                "ℹ️ El paquete visualkeras no está instalado. "
                "Omite visualización de arquitectura."
            )
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred_classes)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=class_names
        )
        fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax_cm, cmap="Blues", xticks_rotation=45)
        plt.title("Matriz de Confusión")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png")
        plt.close(fig_cm)
        # ROC y Precision-Recall
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        # If y_true_bin is sparse, convert to dense
        if isinstance(y_true_bin, scipy.sparse.spmatrix):
            y_true_bin = y_true_bin.toarray()
        # ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Macro/micro ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_bin.ravel(), y_pred.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # Plot ROC
        fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
        for i in range(n_classes):
            ax_roc.plot(
                fpr[i],
                tpr[i],
                label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})",
            )
        ax_roc.plot([0, 1], [0, 1], "k--", lw=2)
        ax_roc.plot(
            fpr["micro"],
            tpr["micro"],
            label=f"micro-average (AUC = {roc_auc['micro']:.2f})",
            color="navy",
            linestyle=":",
            lw=2,
        )
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Curva ROC Multiclase")
        ax_roc.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png")
        plt.close(fig_roc)
        # Precision-Recall
        precision = dict()
        recall = dict()
        avg_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                y_true_bin[:, i], y_pred[:, i]
            )
            avg_precision[i] = average_precision_score(
                y_true_bin[:, i], y_pred[:, i]
            )
        # Macro/micro PR
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true_bin.ravel(), y_pred.ravel()
        )
        avg_precision["micro"] = average_precision_score(
            y_true_bin, y_pred, average="micro"
        )
        # Plot PR
        fig_pr, ax_pr = plt.subplots(figsize=(8, 8))
        for i in range(n_classes):
            ax_pr.plot(
                recall[i],
                precision[i],
                label=f"{class_names[i]} (AP = {avg_precision[i]:.2f})",
            )
        ax_pr.plot(
            recall["micro"],
            precision["micro"],
            label=f"micro-average (AP = {avg_precision['micro']:.2f})",
            color="navy",
            linestyle=":",
            lw=2,
        )
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Curva Precision-Recall Multiclase")
        ax_pr.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(output_dir / "precision_recall_curve.png")
        plt.close(fig_pr)
        print(
            f"✅ Reporte de clasificación y gráficos guardados en {output_dir}"
        )
    except ValueError as ve:
        msg = str(ve)
        if "expected shape" in msg and not _retry:
            # Extraer tamaño esperado del error
            import re

            m = re.search(r"expected shape=\(None, (\d+), (\d+), (\d+)\)", msg)
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
