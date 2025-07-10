from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


def save_confusion_matrix(y_true, y_pred_classes, model_dir: Path):
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Matriz de confusión")
    plt.tight_layout()
    plt.savefig(model_dir / "confusion_matrix.png")
    plt.close()


def save_classification_report(y_true, y_pred_classes, model_dir: Path):
    report = classification_report(
        y_true, y_pred_classes, digits=4, output_dict=False
    )
    with open(model_dir / "classification_report.txt", "w") as f:
        f.write(
            """Matriz de reporte de clasificación (Classification Report):\n
            - Precision: de las veces que el modelo predijo una clase, 
            cuántas fueron correctas.\n
            - Recall: de las veces que una clase realmente apareció, 
            cuántas veces el modelo la detectó.\n
            - F1-Score: media armónica de precisión y recall.\n
            - Support: número real de muestras por clase.\n\n"""
        )
        f.write(str(report))
