import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_training_reports(
    model, history, val_gen, model_dir: Path, model_name: str
):
    """
    Guarda las gráficas de accuracy, loss y la matriz de confusión.
    """
    # Guardar gráfica de accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="Entrenamiento")
    plt.plot(history.history["val_accuracy"], label="Validación")
    plt.title("Precisión (accuracy)")
    plt.xlabel("Época")
    plt.ylabel("Precisión")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(model_dir / "accuracy.png")
    plt.close()

    # Guardar gráfica de loss
    plt.figure()
    plt.plot(history.history["loss"], label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validación")
    plt.title("Pérdida (loss)")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(model_dir / "loss.png")
    plt.close()

    # Matriz de confusión
    val_gen.reset()
    y_true = val_gen.classes
    y_pred = model.predict(val_gen, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Matriz de confusión")
    plt.tight_layout()
    plt.savefig(model_dir / "confusion_matrix.png")
    plt.close()
