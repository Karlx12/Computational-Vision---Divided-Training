from argparse import Namespace
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Solo errores fatales
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.getenv("TF_CPP_MIN_LOG_LEVEL", "3")
import keras
from datetime import datetime
from utils.config import get_directories, setup_directories
from utils.data_loader import create_data_generators
from utils.model_reports import save_training_reports


def setup_training_components(args: Namespace, input_shape, num_classes):
    dirs: dict[str, Path] = get_directories()
    train_gen, val_gen = create_data_generators(
        dirs["DATASET_DIR"], input_shape, args.batch_size
    )
    model = None
    # Selección de modelo
    if args.model == "cnn":
        from models.cnn import CNNModel

        model = CNNModel(input_shape, num_classes).model
    elif args.model == "vgg":
        from models.vgg import VGGModel

        model = VGGModel(input_shape, num_classes).model
    elif args.model == "densenet201":
        from models.densenet201 import DenseNet201FinetuneModel

        trainable_layers = int(os.getenv("FINETUNE_TRAINABLE_LAYERS", 30))
        freeze_batchnorm = (
            os.getenv("FINETUNE_FREEZE_BATCHNORM", "true").lower() == "true"
        )
        model = DenseNet201FinetuneModel(
            input_shape, num_classes, trainable_layers, freeze_batchnorm
        ).model
    else:
        raise ValueError(f"Modelo no soportado: {args.model}")

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            monitor="val_accuracy",
            mode="max",
            verbose=1,
            filepath=str(
                dirs["CHECKPOINTS_DIR"] / f"{args.model}_weights.weights.h5"
            ),
            save_weights_only=True,
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.CSVLogger(
            str(dirs["LOG_DIR"] / f"{args.model}_training_log.csv"),
            append=True,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(
                dirs["LOG_DIR"]
                / f"{args.model}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.1,
            patience=3,
            min_lr=1e-6,
        ),
    ]
    return model, train_gen, val_gen, callbacks


def main():
    from argparse import ArgumentParser
    from utils.gpu_config import configure_environment

    configure_environment()
    setup_directories()
    dirs = get_directories()
    input_shape = (256, 256, 3)
    num_classes: int = len(
        [d for d in dirs["DATASET_DIR"].iterdir() if d.is_dir()]
    )

    parser = ArgumentParser()
    parser.add_argument(
        "--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", 48))
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "vgg", "densenet201"],
        default=os.getenv("MODEL", "cnn"),
    )
    parser.add_argument(
        "--epochs", type=int, default=int(os.getenv("EPOCHS", 100))
    )
    args = parser.parse_args()

    model, train_gen, val_gen, callbacks = setup_training_components(
        args, input_shape, num_classes
    )
    if model:
        history = model.fit(
            train_gen,
            epochs=args.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
        )
        date = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir = dirs["MODELS_DIR"] / f"{args.model}_final_{date}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{args.model}_final_{date}.keras"
        model.save(model_path)
        print(f"✅ Modelo guardado en {model_path}")
        save_training_reports(model, history, val_gen, model_dir, args.model)


if __name__ == "__main__":
    main()
