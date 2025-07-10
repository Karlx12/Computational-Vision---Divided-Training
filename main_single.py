from argparse import Namespace
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
import keras
from datetime import datetime
from utils.config import (
    get_directories,
    setup_directories,
    get_img_size,
    get_batch_size,
    get_epochs,
    get_model,
    get_finetune_trainable_layers,
    get_finetune_freeze_batchnorm,
    get_validation_dir,
)
from utils.data_loader import create_data_generators_training
from utils.model_reports import save_training_reports


def setup_training_components(args: Namespace, input_shape, num_classes):
    validation_dir = args.validation_dir or get_validation_dir()
    dirs: dict[str, Path] = get_directories(validation_dir=validation_dir)
    train_gen, val_gen = create_data_generators_training(
        dirs["DATASET_DIR"],
        input_shape,
        args.batch_size,
        validation_dir=validation_dir,
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

        trainable_layers = get_finetune_trainable_layers()
        freeze_batchnorm = get_finetune_freeze_batchnorm()
        model = DenseNet201FinetuneModel(
            input_shape, num_classes, trainable_layers, freeze_batchnorm
        ).model
    elif args.model == "resnet":
        from models.resnet import ResNet50FinetuneModel

        trainable_layers = get_finetune_trainable_layers()
        freeze_batchnorm = get_finetune_freeze_batchnorm()
        model = ResNet50FinetuneModel(
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
    IMG_SIZE = get_img_size()
    input_shape = (IMG_SIZE, IMG_SIZE, 3)  # 256 x 256 píxeles, 3 canales (RGB)
    num_classes: int = len(
        [d for d in dirs["DATASET_DIR"].iterdir() if d.is_dir()]
    )

    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=get_batch_size())
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "vgg", "densenet201", "resnet"],
        default=get_model(),
    )
    parser.add_argument("--epochs", type=int, default=get_epochs())
    parser.add_argument(
        "--validation-dir",
        type=str,
        default="",
        help="Directorio de validación externo (opcional)",
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
