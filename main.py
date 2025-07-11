from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from datetime import datetime
from utils.config import (
    get_directories,
    get_finetune_trainable_layers,
    get_finetune_freeze_batchnorm,
    get_validation_dir,
)
from utils.data_loader import create_data_generators_training
from utils.distributed import configure_distributed_environment
from utils.model_reports import save_training_reports
from models import (
    CNNModel,
    VGGModel,
    DenseNet201FinetuneModel,
    ResNet50FinetuneModel,
)


def setup_training_components(args, input_shape, num_classes):
    import keras

    validation_dir = args.validation_dir or get_validation_dir()
    dirs: dict[str, Path] = get_directories(validation_dir=validation_dir)
    train_gen, val_gen = create_data_generators_training(
        dirs["DATASET_DIR"],
        input_shape,
        args.batch_size,
        validation_dir=validation_dir,
    )
    # Selección de modelo
    if args.model == "cnn":
        model = CNNModel(input_shape, num_classes).model
    elif args.model == "vgg":
        model = VGGModel(input_shape, num_classes).model
    elif args.model == "densenet201":
        trainable_layers = get_finetune_trainable_layers()
        freeze_batchnorm = get_finetune_freeze_batchnorm()
        model = DenseNet201FinetuneModel(
            input_shape, num_classes, trainable_layers, freeze_batchnorm
        ).model
    elif args.model == "resnet":
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


def train_distributed(strategy, args):
    dirs = get_directories()
    input_shape = (256, 256, 3)
    num_classes: int = len(
        [d for d in dirs["DATASET_DIR"].iterdir() if d.is_dir()]
    )

    with strategy.scope():
        model, train_gen, val_gen, callbacks = setup_training_components(
            args, input_shape, num_classes
        )
        history = model.fit(
            train_gen,
            epochs=args.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            steps_per_epoch=train_gen.samples
            // (args.batch_size * args.gpus_per_node * args.world_size),
            validation_steps=val_gen.samples
            // (args.batch_size * args.gpus_per_node * args.world_size),
        )
        # Guardado solo por el worker 0 (chief)
        if args.rank == 0:
            date = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_dir = dirs["MODELS_DIR"] / f"{args.model}_final_{date}"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{args.model}_final_{date}.keras"
            model.save(model_path)
            print(f"✅ Modelo guardado en {model_path}")
            save_training_reports(
                model, history, val_gen, model_dir, args.model
            )


def main():
    import argparse
    from utils.gpu_config import configure_environment

    # Configurar GPU
    configure_environment()
    # Configurar entorno distribuido y argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "vgg", "densenet201", "resnet"],
        default="cnn",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--validation-dir",
        type=str,
        default="",
        help="Directorio de validación externo (opcional)",
    )
    # Agrega aquí otros argumentos si es necesario
    strategy, args = configure_distributed_environment()
    cli_args, _ = parser.parse_known_args()
    for k, v in vars(cli_args).items():
        setattr(args, k, v)
    # Ejecutar entrenamiento
    train_distributed(strategy, args)


if __name__ == "__main__":
    main()
