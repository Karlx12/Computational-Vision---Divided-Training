import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.getenv("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from datetime import datetime
from utils.config import (
    get_directories,
    setup_directories,
    get_finetune_trainable_layers,
    get_finetune_freeze_batchnorm,
)
from utils.data_loader import create_data_generators_training
from utils.distributed import configure_distributed_environment
from utils.model_reports import save_training_reports


def setup_training_components(args, input_shape, num_classes):
    dirs: dict[str, Path] = get_directories()
    train_gen, val_gen = create_data_generators_training(
        dirs["DATASET_DIR"], input_shape, args.batch_size
    )
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
    else:
        raise ValueError(f"Modelo no soportado: {args.model}")
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(  # type: ignore
            monitor="val_accuracy",
            mode="max",
            verbose=1,
            dirpath=str(dirs["CHECKPOINTS_DIR"]),
            filepath=str(
                dirs["CHECKPOINTS_DIR"] / f"{args.model}_weights.weights.h5"
            ),
            save_weights_only=True,
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(  # type: ignore
            log_dir=str(
                dirs["LOG_DIR"]
                / f"{args.model}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
        ),
        tf.keras.callbacks.ReduceLROnPlateau(  # type: ignore
            monitor="val_accuracy",
            mode="max",
            factor=0.1,
            patience=3,
            min_lr=1e-6,
        ),
    ]
    return model, train_gen, val_gen, callbacks


def train_distributed(strategy, args):
    setup_directories()
    dirs = get_directories()
    input_shape = (256, 256, 3)
    num_classes: int = len(
        [d for d in dirs["DATASET_DIR"].iterdir() if d.is_dir()]
    )
    with strategy.scope():
        model, train_gen, val_gen, callbacks = setup_training_components(
            args, input_shape, num_classes
        )
        history = model.fit(  # type: ignore
            train_gen,
            epochs=args.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            steps_per_epoch=train_gen.samples
            // (args.batch_size * args.gpus_per_node * args.world_size),
            validation_steps=val_gen.samples
            // (args.batch_size * args.gpus_per_node * args.world_size),
        )
        if args.rank == 0:
            # Crear carpeta específica para el modelo final
            date = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_dir = dirs["MODELS_DIR"] / f"{args.model}_final_{date}"
            model_dir.mkdir(parents=True, exist_ok=True)
            # Guardar modelo
            model_path = model_dir / f"{args.model}_final_{date}.keras"
            model.save(model_path)  # type: ignore
            print(f"✅ Modelo guardado en {model_path}")
            # Guardar reportes y gráficas
            save_training_reports(
                model, history, val_gen, model_dir, args.model
            )


def main():
    # Configurar directorios primero
    setup_directories()
    # Configurar GPU
    from utils.gpu_config import configure_environment

    configure_environment()
    # Configurar entorno distribuido
    strategy, args = configure_distributed_environment()
    # Ejecutar entrenamiento
    train_distributed(strategy, args)


if __name__ == "__main__":
    main()
