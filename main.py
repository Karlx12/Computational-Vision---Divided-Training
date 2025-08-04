from dotenv import load_dotenv

load_dotenv()
import argparse
from datetime import datetime
from utils.config import (
    get_directories,
    get_finetune_trainable_layers,
    get_finetune_freeze_batchnorm,
    get_validation_dir,
)
from utils.distributed import configure_distributed_environment
from models import (
    CNNModel,
    VGGLikeModel,
    DenseNet201FinetuneModel,
    ResNet50FinetuneModel,
    EfficientNetB0FinetuneModel,
    VGG16Model,
    VGG19Model,
)
from models.densenet121 import DenseNet121FinetuneModel
from models.efficientnetb1 import EfficientNetB1FinetuneModel
import os
import logging
import tensorflow as tf
import keras
from keras.utils import image_dataset_from_directory


def get_model_instance(model_name, input_shape, num_classes):
    trainable_layers = get_finetune_trainable_layers()
    freeze_batchnorm = get_finetune_freeze_batchnorm()
    if model_name == "cnn":
        return CNNModel(input_shape, num_classes).model
    elif model_name == "vgg":
        return VGGLikeModel(input_shape, num_classes).model
    elif model_name == "densenet201":
        return DenseNet201FinetuneModel(
            input_shape, num_classes, trainable_layers, freeze_batchnorm
        ).model
    elif model_name == "resnet":
        return ResNet50FinetuneModel(
            input_shape, num_classes, trainable_layers, freeze_batchnorm
        ).model
    elif model_name == "efficientnetb0":
        return EfficientNetB0FinetuneModel(
            input_shape, num_classes, trainable_layers, freeze_batchnorm
        ).model
    elif model_name == "vgg16":
        return VGG16Model(
            input_shape, num_classes, trainable_layers, freeze_batchnorm
        ).model
    elif model_name == "vgg19":
        return VGG19Model(
            input_shape, num_classes, trainable_layers, freeze_batchnorm
        ).model
    elif model_name == "densenet121":
        return DenseNet121FinetuneModel(
            input_shape, num_classes, trainable_layers, freeze_batchnorm
        ).model
    elif model_name == "efficientnetb1":
        return EfficientNetB1FinetuneModel(
            input_shape, num_classes, trainable_layers, freeze_batchnorm
        ).model
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")


def create_distributed_dataset(
    directory, input_shape, batch_size, validation_split=0.2
):
    # Crear dataset principal
    train_ds = image_dataset_from_directory(
        directory,
        image_size=input_shape[:2],
        batch_size=batch_size,
        label_mode="sparse_categorical_crossentropy",
        shuffle=True,
        seed=42,
        validation_split=validation_split,
        subset="training",
    )

    # Dataset de validación
    val_ds = image_dataset_from_directory(
        directory,
        image_size=input_shape[:2],
        batch_size=batch_size,
        label_mode="sparse_categorical_crossentropy",
        shuffle=True,
        seed=42,
        validation_split=validation_split,
        subset="validation",
    )

    # Función de normalización
    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    # Aplicar normalización y optimización
    train_ds = train_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    # Optimización de rendimiento
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def get_callbacks(dirs, model_name):
    return [
        keras.callbacks.ModelCheckpoint(
            monitor="val_accuracy",
            mode="max",
            verbose=1,
            filepath=str(
                dirs["CHECKPOINTS_DIR"] / f"{model_name}_weights.weights.h5"
            ),
            save_weights_only=True,
            save_best_only=True,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(
                dirs["LOG_DIR"]
                / f"{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
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


def train_distributed(strategy, args):
    dirs = get_directories()
    input_shape = (256, 256, 3)
    num_classes = len([d for d in dirs["DATASET_DIR"].iterdir() if d.is_dir()])
    log_dir = dirs["LOG_DIR"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    with strategy.scope():
        global_batch_size = args.batch_size
        per_worker_batch_size = (
            global_batch_size // strategy.num_replicas_in_sync
        )
        logger.info(f"Batch size por worker: {per_worker_batch_size}")
        train_dataset = create_distributed_dataset(
            dirs["DATASET_DIR"], input_shape, per_worker_batch_size
        )
        val_dir = get_validation_dir()
        if val_dir:
            val_dataset = create_distributed_dataset(
                val_dir, input_shape, per_worker_batch_size
            )
        else:
            val_dataset = None
        model = get_model_instance(args.model, input_shape, num_classes)
        callbacks = []
        # Chief worker only
        task_id = getattr(strategy.cluster_resolver, "task_id", 0)
        if task_id == 0:
            checkpoint_path = os.path.join(
                dirs["CHECKPOINTS_DIR"], "model_{epoch}"
            )
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path, save_weights_only=True
                )
            )
            callbacks.append(keras.callbacks.BackupAndRestore("/tmp/backup"))
        logger.info("Comenzando entrenamiento distribuido...")
        model.get_model().fit(
            train_dataset,
            epochs=args.epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
        )
        logger.info("Entrenamiento finalizado.")
        if task_id == 0:
            final_model_path = os.path.join(dirs["MODELS_DIR"], "final_model")
            model.get_model().save(final_model_path)
            logger.info(f"Modelo guardado en {final_model_path}")


def main():
    from utils.gpu_config import configure_environment

    # Configurar GPU
    configure_environment()
    # Configurar entorno distribuido y argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "cnn",
            "vgg",
            "densenet201",
            "resnet",
            "efficientnetb0",
            "vgg16",
            "vgg19",
            "densenet121",
            "efficientnetb1",
        ],
        default="cnn",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--validation-dir",
        type=str,
        default="",
        help="Directorio de validación externo",
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
