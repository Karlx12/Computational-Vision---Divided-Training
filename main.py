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
from utils.data_loader import create_data_generators_training
from utils.distributed import configure_distributed_environment
from utils.model_reports import save_training_reports
from models import (
    CNNModel,
    VGGLikeModel,
    DenseNet201FinetuneModel,
    ResNet50FinetuneModel,
    EfficientNetB0FinetuneModel,
    VGG16Model,
    VGG19Model,
)
import keras
import tensorflow as tf


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
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")


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


def get_data_generators(dirs, input_shape, batch_size, validation_dir):
    return create_data_generators_training(
        dirs["DATASET_DIR"],
        input_shape,
        batch_size,
        validation_dir=validation_dir,
    )


def setup_training_components(args, input_shape, num_classes):
    validation_dir = args.validation_dir or get_validation_dir()
    dirs = get_directories(validation_dir=validation_dir)
    train_gen, val_gen = get_data_generators(
        dirs, input_shape, args.batch_size, validation_dir
    )
    model = get_model_instance(args.model, input_shape, num_classes)
    callbacks = get_callbacks(dirs, args.model)
    return model, train_gen, val_gen, callbacks


def train_distributed(strategy, args):
    dirs = get_directories()
    input_shape = (256, 256, 3)
    num_classes = len([d for d in dirs["DATASET_DIR"].iterdir() if d.is_dir()])
    with strategy.scope():
        validation_dir = getattr(args, "validation_dir", get_validation_dir())
        dirs_full = get_directories(validation_dir=validation_dir)
        train_gen, val_gen = get_data_generators(
            dirs_full, input_shape, args.batch_size, validation_dir
        )
        model = get_model_instance(args.model, input_shape, num_classes)
        callbacks = get_callbacks(dirs_full, args.model)

        def generator_wrapper(gen):
            while True:
                x, y = gen.next()
                yield x, y

        train_dataset = tf.data.Dataset.from_generator(
            lambda: generator_wrapper(train_gen),
            output_types=(tf.float32, tf.int32),
            output_shapes=((None, *input_shape), (None,)),
        ).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_generator(
            lambda: generator_wrapper(val_gen),
            output_types=(tf.float32, tf.int32),
            output_shapes=((None, *input_shape), (None,)),
        ).prefetch(tf.data.AUTOTUNE)
        history = model.fit(
            train_dataset,
            epochs=args.epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            steps_per_epoch=train_gen.samples
            // (args.batch_size * args.gpus_per_node * args.world_size),
            validation_steps=val_gen.samples
            // (args.batch_size * args.gpus_per_node * args.world_size),
        )
        if args.rank == 0:
            model_dir = dirs_full["MODELS_DIR"] / (
                f"{args.model}_final_"
                + datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / (
                f"{args.model}_final_"
                + datetime.now().strftime("%Y%m%d-%H%M%S")
                + ".keras"
            )
            model.save(model_path)
            print(f"✅ Modelo guardado en {model_path}")
            save_training_reports(
                model, history, val_gen, model_dir, args.model
            )


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
        choices=["cnn", "vgg", "densenet201", "resnet", "efficientnetb0"],
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
