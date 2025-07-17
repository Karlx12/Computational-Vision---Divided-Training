from argparse import Namespace
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
from models import (
    CNNModel,
    VGGLikeModel,
    DenseNet201FinetuneModel,
    ResNet50FinetuneModel,
    EfficientNetB0FinetuneModel,
    VGG16Model,
    VGG19Model,
)


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
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.CSVLogger(
            str(dirs["LOG_DIR"] / f"{model_name}_training_log.csv"),
            append=True,
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


def setup_training_components(args: Namespace, input_shape, num_classes):
    validation_dir = args.validation_dir or get_validation_dir()
    dirs = get_directories(validation_dir=validation_dir)
    train_gen, val_gen = get_data_generators(
        dirs, input_shape, args.batch_size, validation_dir
    )
    model = get_model_instance(args.model, input_shape, num_classes)
    callbacks = get_callbacks(dirs, args.model)
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
        choices=["cnn", "vgg", "densenet201", "resnet", "efficientnetb0"],
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
