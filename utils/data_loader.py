from keras.src.legacy.preprocessing.image import ImageDataGenerator
from typing import Optional


def get_last_checkpoint(checkpoints_dir):
    checkpoints = list(checkpoints_dir.glob("cp-*.ckpt.index"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.stem.split("-")[1].split(".")[0]))
    last_checkpoint = checkpoints[-1].stem.replace(".index", "")
    return str(checkpoints_dir / last_checkpoint)


def _create_data_generators(
    dataset_dir,
    input_shape,
    batch_size,
    augment=False,
    validation_dir: Optional[str] = "",
):
    use_custom_val = bool(validation_dir)
    if augment:
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
            validation_split=0.2 if not use_custom_val else 0.0,
        )
    else:
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2 if not use_custom_val else 0.0,
        )
    train_gen = datagen.flow_from_directory(
        str(dataset_dir),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode="sparse",
        subset="training" if not use_custom_val else None,
        shuffle=True,
    )
    if use_custom_val:
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)
        val_gen = val_datagen.flow_from_directory(
            str(validation_dir),
            target_size=input_shape[:2],
            batch_size=batch_size,
            class_mode="sparse",
            shuffle=False,
        )
    else:
        val_gen = datagen.flow_from_directory(
            str(dataset_dir),
            target_size=input_shape[:2],
            batch_size=batch_size,
            class_mode="sparse",
            subset="validation",
            shuffle=False,
        )
    return train_gen, val_gen


def create_data_generators_training(
    dataset_dir, input_shape, batch_size, validation_dir: Optional[str] = ""
):
    train_gen, val_gen = _create_data_generators(
        dataset_dir,
        input_shape,
        batch_size,
        augment=True,
        validation_dir=validation_dir,
    )
    total_images = train_gen.samples + val_gen.samples
    print(
        f"🖼️ Total de imágenes generadas por ImageDataGenerator: {total_images}"
    )
    return train_gen, val_gen


def create_data_generators_fine_tunning(
    dataset_dir, input_shape, batch_size, validation_dir: Optional[str] = ""
):
    return _create_data_generators(
        dataset_dir,
        input_shape,
        batch_size,
        augment=False,
        validation_dir=validation_dir,
    )
