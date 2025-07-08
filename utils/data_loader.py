from tensorflow.keras.preprocessing.image import ImageDataGenerator  # pyright: ignore[reportMissingImports]


def get_last_checkpoint(checkpoints_dir):
    checkpoints = list(checkpoints_dir.glob("cp-*.ckpt.index"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.stem.split("-")[1].split(".")[0]))
    last_checkpoint = checkpoints[-1].stem.replace(".index", "")
    return str(checkpoints_dir / last_checkpoint)


def create_data_generators_training(dataset_dir, input_shape, batch_size):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )
    train_gen = datagen.flow_from_directory(
        str(dataset_dir),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode="sparse",
        subset="training",
        shuffle=True,
    )
    val_gen = datagen.flow_from_directory(
        str(dataset_dir),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode="sparse",
        subset="validation",
        shuffle=False,
    )
    total_images = train_gen.samples + val_gen.samples
    print(
        f"🖼️ Total de imágenes generadas por ImageDataGenerator: {total_images}"
    )
    return train_gen, val_gen


def create_data_generators_fine_tunning(dataset_dir, input_shape, batch_size):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )
    train_gen = datagen.flow_from_directory(
        str(dataset_dir),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode="sparse",
        subset="training",
        shuffle=True,
    )
    val_gen = datagen.flow_from_directory(
        str(dataset_dir),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode="sparse",
        subset="validation",
        shuffle=False,
    )
    return train_gen, val_gen
