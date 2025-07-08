from tensorflow.keras.preprocessing.image import ImageDataGenerator  # pyright: ignore[reportMissingImports]


def get_last_checkpoint(checkpoints_dir):
    checkpoints = list(checkpoints_dir.glob("cp-*.ckpt.index"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.stem.split("-")[1].split(".")[0]))
    last_checkpoint = checkpoints[-1].stem.replace(".index", "")
    return str(checkpoints_dir / last_checkpoint)


def create_data_generators(dataset_dir, input_shape, batch_size):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,  # 20% para validación
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
