from pathlib import Path
import random
from tqdm import tqdm
import albumentations as A
import cv2

# Configuración
INPUT_DIR = Path("dataset/Training")
OUTPUT_DIR = Path("dataset/preprocessed_mri")
TRAIN_DIR = OUTPUT_DIR / "training"
VAL_DIR = OUTPUT_DIR / "validation"
IMG_SIZE = 640
VAL_SPLIT = 0.2
AUG_PER_IMAGE = 0  # Hasta 3 imágenes augmentadas por original

# Transformaciones leves para MRI en escala de grises
transform = A.Compose(
    [
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
            rotate=(-5, 5),
            p=0.5,
        ),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.2),  # Ruido leve
        A.Resize(IMG_SIZE, IMG_SIZE),
    ]
)

# Crear carpetas destino
for split_dir in [TRAIN_DIR, VAL_DIR]:
    split_dir.mkdir(parents=True, exist_ok=True)

# Listar clases
classes = [d.name for d in INPUT_DIR.iterdir() if d.is_dir()]
for cls in classes:
    (TRAIN_DIR / cls).mkdir(exist_ok=True)
    (VAL_DIR / cls).mkdir(exist_ok=True)

# Procesar cada clase
for cls in tqdm(classes, desc="Procesando clases"):
    images = list((INPUT_DIR / cls).glob("*.jpg")) + list(
        (INPUT_DIR / cls).glob("*.png")
    )
    random.shuffle(images)
    n_val = int(len(images) * VAL_SPLIT)
    val_images = images[:n_val]
    train_images = images[n_val:]

    # Copiar imágenes de validación (sin augmentar)
    for img_path in val_images:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ No se pudo leer la imagen: {img_path}")
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        out_path = VAL_DIR / cls / img_path.name
        cv2.imwrite(str(out_path), img)

    # Augmentación para entrenamiento
    for img_path in train_images:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ No se pudo leer la imagen: {img_path}")
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Guardar imagen original
        out_path = TRAIN_DIR / cls / img_path.name
        cv2.imwrite(str(out_path), img)
        # Generar augmentaciones
        for i in range(AUG_PER_IMAGE):
            augmented = transform(image=img)["image"]
            aug_name = img_path.stem + f"_aug{i + 1}" + img_path.suffix
            aug_path = TRAIN_DIR / cls / aug_name
            cv2.imwrite(str(aug_path), augmented)

print(
    "✅ Augmentación y partición completadas (MRI, grises, leves). "
    "Las imágenes están en:",
    OUTPUT_DIR,
)
