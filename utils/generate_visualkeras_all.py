import visualkeras
from pathlib import Path


def generate_visualizations(models_dir: Path, output_dir: Path = None):
    models = list(models_dir.glob("**/*.keras"))
    if not models:
        print(f"No se encontraron modelos .keras en {models_dir}")
        return
    import keras.layers as kl

    for model_path in models:
        try:
            from keras.models import load_model

            model = load_model(model_path, safe_mode=True)
            if output_dir is None:
                out_dir = model_path.parent
            else:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{model_path.stem}_visualkeras.png"

            model_name = model_path.stem.lower()
            hide_types = []
            scale_xy = 1.0
            if any(
                x in model_name
                for x in ["densenet", "efficientnet", "inceptionv3"]
            ):
                hide_types = [
                    kl.BatchNormalization,
                    kl.Activation,
                    kl.ZeroPadding2D,
                    kl.Dropout,
                    kl.Add,
                    kl.Concatenate,
                    kl.Multiply,
                    kl.Reshape,
                    kl.ReLU,
                    kl.GlobalAveragePooling2D,
                    kl.GlobalMaxPooling2D,
                    kl.AveragePooling2D,
                    kl.MaxPooling2D,
                    kl.DepthwiseConv2D,
                    kl.InputLayer,
                ]
                scale_xy = 0.5
            elif "nasnet" in model_name:
                hide_types = [
                    kl.BatchNormalization,
                    kl.Activation,
                    kl.ZeroPadding2D,
                    kl.Dropout,
                    kl.Add,
                    kl.Concatenate,
                    kl.Multiply,
                    kl.Reshape,
                    kl.ReLU,
                    kl.GlobalAveragePooling2D,
                    kl.GlobalMaxPooling2D,
                    kl.AveragePooling2D,
                    kl.MaxPooling2D,
                    kl.DepthwiseConv2D,
                    kl.InputLayer,
                ]

                sep_layers = [
                    l
                    for l in model.layers
                    if isinstance(l, kl.SeparableConv2D)  # noqa: E501
                ]
                if len(sep_layers) > 2:
                    # Mantener la primera y última, ocultar el resto
                    for l in sep_layers[1:-1]:
                        l._should_hide = True

                    class SeparableConv2DProxy(kl.SeparableConv2D):
                        pass

                    hide_types.append(SeparableConv2DProxy)
                scale_xy = 0.4

            # Visualización
            visualkeras.layered_view(
                model,
                to_file=str(out_file),
                scale_xy=scale_xy,
                max_z=8,
                legend=True,
                draw_volume=True,
                type_ignore=hide_types if hide_types else None,
            )
            print(f"🖼️ Visualización guardada: {out_file}")
        except Exception as e:
            print(f"❌ Error visualizando {model_path}: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Genera visualizaciones visualkeras para todos los modelos "
            ".keras en un directorio."
        )
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models_trained",
        help="Directorio raíz donde buscar modelos .keras (recursivo)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directorio de salida para las imágenes (opcional)",
    )
    args = parser.parse_args()
    generate_visualizations(Path(args.models_dir), args.output_dir)


if __name__ == "__main__":
    main()
