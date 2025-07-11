import os
import json
import argparse
import tensorflow as tf
from typing import Tuple

from tensorflow.python.distribute.collective_all_reduce_strategy import (
    CollectiveAllReduceStrategy,
)
from tensorflow.python.distribute.distribute_lib import StrategyBase


def configure_distributed_environment() -> Tuple[
    CollectiveAllReduceStrategy | StrategyBase, argparse.Namespace
]:
    """Configura el entorno para entrenamiento distribuido."""
    parser = argparse.ArgumentParser()

    # Configuración
    parser.add_argument(
        "--batch-size",
        type=int,
        default=48,
        help="Batch size por GPU (48-64)",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=1,
        help="GPUs por nodo (1)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=2,
        help="Pasos de acumulación para batches grandes",
    )

    # Argumentos para distribución
    parser.add_argument(
        "--master-ip",
        type=str,
        default="0.0.0.0",
        help="IP del nodo maestro",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=23456,
        help="Puerto base para comunicación",
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="Rango del nodo (0 para maestro)"
    )
    parser.add_argument(
        "--world-size", type=int, default=2, help="Número total de nodos"
    )
    parser.add_argument(
        "--node-ips",
        type=str,
        default="192.168.0.1,192.168.0.2",
        help="Lista de IPs de los nodos separadas por coma",
    )

    # Argumentos del modelo
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "vgg", "densenet201", "resnet"],
        default="cnn",
        help="Tipo de modelo a entrenar",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Número de épocas"
    )

    args: argparse.Namespace = parser.parse_args()

    # Configuración de estrategia distribuida
    if args.world_size > 1:
        node_ips = args.node_ips.split(",")
        if len(node_ips) != args.world_size:
            raise ValueError(
                f"El número de IPs en --node-ips debe coincidir con "
                f"--world-size. Recibido: {node_ips}"
            )

        os.environ["TF_CONFIG"] = json.dumps(
            {
                "cluster": {
                    "worker": [f"{ip}:{args.master_port}" for ip in node_ips]
                },
                "task": {"type": "worker", "index": args.rank},
            }
        )
        communication_options = tf.distribute.experimental.CommunicationOptions(  # noqa
            timeout_seconds=300,  # Tiempo de espera para la comunicación
            implementation=tf.distribute.experimental.CommunicationImplementation.NCCL,
        )
        strategy = tf.distribute.MultiWorkerMirroredStrategy(
            communication_options=communication_options
        )
        print(
            "🚀 Modo distribuido activado."
            f" Nodo {args.rank} de {args.world_size}"
        )
        print(f"🔧 TF_CONFIG: {os.environ['TF_CONFIG']}")
    else:
        strategy = tf.distribute.get_strategy()
        print("🔹 Entrenamiento en un solo dispositivo (CPU o una GPU)")

    return strategy, args
