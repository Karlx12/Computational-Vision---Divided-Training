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
        choices=["cnn", "vgg", "densenet201", "resnet", "efficientnetb0"],
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
                f"--world-size. Recibido: {node_ips} "
                f"(cantidad: {len(node_ips)}) vs world_size: {args.world_size}"
            )

        # Configurar puertos únicos para cada worker
        # Cada nodo escucha en su propia IP con el mismo puerto base
        worker_ports = []
        base_port = args.master_port

        for i, ip in enumerate(node_ips):
            # Todos usan el mismo puerto pero en IPs diferentes
            worker_ports.append(f"{ip}:{base_port}")

        # Configuración específica del cluster para TensorFlow
        cluster_config = {
            "cluster": {"worker": worker_ports},
            "task": {"type": "worker", "index": args.rank},
        }

        # Variables de entorno para optimizar la comunicación distribuida
        os.environ["GRPC_VERBOSITY"] = "ERROR"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        os.environ["TF_DISABLE_MKL"] = "1"
        # Configuración para evitar timeouts
        os.environ["GRPC_POLL_STRATEGY"] = "poll"
        os.environ["GRPC_SO_REUSEPORT"] = "1"

        os.environ["TF_CONFIG"] = json.dumps(cluster_config)

        # Configuración de comunicación optimizada
        communication_options = tf.distribute.experimental.CommunicationOptions(
            timeout_seconds=1200,  # Timeout más largo para init
            implementation=tf.distribute.experimental.CommunicationImplementation.AUTO,
        )

        try:
            current_ip = node_ips[args.rank]
            print(f"🔄 Iniciando worker {args.rank} en {current_ip}...")
            print(f"🔧 TF_CONFIG: {cluster_config}")

            # Intentar crear estrategia distribuida
            strategy = tf.distribute.MultiWorkerMirroredStrategy(
                communication_options=communication_options
            )

            # Verificar que la estrategia se creó correctamente
            has_replicas = hasattr(strategy, "num_replicas_in_sync")
            valid_replicas = has_replicas and strategy.num_replicas_in_sync > 0
            if valid_replicas:
                print(
                    f"✅ Estrategia distribuida creada exitosamente."
                    f" Worker {args.rank}/{args.world_size}"
                )
                print(f" Worker ports: {worker_ports}")
                print(f"📈 Réplicas: {strategy.num_replicas_in_sync}")
            else:
                raise Exception("Num replicas no válido")

        except Exception as e:
            print(f"❌ Error al crear estrategia distribuida: {e}")
            print("🔄 Fallback a entrenamiento en un solo dispositivo")
            strategy = tf.distribute.get_strategy()
            # Limpiar configuración distribuida
            if "TF_CONFIG" in os.environ:
                del os.environ["TF_CONFIG"]
    else:
        strategy = tf.distribute.get_strategy()
        print("🔹 Entrenamiento en un solo dispositivo (CPU o una GPU)")

    return strategy, args
