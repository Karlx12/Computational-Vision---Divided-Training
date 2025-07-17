#!/bin/bash
# Lanzador para el worker 1 (rank 1)

# Cargar variables de entorno desde .env
set -a
source .env
set +a

# Configurar el rank específico para este worker
export RANK=1

# Usar el Python del entorno virtual si existe
if [ -x ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
else
    PYTHON_EXEC="python"
fi

# Variables de entorno para depuración de NCCL y TensorFlow
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=/tmp/nccl_debug_$HOSTNAME.log
export NCCL_SOCKET_IFNAME=eth0  # Cambiar eth0 si es necesario

export TF_CPP_MIN_LOG_LEVEL=0
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2

echo "🚀 Iniciando Worker 1 (RANK=$RANK) en $(hostname)"
echo "📡 Conectando a master: $MASTER_IP:$MASTER_PORT"
echo "🌐 Cluster: $NODE_IPS"

$PYTHON_EXEC main.py --master-ip "$MASTER_IP" --master-port "$MASTER_PORT" --rank "$RANK" --world-size "$WORLD_SIZE" --node-ips "$NODE_IPS" --model "$MODEL" --batch-size "$BATCH_SIZE" --epochs "$EPOCHS"
