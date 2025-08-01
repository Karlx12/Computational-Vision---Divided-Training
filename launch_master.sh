#!/bin/bash
# Lanzador para el nodo master (rank 0)

# Cargar variables de entorno desde .env
set -a
source .env
set +a

# Asegurarse de que el master tenga rank 0
export RANK=0

# Usar el Python del entorno virtual si existe
if [ -x ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
else
    PYTHON_EXEC="python"
fi

# Configuración de depuración para NCCL y TensorFlow
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=/tmp/nccl_debug_$HOSTNAME.log
export NCCL_SOCKET_IFNAME=eth0  

export TF_CPP_MIN_LOG_LEVEL=0
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2

echo "🎯 Iniciando Master (RANK=$RANK) en $(hostname)"
echo "📡 Master IP: $MASTER_IP:$MASTER_PORT"
echo "🌐 Cluster: $NODE_IPS"
echo "👥 World Size: $WORLD_SIZE"
echo "🧠 Modelo: $MODEL"

$PYTHON_EXEC main.py --master-ip "$MASTER_IP" --master-port "$MASTER_PORT" --rank "$RANK" --world-size "$WORLD_SIZE" --node-ips "$NODE_IPS" --model "$MODEL" --batch-size "$BATCH_SIZE" --epochs "$EPOCHS"
