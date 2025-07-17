#!/bin/bash
# Lanzador para el worker 3 (rank 3)

# Cargar variables de entorno desde .env
set -a
source .env
set +a

# Configurar el rank específico para este worker
export RANK=3

# Usar el Python del entorno virtual si existe
if [ -x ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
else
    PYTHON_EXEC="python"
fi

echo "🚀 Iniciando Worker 3 (RANK=$RANK) en $(hostname)"
echo "📡 Conectando a master: $MASTER_IP:$MASTER_PORT"
echo "🌐 Cluster: $NODE_IPS"

$PYTHON_EXEC main.py --master-ip "$MASTER_IP" --master-port "$MASTER_PORT" --rank "$RANK" --world-size "$WORLD_SIZE" --node-ips "$NODE_IPS" --model "$MODEL" --batch-size "$BATCH_SIZE" --epochs "$EPOCHS"
