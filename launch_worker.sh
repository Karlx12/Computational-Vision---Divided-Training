#!/bin/bash
# Lanzador para un nodo worker

# Cargar variables de entorno desde .env
set -a
source .env
set +a

# Usar el Python del entorno virtual si existe
if [ -x ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
else
    PYTHON_EXEC="python"
fi

$PYTHON_EXEC main.py --master-ip "$MASTER_IP" --master-port "$MASTER_PORT" --rank "$RANK" --world-size "$WORLD_SIZE" --node-ips "$NODE_IPS" --model "$MODEL" --batch-size "$BATCH_SIZE" --epochs "$EPOCHS"
