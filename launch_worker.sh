#!/bin/bash
# Lanzador para un nodo worker

# Cargar variables de entorno desde .env
set -a
source .env
set +a

python main.py --master-ip "$MASTER_IP" --master-port "$MASTER_PORT" --rank "$RANK" --world-size "$WORLD_SIZE" --node-ips "$NODE_IPS" --model "$MODEL" --batch-size "$BATCH_SIZE" --epochs "$EPOCHS"
