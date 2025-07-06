#!/bin/bash
# Lanzador para el nodo master

# Cargar variables de entorno desde .env
set -a
source .env
set +a

python main.py --master-ip "$MASTER_IP" --master-port "$MASTER_PORT" --rank 0 --world-size "$WORLD_SIZE" --node-ips "$NODE_IPS" --model "$MODEL" --batch-size "$BATCH_SIZE" --epochs "$EPOCHS"
