import socket
import pickle
import zlib
import time


class ClusterSyncManager:
    def __init__(
        self, master_ip: str, master_port: int, rank: int, world_size: int
    ):
        self.rank: int = rank
        self.world_size: int = world_size
        self.connections = []

        if rank == 0:  # Master
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind((master_ip, master_port))
            self.server.listen(world_size - 1)

            print(f"🛰 Master escuchando en {master_ip}:{master_port}")
            for _ in range(world_size - 1):
                conn, addr = self.server.accept()
                self.connections.append(conn)
                print(
                    f"🔗 Worker {len(self.connections)} conectado desde {addr}"
                )
        else:  # Worker
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            while True:
                try:
                    self.connection.connect((master_ip, master_port))
                    break
                except ConnectionRefusedError:
                    print("⏳ Esperando conexión con master...")
                    time.sleep(5)
            print(f"📡 Worker {rank} conectado al master")

    def broadcast_model(self, model, model_name: str):
        """Sincroniza el modelo entre todos los nodos"""
        if self.rank == 0:
            # Serializar modelo
            weights = model.get_weights()
            data = zlib.compress(pickle.dumps(weights))

            # Enviar a todos los workers
            for conn in self.connections:
                try:
                    conn.sendall(len(data).to_bytes(4, "big"))
                    conn.sendall(data)
                except Exception as e:
                    print(f"⚠️ Error en broadcast: {e}")
        else:
            # Recibir modelo
            try:
                size = int.from_bytes(self.connection.recv(4), "big")
                received = bytearray()

                while len(received) < size:
                    chunk = self.connection.recv(
                        min(4096, size - len(received))
                    )
                    if not chunk:
                        break
                    received.extend(chunk)

                weights = pickle.loads(zlib.decompress(received))
                model.set_weights(weights)
                print(f"🔄 Modelo {model_name} sincronizado desde master")
            except Exception as e:
                print(f"⚠️ Error recibiendo modelo: {e}")

    def close(self) -> None:
        if self.rank == 0:
            for conn in self.connections:
                conn.close()
            self.server.close()
        else:
            self.connection.close()


def wait_for_master(
    master_ip: str, master_port: int, max_retries=30, retry_delay=5
):
    retry_count = 0
    while retry_count < max_retries:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((master_ip, master_port))
            sock.close()
            print("✅ Master está listo")
            return True
        except (ConnectionRefusedError, socket.timeout):
            print(
                "⏳ Esperando a master..."
                f"(Intento {retry_count + 1}/{max_retries})"
            )
            time.sleep(retry_delay)
            retry_count += 1
    raise ConnectionError(
        "No se pudo conectar al master después de múltiples intentos"
    )
