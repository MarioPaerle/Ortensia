import socket
import threading
import json


class NetworkManager:
    def __init__(self, is_host=True, ip='127.0.0.1', port=5555):
        self.is_host = is_host
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)  # Rende il socket non bloccante

        if self.is_host:
            self.sock.bind(('0.0.0.0', port))
            self.target_addr = None  # Lo scopriamo quando il client si connette
        else:
            self.target_addr = (ip, port)
            # Manda un pacchetto fittizio per farsi riconoscere dall'host
            self.send_data({"type": "connect"})

        self.remote_player_data = None

        # Avvia il thread di ascolto
        self.listen_thread = threading.Thread(target=self._listen, daemon=True)
        self.listen_thread.start()

    def _listen(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(1024)
                if self.is_host and self.target_addr is None:
                    self.target_addr = addr  # Registra l'IP del client

                parsed = json.loads(data.decode())
                if parsed.get("type") == "player_update":
                    self.remote_player_data = parsed
            except BlockingIOError:
                pass  # Nessun dato ricevuto
            except Exception as e:
                pass

    def send_data(self, data):
        if self.target_addr:
            try:
                self.sock.sendto(json.dumps(data).encode(), self.target_addr)
            except:
                pass