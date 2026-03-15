import pygame
import sys
# Importa il NetworkManager che hai salvato in Graphic/network.py
from Graphic.network import NetworkManager

# Inizializzazione Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ortensia Multiplayer Test")
clock = pygame.time.Clock()

# Colori per i diversi "stati" del giocatore
COLORS = [(255, 50, 50), (50, 255, 50), (50, 50, 255), (255, 255, 50)]


class DummyPlayer:
    def __init__(self, x, y, is_local=True):
        self.x = x
        self.y = y
        self.color_idx = 0
        self.is_local = is_local
        self.radius = 30
        self.speed = 5

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def change_state(self):
        # Passa al colore successivo
        self.color_idx = (self.color_idx + 1) % len(COLORS)

    def draw(self, surface):
        color = COLORS[self.color_idx]
        # Bordo bianco se sei tu, bordo grigio se è l'altro giocatore
        border_color = (255, 255, 255) if self.is_local else (100, 100, 100)

        # Disegna il cerchio riempito e poi il contorno
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(surface, border_color, (int(self.x), int(self.y)), self.radius, 3)


def main():
    # 1. SETUP DELLA RETE TRAMITE CONSOLE
    scelta = input("Vuoi essere Host (1) o Client (2)? ")
    if scelta == '1':
        print("Avvio come HOST in ascolto sulla porta 5555...")
        net = NetworkManager(is_host=True, port=5555)
        # L'Host spawna a sinistra, di colore rosso
        local_player = DummyPlayer(200, 300, is_local=True)
        local_player.color_idx = 0
    else:
        ip = input("Inserisci l'IP dell'Host (premi Invio per 127.0.0.1): ")
        if ip == "": ip = "127.0.0.1"
        print(f"Connessione a {ip}:5555...")
        net = NetworkManager(is_host=False, ip=ip, port=5555)
        # Il client spawna a destra, di colore blu
        local_player = DummyPlayer(600, 300, is_local=True)
        local_player.color_idx = 2

        # Spawna l'altro giocatore fuori dallo schermo all'inizio
    remote_player = DummyPlayer(-100, -100, is_local=False)

    running = True
    while running:
        # --- 2. GESTIONE INPUT ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    local_player.change_state()  # Cambia colore premendo barra spaziatrice

        keys = pygame.key.get_pressed()
        dx, dy = 0, 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:  dx -= local_player.speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: dx += local_player.speed
        if keys[pygame.K_UP] or keys[pygame.K_w]:    dy -= local_player.speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:  dy += local_player.speed

        local_player.move(dx, dy)

        # --- 3. LOGICA DI RETE ---
        # A) Inviamo la nostra posizione e il nostro stato
        net.send_data({
            "type": "player_update",
            "x": local_player.x,
            "y": local_player.y,
            "color_idx": local_player.color_idx
        })

        # B) Leggiamo i dati ricevuti per aggiornare l'altro giocatore
        if net.remote_player_data:
            remote_player.x = net.remote_player_data.get("x", remote_player.x)
            remote_player.y = net.remote_player_data.get("y", remote_player.y)
            remote_player.color_idx = net.remote_player_data.get("color_idx", remote_player.color_idx)

        # --- 4. RENDERIZZAZIONE ---
        screen.fill((30, 30, 40))  # Sfondo grigio scuro

        # Disegniamo prima il remoto, così il player locale viene disegnato sopra se si sovrappongono
        remote_player.draw(screen)
        local_player.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()