import json
import random
from perlin_noise import PerlinNoise  # Requires: pip install perlin-noise

# ================= CONFIGURATION =================
OUTPUT_FILE = "saves/World/-blockmap_middle.json"  # Di solito le collisioni sono nel layer 'middle'

MAP_WIDTH = 100  # Larghezza della mappa in blocchi
MAP_HEIGHT = 30  # Altezza massima della mappa
BASE_HEIGHT = 15  # Livello base del terreno (y = 70)
AMPLITUDE = 25  # Quanto sono alte le montagne

# ================= PALETTE BLOCCHI =================
# Usiamo i blocchi che hai già registrato nel tuo blocks.py
ROCKS = ["deepslate2.png", "deepslate2_b.png", "deepslate2_c.png", "deepslate2_d.png", "cobbled_deepslate.png"]
DEEP_ROCK = "deepslate_down.png"
FLOWER = "BlueOrtensia.png"
GLASS = "HighStainedGlass"
LIGHT = "_LightBlue"
CHEST = "Chest"

block_data = {}


def place_block(x, y, block_id):
    """Piazza un blocco se è all'interno dei limiti della mappa."""
    if 0 <= x < MAP_WIDTH and 0 <= y < MAP_HEIGHT:
        key = str((x, y))
        block_data[key] = block_id


def get_block(x, y):
    """Ritorna il blocco presente in una coordinata, o None."""
    return block_data.get(str((x, y)))


def generate_tree(base_x, base_y):
    """Regola: Genera un albero incandescente (di roccia e vetro/luce)"""
    height = random.randint(4, 7)

    # Tronco
    for i in range(height):
        place_block(base_x, base_y - i, random.choice(ROCKS))

    # Chioma (Foglie/Vetro luminoso)
    top_y = base_y - height
    for dx in range(-2, 3):
        for dy in range(-2, 2):
            # Arrotonda gli angoli della chioma
            if abs(dx) == 2 and abs(dy) == 2:
                continue

            # Non sovrascrivere il tronco
            if dx == 0 and dy > 0:
                continue

            # Metti luce al centro, vetro intorno
            if abs(dx) <= 1 and abs(dy) <= 1 and random.random() < 0.3:
                place_block(base_x + dx, top_y + dy, LIGHT)
            else:
                place_block(base_x + dx, top_y + dy, GLASS)


def generate_map():
    print(f"Generazione di una mappa {MAP_WIDTH}x{MAP_HEIGHT} in corso...")

    # Inizializza i generatori di rumore
    seed = random.randint(1, 100000)
    terrain_noise = PerlinNoise(octaves=3, seed=seed)
    cave_noise = PerlinNoise(octaves=4, seed=seed + 1)

    surface_heights = []

    # FASE 1 & 2: Terreno e Caverne
    for x in range(MAP_WIDTH):
        # 1D Noise per la superficie (divido per 50 per rendere le colline dolci)
        noise_val = terrain_noise([x / 50.0])
        surface_y = int(BASE_HEIGHT + noise_val * AMPLITUDE)
        surface_heights.append(surface_y)

        for y in range(MAP_HEIGHT):
            if y < surface_y:
                # Aria (Sopra la superficie)
                continue

            elif y == surface_y:
                # Strato di superficie
                place_block(x, y, random.choice(ROCKS))

            else:
                # Sottosuolo: Controlliamo le caverne con 2D Noise
                # Moltiplico per scalare la dimensione delle grotte
                c_noise = cave_noise([x / 30.0, y / 30.0])

                # Se il noise supera una certa soglia (es. 0.25), creiamo un buco (caverna)
                if c_noise > 0.25:
                    continue
                else:
                    # Più si scende, più usiamo la roccia profonda
                    if y > surface_y + 20 and random.random() < 0.8:
                        place_block(x, y, DEEP_ROCK)
                    else:
                        place_block(x, y, random.choice(ROCKS))

    # FASE 3: Strutture Rule-based (Alberi, Fiori, Casse)
    print("Applicazione regole e strutture...")
    for x in range(MAP_WIDTH):
        surface_y = surface_heights[x]

        # 1. Piantiamo gli alberi sulla superficie (5% probabilità)
        if random.random() < 0.05:
            # Assicuriamoci che non ci sia già roba e che non siamo ai bordi
            if 5 < x < MAP_WIDTH - 5:
                generate_tree(x, surface_y - 1)

        # 2. Piantiamo fiori (Ortensie) (10% probabilità)
        elif random.random() < 0.10:
            if not get_block(x, surface_y - 1):  # Se c'è spazio vuoto sopra
                place_block(x, surface_y - 1, FLOWER)

    # FASE 4: Casse nelle caverne (Passata su tutta la mappa)
    for x in range(MAP_WIDTH):
        for y in range(MAP_HEIGHT):
            # Regola cassa: deve avere roccia sotto, aria sopra, ed essere profonda
            if y > BASE_HEIGHT + 10:
                block_below = get_block(x, y + 1)
                block_here = get_block(x, y)
                if block_below in ROCKS or block_below == DEEP_ROCK:
                    if block_here is None:
                        if random.random() < 0.005:  # Molto rare (0.5%)
                            place_block(x, y, CHEST)

    # Salvataggio
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(block_data, f)

    print(f"Successo! Mappa salvata in {OUTPUT_FILE}")
    print(f"Totale blocchi generati: {len(block_data)}")


if __name__ == "__main__":
    generate_map()