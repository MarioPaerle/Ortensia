import pygame
import os
import json
import sys

# Importa le classi esistenti dal tuo progetto
# Assicurati che main.py sia nella stessa cartella
from main import Engine
from Graphic._layers import UILayer, LitLayer, ChunkedLayer, Layer
from Graphic._sprites import Sprite
from Graphic.gui import UIText, UIButton

# --- CONFIGURAZIONE ---
TEXTURE_PATH = "assets/textures"
OUTPUT_FILE = "saves/level_decor_data.json"


def get_all_textures(path):
    """Scansiona ricorsivamente la cartella per trovare file .png"""
    tex_list = []
    if not os.path.exists(path):
        print(f"Attenzione: Cartella '{path}' non trovata.")
        return tex_list

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".png"):
                # Normalizza i path per evitare problemi tra Windows/Linux
                full_path = os.path.join(root, file).replace("\\", "/")
                tex_list.append(full_path)
    return sorted(tex_list)


class EditorEngine(Engine):
    def __init__(self, name='Ortensia Level Editor', base_size=(1000, 600)):
        super().__init__(name=name, base_size=base_size)

        # Stato dell'Editor
        self.textures = get_all_textures(TEXTURE_PATH)
        self.current_tex_index = 0
        self.selected_layer_index = 0
        self.editor_layers = []  # Lista dei layer modificabili
        self.cam_speed = 600
        self.mode = "PLACE"  # PLACE, DELETE
        self.show_grid = True
        self.main_camera.y = 400

        # UI Setup
        self.ui_layer = UILayer("EditorUI")
        self.add_layer(self.ui_layer)
        self.init_ui()

        # Carica dati precedenti se esistono
        self.load_if_exists()

    def register_editable_layers(self, layers):
        """Registra i layer su cui l'editor può agire"""
        self.editor_layers = layers

    def init_ui(self):
        # Etichette informative

        help_y = self.base_size[1] - 120
        help_x = self.base_size[0] - 120
        self.ui_layer.add_element(UIText(10, help_y, "[WASD] Muovi Camera | [TAB] Cambia Layer", 14))
        self.ui_layer.add_element(UIText(10, help_y + 20, "[t/y] Cambia Texture | [M] Cambia Mode (Place/Del)", 14))
        self.ui_layer.add_element(UIText(10, help_y + 40, "[K] Salva Livello | [L] Ricarica", 14))

        self.lbl_info = UIText(help_x, help_y, "Ortensia Editor", size=20, color=(255, 215, 0))
        self.lbl_layer = UIText(help_x, help_y+20, "Layer: ", size=18)
        self.lbl_mode = UIText(help_x, help_y+40, "Mode: ", size=18)
        self.lbl_tex = UIText(help_x, help_y+60, "Texture: ", size=14)
        self.lbl_pos = UIText(help_x, help_y+80, "x, y: ", size=16)
        self.ui_layer.add_element(self.lbl_info)
        self.ui_layer.add_element(self.lbl_layer)
        self.ui_layer.add_element(self.lbl_mode)
        self.ui_layer.add_element(self.lbl_tex)
        self.ui_layer.add_element(self.lbl_pos)

    def update_ui(self):
        # Aggiorna i testi
        if self.editor_layers:
            curr_l = self.editor_layers[self.selected_layer_index]
            self.lbl_layer.set_text(f"Layer: {curr_l.name} (Parallax: {curr_l.parallax})")

        self.lbl_mode.set_text(f"Mode: {self.mode}")

        if self.textures:
            tex_name = self.textures[self.current_tex_index].split("/")[-1]
            self.lbl_tex.set_text(f"Tex ({self.current_tex_index + 1}/{len(self.textures)}): {tex_name}")
        else:
            self.lbl_tex.set_text("Nessuna texture trovata in assets/")

        self.lbl_pos.set_text(f"{pygame.mouse.get_pos()[0] + self.main_camera.x: .0f}, {pygame.mouse.get_pos()[1] + self.main_camera.y: .0f}")

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                self.selected_layer_index = (self.selected_layer_index + 1) % len(self.editor_layers)

            elif event.key == pygame.K_m:
                self.mode = "DELETE" if self.mode == "PLACE" else "PLACE"

            elif event.key == pygame.K_y:
                if self.textures:
                    self.current_tex_index = (self.current_tex_index + 1) % len(self.textures)
            elif event.key == pygame.K_t:
                if self.textures:
                    self.current_tex_index = (self.current_tex_index - 1) % len(self.textures)

            elif event.key == pygame.K_k:
                self.save_level()
            elif event.key == pygame.K_l:
                self.load_if_exists()

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Click Sinistro
                self.use_tool(pygame.mouse.get_pos())

    def mechaniches(self, dt):
        # Movimento Camera (Sostituisce mechaniches del Player)
        keys = pygame.key.get_pressed()
        speed = self.cam_speed
        if keys[pygame.K_LSHIFT]: speed *= 2 # Turbo

        # CORREZIONE: Modifichiamo scroll_x/scroll_y invece di x/y
        if keys[pygame.K_a]: self.main_camera.scroll_x -= speed * dt
        if keys[pygame.K_d]: self.main_camera.scroll_x += speed * dt
        if keys[pygame.K_w]: self.main_camera.scroll_y -= speed * dt
        if keys[pygame.K_s]: self.main_camera.scroll_y += speed * dt

    def use_tool(self, mouse_pos):
        if not self.editor_layers: return

        mx, my = mouse_pos
        layer = self.editor_layers[self.selected_layer_index]
        cam = self.main_camera

        # Calcolo coordinate nel mondo (compensando parallasse e camera)
        # World = Screen + (Camera * Parallax)
        wx = mx + (cam.x * layer.parallax)
        wy = my + (cam.y * layer.parallax)

        if self.mode == "PLACE" and self.textures:
            tex_path = self.textures[self.current_tex_index]

            # Carica al volo per ottenere dimensioni
            temp_surf = pygame.image.load(tex_path)
            w, h = temp_surf.get_size()

            # Piazza centrato sul mouse
            x = wx - (w // 2)
            y = wy - (h // 2)

            # Crea Sprite
            s = Sprite(x, y, w, h, texture=tex_path, alpha=True)
            s.texture_path = tex_path  # Importante per il salvataggio!

            # Aggiungi al layer (supporta sia LitLayer che ChunkedLayer)
            if hasattr(layer, 'add_static'):
                layer.add_static(s)
            else:
                layer.sprites.append(s)

            print(f"Piazzato {os.path.basename(tex_path)} su {layer.name} a ({int(x)}, {int(y)})")

        elif self.mode == "DELETE":
            # Logica di cancellazione semplice (box check)
            # Nota: Con i chunked layer è complesso trovare l'oggetto velocemente,
            # qui controlliamo le liste principali per semplicità.
            to_remove = None

            # Unisci tutte le liste di sprite possibili del layer
            candidates = []
            if hasattr(layer, 'sprites'): candidates.extend(layer.sprites)
            if hasattr(layer, 'large_sprites'): candidates.extend(layer.large_sprites)
            # Per i chunk servirebbe iterare layer.chunks...
            if hasattr(layer, 'chunks'):
                for chunk_list in layer.chunks.values():
                    candidates.extend(chunk_list)

            for s in reversed(candidates):  # Cancella quello più in alto (ultimo disegnato)
                if s.x < wx < s.x + s.width and s.y < wy < s.y + s.height:
                    to_remove = s
                    break

            if to_remove:
                # Usa il metodo del layer per rimuovere correttamente
                if hasattr(layer, 'remove_static'):
                    layer.remove_static(to_remove)
                elif to_remove in layer.sprites:
                    layer.sprites.remove(to_remove)
                print("Oggetto rimosso.")

    def draw_preview(self):
        """Disegna lo sprite semitrasparente sotto il mouse"""
        if self.mode == "PLACE" and self.textures:
            mx, my = pygame.mouse.get_pos()
            tex_path = self.textures[self.current_tex_index]

            try:
                # Nota: In un vero engine faresti caching delle surface, qui carichiamo per semplicità
                surf = pygame.image.load(tex_path).convert_alpha()
                w, h = surf.get_size()
                surf.set_alpha(150)  # Semitrasparente
                self.screen.blit(surf, (mx - w // 2, my - h // 2))

                # Bordo indicativo
                pygame.draw.rect(self.screen, (0, 255, 0), (mx - w // 2, my - h // 2, w, h), 1)
            except:
                pass
        elif self.mode == "DELETE":
            # Cursore rosso
            mx, my = pygame.mouse.get_pos()
            pygame.draw.circle(self.screen, (255, 0, 0), (mx, my), 10, 2)

    def save_level(self):
        data = []
        print("Salvataggio in corso...")

        for layer in self.editor_layers:
            layer_data = {
                "name": layer.name,
                "parallax": layer.parallax,
                "type": layer.__class__.__name__,
                "sprites": []
            }

            # Raccogli tutti gli sprite (Statici, Dinamici, Chunked)
            all_sprites = []
            if hasattr(layer, 'sprites'): all_sprites.extend(layer.sprites)
            if hasattr(layer, 'large_sprites'): all_sprites.extend(layer.large_sprites)
            if hasattr(layer, 'chunks'):
                for chunk in layer.chunks.values():
                    all_sprites.extend(chunk)

            for s in all_sprites:
                # Salva solo se ha una texture associata
                if hasattr(s, 'texture_path') and s.texture_path:
                    all_sprites_data = {
                        "x": s.x,
                        "y": s.y,
                        "w": s.width,
                        "h": s.height,
                        "texture": s.texture_path,
                        "class": s.__class__.__name__
                    }
                    layer_data["sprites"].append(all_sprites_data)

            data.append(layer_data)

        os.makedirs("saves", exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Livello salvato correttamente in {OUTPUT_FILE}")

    def load_if_exists(self):
        if not os.path.exists(OUTPUT_FILE): return

        print(f"Caricamento {OUTPUT_FILE}...")
        try:
            with open(OUTPUT_FILE, 'r') as f:
                data = json.load(f)

            for l_data in data:
                # Trova il layer corrispondente per nome
                target_layer = next((l for l in self.editor_layers if l.name == l_data["name"]), None)
                if not target_layer: continue

                # Pulisci (opzionale, per ora aggiungiamo solo)
                # target_layer.sprites.clear()

                for s_data in l_data["sprites"]:
                    try:
                        s = Sprite(
                            s_data["x"], s_data["y"],
                            s_data["w"], s_data["h"],
                            texture=s_data["texture"],
                            alpha=True
                        )
                        s.texture_path = s_data["texture"]

                        if hasattr(target_layer, 'add_static'):
                            target_layer.add_static(s)
                        else:
                            target_layer.sprites.append(s)
                    except Exception as e:
                        print(f"Errore caricamento sprite: {e}")

            print("Caricamento completato.")
        except Exception as e:
            print(f"Errore caricamento JSON: {e}")

    # --- Override del ciclo principale per gestire eventi custom ---
    def update(self):
        dt = self.dt()
        camera = self.cameras[self.camera_id]

        # Event Loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Input Editor
            self.handle_input(event)

            # Input UI Layer
            self.ui_layer.process_events(event)

        # Logica Movimento
        self.mechaniches(dt)
        camera.update()
        self.update_ui()

        # Rendering
        self.screen.fill((40, 40, 50))

        for layer in self.layers:
            layer.render(self.screen, camera)

        self.draw_preview()

        pygame.display.set_caption(f"{self.name} | FPS: {int(self.clock.get_fps())}")
        pygame.display.flip()


# --- MAIN BLOCK ---
if __name__ == "__main__":
    # Inizializza l'EditorEngine
    game = EditorEngine()

    # --- DEFINIZIONE LAYER (Copia o adatta quelli di main.py) ---
    # Definisci qui i layer che vuoi nel tuo livello
    bg1 = game.add_create_layer("bg1", 0.001)
    bg2 = game.add_create_layer("bg2", 0.005)
    bg3 = game.add_create_layer("bg3", 0.01)
    bg4 = game.add_create_layer("bg4", 0.05)
    bg5 = game.add_create_layer("bg5", 0.1)

    # Foreground (LitLayer per luci o ChunkedLayer standard)
    fg = LitLayer("Foreground", 1.0)
    game.add_layer(fg)

    # Registra quali di questi layer vuoi poter modificare premendo TAB
    game.register_editable_layers([bg1, bg2, bg3, bg4, bg5, fg])

    # Avvia Loop
    while game.running:
        game.update()