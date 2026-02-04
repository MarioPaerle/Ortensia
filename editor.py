import pygame
import os
import json
import sys

# Import existing classes
from main import Engine
from Graphic._layers import UILayer, LitLayer, ChunkedLayer, Layer
from Graphic._sprites import Sprite
from Graphic.gui import UIText, UIButton

# --- CONFIGURATION ---
TEXTURE_PATH = "assets/textures"
OUTPUT_FILE = "saves/level_decor_data.json"


def get_all_textures(path):
    """Recursively scans the folder for .png files"""
    tex_list = []
    if not os.path.exists(path):
        print(f"Warning: Folder '{path}' not found.")
        return tex_list

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".png"):
                full_path = os.path.join(root, file).replace("\\", "/")
                tex_list.append(full_path)
    return sorted(tex_list)


class EditorEngine(Engine):
    def __init__(self, name='Ortensia Level Editor', base_size=(1000, 600)):
        super().__init__(name=name, base_size=base_size)

        # Editor State
        self.textures = get_all_textures(TEXTURE_PATH)
        self.current_tex_index = 0
        self.selected_layer_index = 0
        self.editor_layers = []
        self.cam_speed = 600
        self.mode = "PLACE"  # PLACE, DELETE
        self.show_grid = True
        self.main_camera.y = 400

        # New: Scaling State
        self.current_scale = 1.0

        # UI Setup
        self.ui_layer = UILayer("EditorUI")
        self.add_layer(self.ui_layer)
        self.init_ui()

        # Load previous data
        self.load_if_exists()

    def register_editable_layers(self, layers):
        self.editor_layers = layers

    def init_ui(self):
        help_y = self.base_size[1] - 120
        help_x = self.base_size[0] - 250  # Moved slightly left to fit text

        # Instructions
        self.ui_layer.add_element(UIText(10, help_y, "[WASD] Move Cam | [TAB] Layer | [Scroll] Size", 14))
        self.ui_layer.add_element(UIText(10, help_y + 20, "[t/y] Texture | [M] Mode (Place/Del)", 14))
        self.ui_layer.add_element(UIText(10, help_y + 40, "[K] Save | [L] Load", 14))

        # Status Labels
        self.lbl_info = UIText(help_x, help_y, "Ortensia Editor", size=20, color=(255, 215, 0))
        self.lbl_layer = UIText(help_x, help_y + 20, "Layer: ", size=18)
        self.lbl_mode = UIText(help_x, help_y + 40, "Mode: ", size=18)
        self.lbl_scale = UIText(help_x, help_y + 60, "Scale: ", size=18)  # New Label
        self.lbl_tex = UIText(help_x, help_y + 80, "Texture: ", size=14)
        self.lbl_pos = UIText(help_x, help_y + 100, "x, y: ", size=16)

        self.ui_layer.add_element(self.lbl_info)
        self.ui_layer.add_element(self.lbl_layer)
        self.ui_layer.add_element(self.lbl_mode)
        self.ui_layer.add_element(self.lbl_scale)
        self.ui_layer.add_element(self.lbl_tex)
        self.ui_layer.add_element(self.lbl_pos)

    def update_ui(self):
        # Update texts
        if self.editor_layers:
            curr_l = self.editor_layers[self.selected_layer_index]
            self.lbl_layer.set_text(f"Layer: {curr_l.name} (P: {curr_l.parallax})")

        self.lbl_mode.set_text(f"Mode: {self.mode}")
        self.lbl_scale.set_text(f"Scale: {self.current_scale:.2f}x")  # Show Scale

        if self.textures:
            tex_name = self.textures[self.current_tex_index].split("/")[-1]
            self.lbl_tex.set_text(f"Tex ({self.current_tex_index + 1}/{len(self.textures)}): {tex_name}")
        else:
            self.lbl_tex.set_text("No textures found")

        # Show Mouse World Position
        mx, my = pygame.mouse.get_pos()
        cam = self.main_camera
        self.lbl_pos.set_text(f"Pos: {int(mx + cam.x)}, {int(my + cam.y)}")

    def handle_input(self, event):
        # Mouse Wheel for Scaling
        if event.type == pygame.MOUSEWHEEL:
            self.current_scale += event.y * 0.1
            # Clamp scale (minimum 0.1, maximum 5.0)
            if self.current_scale < 0.1: self.current_scale = 0.1
            if self.current_scale > 5.0: self.current_scale = 5.0

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

            # Optional: Reset Scale with 'R'
            elif event.key == pygame.K_r:
                self.current_scale = 1.0

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left Click
                self.use_tool(pygame.mouse.get_pos())

    def mechaniches(self, dt):
        keys = pygame.key.get_pressed()
        speed = self.cam_speed
        if keys[pygame.K_LSHIFT]: speed *= 2

        if keys[pygame.K_a]: self.main_camera.scroll_x -= speed * dt
        if keys[pygame.K_d]: self.main_camera.scroll_x += speed * dt
        if keys[pygame.K_w]: self.main_camera.scroll_y -= speed * dt
        if keys[pygame.K_s]: self.main_camera.scroll_y += speed * dt

    def use_tool(self, mouse_pos):
        if not self.editor_layers: return

        mx, my = mouse_pos
        layer = self.editor_layers[self.selected_layer_index]
        cam = self.main_camera

        wx = mx + (cam.x * layer.parallax)
        wy = my + (cam.y * layer.parallax)

        if self.mode == "PLACE" and self.textures:
            tex_path = self.textures[self.current_tex_index]

            # Load and SCALE
            temp_surf = pygame.image.load(tex_path)
            orig_w, orig_h = temp_surf.get_size()

            # Apply Scale
            final_w = int(orig_w * self.current_scale)
            final_h = int(orig_h * self.current_scale)

            # Center position based on NEW size
            x = wx - (final_w // 2)
            y = wy - (final_h // 2)

            # Create Sprite with scaled dimensions
            s = Sprite(x, y, final_w, final_h, texture=tex_path, alpha=True)
            s.texture_path = tex_path

            if hasattr(layer, 'add_static'):
                layer.add_static(s)
            else:
                layer.sprites.append(s)

            print(f"Placed {os.path.basename(tex_path)} (x{self.current_scale:.1f})")

        elif self.mode == "DELETE":
            to_remove = None
            candidates = []
            if hasattr(layer, 'sprites'): candidates.extend(layer.sprites)
            if hasattr(layer, 'large_sprites'): candidates.extend(layer.large_sprites)
            if hasattr(layer, 'chunks'):
                for chunk_list in layer.chunks.values():
                    candidates.extend(chunk_list)

            for s in reversed(candidates):
                if s.x < wx < s.x + s.width and s.y < wy < s.y + s.height:
                    to_remove = s
                    break

            if to_remove:
                if hasattr(layer, 'remove_static'):
                    layer.remove_static(to_remove)
                elif to_remove in layer.sprites:
                    layer.sprites.remove(to_remove)
                print("Object removed.")

    def draw_preview(self):
        """Draws scaled preview under mouse"""
        if self.mode == "PLACE" and self.textures:
            mx, my = pygame.mouse.get_pos()
            tex_path = self.textures[self.current_tex_index]

            try:
                # Load
                surf = pygame.image.load(tex_path).convert_alpha()

                # Scale
                w, h = surf.get_size()
                scaled_w = int(w * self.current_scale)
                scaled_h = int(h * self.current_scale)

                surf = pygame.transform.scale(surf, (scaled_w, scaled_h))
                surf.set_alpha(150)

                # Draw
                self.screen.blit(surf, (mx - scaled_w // 2, my - scaled_h // 2))

                # Draw border
                pygame.draw.rect(self.screen, (0, 255, 0),
                                 (mx - scaled_w // 2, my - scaled_h // 2, scaled_w, scaled_h), 1)
            except:
                pass
        elif self.mode == "DELETE":
            mx, my = pygame.mouse.get_pos()
            pygame.draw.circle(self.screen, (255, 0, 0), (mx, my), 10, 2)

    def save_level(self):
        data = []
        print("Saving...")

        for layer in self.editor_layers:
            layer_data = {
                "name": layer.name,
                "parallax": layer.parallax,
                "type": layer.__class__.__name__,
                "sprites": []
            }

            all_sprites = []
            if hasattr(layer, 'sprites'): all_sprites.extend(layer.sprites)
            if hasattr(layer, 'large_sprites'): all_sprites.extend(layer.large_sprites)
            if hasattr(layer, 'chunks'):
                for chunk in layer.chunks.values():
                    all_sprites.extend(chunk)

            for s in all_sprites:
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
        print(f"Level saved to {OUTPUT_FILE}")

    def load_if_exists(self):
        if not os.path.exists(OUTPUT_FILE): return
        print(f"Loading {OUTPUT_FILE}...")
        try:
            with open(OUTPUT_FILE, 'r') as f:
                data = json.load(f)

            for l_data in data:
                target_layer = next((l for l in self.editor_layers if l.name == l_data["name"]), None)
                if not target_layer: continue

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
                        print(f"Error loading sprite: {e}")
            print("Loaded.")
        except Exception as e:
            print(f"Error loading JSON: {e}")

    # --- Overridden Update Loop for Correct Z-Ordering ---
    def update(self):
        dt = self.dt()
        camera = self.cameras[self.camera_id]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            self.handle_input(event)
            self.ui_layer.process_events(event)

        self.mechaniches(dt)
        camera.update()
        self.update_ui()

        # Rendering Order
        self.screen.fill((40, 40, 50))

        # 1. Draw Game Layers EXCEPT UI
        for layer in self.layers:
            if layer != self.ui_layer:
                layer.render(self.screen, camera)

        self.draw_preview()

        self.ui_layer.render(self.screen, camera)

        pygame.display.set_caption(f"{self.name} | FPS: {int(self.clock.get_fps())}")
        pygame.display.flip()


# --- MAIN BLOCK ---
if __name__ == "__main__":
    game = EditorEngine()

    # Define Layers
    bg1 = game.add_create_layer("bg1", 0.001)
    bg2 = game.add_create_layer("bg2", 0.005)
    bg3 = game.add_create_layer("bg3", 0.1)
    bg4 = game.add_create_layer("bg4", 0.3)
    bg5 = game.add_create_layer("bg5", 0.6)

    fg = LitLayer("Foreground", 1.0)
    game.add_layer(fg)

    game.register_editable_layers([bg1, bg2, bg3, bg4, bg5, fg])

    while game.running:
        game.update()