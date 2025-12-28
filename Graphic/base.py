import pygame
import numpy as np
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass

PARTICLE_EMISSION_QUALITY = 1
EFFECTS_QUALITY = 3

import pygame
import numpy as np


class PostProcessing:
    _vignette_mask = None
    _vignette_params = None  # (size, intensity, quality)

    @staticmethod
    def bloom(surface: pygame.Surface, threshold=50, scale=0, strength=1.5, quality=EFFECTS_QUALITY):
        if quality <= 0: return

        # Scale: 16x (Fastest), 8x (Medium), 4x (High)
        scale = [0, 32, 24, 16, 8][quality] + scale
        size = surface.get_size()
        small_size = (max(1, size[0] // scale), max(1, size[1] // scale))

        # 1. Downsample
        small_surf = pygame.transform.smoothscale(surface, small_size)
        pixels = pygame.surfarray.pixels3d(small_surf)

        # 2. Thresholding
        luma = np.dot(pixels, [0.299, 0.587, 0.114])
        mask = luma < threshold

        processed = (pixels.astype(np.float16) * strength)
        processed[mask] = 0
        np.clip(processed, 0, 255, out=processed)

        pygame.surfarray.blit_array(small_surf, processed.astype(np.uint8))
        del pixels

        # 3. Upscale & Add
        final_glow = pygame.transform.smoothscale(small_surf, size)
        surface.blit(final_glow, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    @staticmethod
    def general_controls(surface: pygame.Surface, saturation=1.2, tint=(1.0, 1.0, 1.0), brightness=1.0, quality=EFFECTS_QUALITY):
        if quality <= 0: return

        size = surface.get_size()

        # Quality 1 & 2 process at lower resolutions to save CPU
        # Quality 3 processes full resolution
        if quality < 3:
            div = 12 if quality == 1 else (8 if quality == 2 else 4)
            work_size = (size[0] // div, size[1] // div)
            work_surf = pygame.transform.smoothscale(surface, work_size)
            pixels = pygame.surfarray.pixels3d(work_surf)
        else:
            work_surf = surface
            pixels = pygame.surfarray.pixels3d(surface)

        arr = pixels.astype(np.float16)
        arr *= (np.array(tint) * brightness)

        luma = (arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114)[..., None]
        arr = luma + (arr - luma) * saturation

        np.clip(arr, 0, 255, out=arr)
        pixels[...] = arr.astype(np.uint8)
        del pixels

        if quality < 3:
            pygame.transform.smoothscale(work_surf, size, surface)

    @staticmethod
    def blur(surface: pygame.Surface, amount=2, quality=EFFECTS_QUALITY):
        if quality <= 0 or amount <= 0: return
        size = surface.get_size()

        # Q1: Tiny internal buffer (Fast/Blurry) | Q3: High-res buffer
        q_map = [0, 6, 3, 1.5]
        shrink = 1 / (amount * q_map[quality])

        small_size = (max(1, int(size[0] * shrink)), max(1, int(size[1] * shrink)))
        temp = pygame.transform.smoothscale(surface, small_size)
        pygame.transform.smoothscale(temp, size, surface)

    @staticmethod
    def vignette(surface: pygame.Surface, intensity=0.5, quality=EFFECTS_QUALITY):
        if quality <= 0: return

        size = surface.get_size()
        current_params = (size, intensity, quality)

        if PostProcessing._vignette_mask is None or PostProcessing._vignette_params != current_params:
            # Q1 uses a 1/4 size mask stretched up to avoid heavy pixel math
            mask_res = [0, size[0] // 4, size[0] // 2, size[0]][quality]
            aspect = size[1] / size[0]
            v_size = (mask_res, int(mask_res * aspect))

            X, Y = np.ogrid[:v_size[0], :v_size[1]]
            center_x, center_y = v_size[0] / 2, v_size[1] / 2
            dist_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
            max_dist_sq = center_x ** 2 + center_y ** 2

            mask = 1.0 - (dist_sq / max_dist_sq) * intensity
            mask = np.clip(mask * 255, 0, 255).astype(np.uint8)

            v_arr = np.stack([mask] * 3, axis=2)
            raw_mask = pygame.surfarray.make_surface(v_arr)
            # Stretching the mask back up to screen size
            PostProcessing._vignette_mask = pygame.transform.smoothscale(raw_mask, size)
            PostProcessing._vignette_params = current_params

        surface.blit(PostProcessing._vignette_mask, (0, 0), special_flags=pygame.BLEND_RGB_MULT)


class ParticleEmitter:
    def __init__(self, color=(0, 255, 255), count=1000, size=1):
        self.color = color
        self.data = np.zeros((int(count * PARTICLE_EMISSION_QUALITY), 5))  # [x, y, vx, vy, life]
        self.active = np.zeros(int(count * PARTICLE_EMISSION_QUALITY), dtype=bool)
        self.g = 10
        self.sparsity = 0.75
        self.size = size

    def emit(self, x, y, amount=5):
        inactive = np.where(~self.active)[0]
        if len(inactive) == 0: return

        idx = inactive[:amount]
        actual_count = len(idx)

        jitter = np.random.uniform(-3, 3, (actual_count, 2))
        self.data[idx, 0:2] = [x, y] + jitter

        self.data[idx, 2:4] = np.random.uniform(-150, 150, (actual_count, 2)) * self.sparsity

        self.data[idx, 4] = np.random.uniform(0.5, 1.5, actual_count)

        self.active[idx] = True

    def update(self, dt):
        if not np.any(self.active): return

        self.data[self.active, 3] += self.g * dt
        self.data[self.active, 0:2] += self.data[self.active, 2:4] * dt

        # 3. Decay life
        self.data[self.active, 4] -= dt
        self.active &= (self.data[:, 4] > 0)

    def draw(self, surface, camera):
        indices = np.where(self.active)[0]
        for i in indices:
            px = int(self.data[i, 0] - camera.x)
            py = int(self.data[i, 1] - camera.y)
            size = max(1, int(self.data[i, 4] * 5))
            pygame.draw.circle(surface, self.color, (px, py), int(size*self.size))


@dataclass
class Camera:
    x: float = 0.0
    y: float = 0.0
    width: int = 800
    height: int = 600
    target: Optional[Any] = None
    smooth: float = 0.1

    def update(self):
        if self.target:
            tx = self.target.x - self.width // 2
            ty = self.target.y - self.height // 2
            self.x += (tx - self.x) * self.smooth
            self.y += (ty - self.y) * self.smooth


class Sprite:
    def __init__(self, x, y, w, h, color):
        self.x, self.y, self.width, self.height = x, y, w, h
        self.surface = pygame.Surface((w, h))
        self.surface.fill(color)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy


class Layer:
    def __init__(self, name: str, parallax: float = 1.0):
        self.name = name
        self.parallax = parallax
        self.sprites: List[Sprite] = []
        self.effects: List[Tuple[Any, tuple]] = []
        self.visible = True

    def add_effect(self, effect_fn, *args):
        self.effects.append((effect_fn, args))

    def render(self, screen: pygame.Surface, camera: Camera):
        if not self.visible: return

        # Temp surface for the layer
        layer_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        cx, cy = camera.x * self.parallax, camera.y * self.parallax

        for s in self.sprites:
            sx, sy = s.x - cx, s.y - cy
            if -s.width < sx < screen.get_width() and -s.height < sy < screen.get_height():
                layer_surf.blit(s.surface, (int(sx), int(sy)))
                pass

        for effect_fn, args in self.effects:
            effect_fn(layer_surf, *args)

        screen.blit(layer_surf, (0, 0))


# --- THE MAIN GAME CLASS ---

class Game:
    def __init__(self, w=200, h=300, title="Ortensia Engine"):
        pygame.init()
        self.screen = pygame.display.set_mode((w, h), pygame.SCALED | pygame.DOUBLEBUF)
        pygame.display.set_caption(title)

        self.clock = pygame.time.Clock()
        self.camera = Camera(width=w, height=h)
        self.layers: List[Layer] = []
        self.particle_emitters = []
        self.running = True
        self.max_fps = 600
        self.game_div = 1000.0  # x : fps = 1000 : 60
        self.scale = 1

    def add_layer(self, name, parallax=1.0) -> Layer:
        """Helper to create and track a layer."""
        l = Layer(name, parallax)
        self.layers.append(l)
        return l

    def run(self, update_callback):
        while self.running:
            dt = self.clock.tick(self.max_fps) / self.game_div

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # 1. Update Logic
            update_callback(self, dt)
            self.camera.update()
            for emitter in self.particle_emitters:
                emitter.update(dt)

            # 2. Render
            self.screen.fill((20, 20, 30))

            for layer in self.layers:
                layer.render(self.screen, self.camera)

            for emitter in self.particle_emitters:
                emitter.draw(self.screen, self.camera)

            pygame.display.set_caption(f"Ortensia | FPS: {int(self.clock.get_fps())}")
            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    def s(x):
        return int(x*0.5)
    # Initialize game
    game = Game(s(1000), s(600))

    # Setup Layers (This was causing your error - fixed!)
    bg = game.add_layer("Background", 0.5)
    fg = game.add_layer("Foreground", 1.0)

    # Add player sprite
    player = Sprite(s(400), s(300), s(40), s(40), (255, 255, 255))
    fg.sprites.append(player)
    game.camera.target = player

    # Add some background pillars
    for i in range(15):
        bg.sprites.append(Sprite(i * s(250), s(200), s(100), s(400), (40, 40, 80)))
        fg.sprites.append(Sprite(i * s(400), s(450), s(60), s(150), (255, 0, 150)))  # Neon Pink

    fg.add_effect(PostProcessing.bloom, 0, 80)
    fg.add_effect(PostProcessing.vignette, 0.9)
    fg.add_effect(PostProcessing.blur, 1)
    emitter1 = ParticleEmitter(color=(0, 255, 255), size=s(1))
    emitter2 = ParticleEmitter(color=(150, 150, 255), count=100)
    game.particle_emitters.append(emitter1)
    game.particle_emitters.append(emitter2)


    def my_update(game_inst, dt):
        keys = pygame.key.get_pressed()
        speed = 500 * dt

        if keys[pygame.K_LEFT]:  player.move(-speed, 0)
        if keys[pygame.K_RIGHT]: player.move(speed, 0)
        if keys[pygame.K_UP]:    player.move(0, -speed)
        if keys[pygame.K_DOWN]:  player.move(0, speed)

        emitter1.emit(player.x + 20, player.y + 20, 3)
        # emitter2.emit(player.x + 20, player.y + 20, 3)


    game.run(my_update)
