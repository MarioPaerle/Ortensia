import pygame
import numpy as np
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass

PARTICLE_EMISSION_QUALITY = 3
EFFECTS_QUALITY = 2

import pygame
import numpy as np


class PostProcessing:
    _vignette_mask = None
    _vignette_params = None

    @staticmethod
    def bloom(surface: pygame.Surface, threshold=50, scale=0, strength=1.5, quality=EFFECTS_QUALITY):
        if quality <= 0: return

        # Scale: 16x (Fastest), 8x (Medium), 4x (High)
        scale = [0, 32, 24, 16, 8, 1][quality] + scale
        size = surface.get_size()
        small_size = (max(1, size[0] // scale), max(1, size[1] // scale))

        small_surf = pygame.transform.smoothscale(surface, small_size)
        pixels = pygame.surfarray.pixels3d(small_surf)

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
    def general_controls(surface: pygame.Surface, saturation=1.2, tint=(1.0, 1.0, 1.0), brightness=1.0,
                         quality=EFFECTS_QUALITY):
        if quality <= 0: return

        size = surface.get_size()

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

        q_map = [0, 6, 3, 1.5, 1, 1, 1]
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

            PostProcessing._vignette_mask = pygame.transform.smoothscale(raw_mask, size)
            PostProcessing._vignette_params = current_params

        surface.blit(PostProcessing._vignette_mask, (0, 0), special_flags=pygame.BLEND_RGB_MULT)


class ParticleEmitter:
    def __init__(self, color=(0, 255, 255), count=1000, size=1, g=10, sparsity=0.75):
        self.color = color
        self.data = np.zeros((int(count * PARTICLE_EMISSION_QUALITY), 5))  # [x, y, vx, vy, life]
        self.active = np.zeros(int(count * PARTICLE_EMISSION_QUALITY), dtype=bool)
        self.g = g
        self.sparsity = sparsity
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
            pygame.draw.circle(surface, self.color, (px, py), max(1, int(size * self.size)))


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
    def __init__(self, x, y, w, h, color=(255, 255, 255)):
        self.x, self.y, self.width, self.height = x, y, w, h
        self.surface = pygame.Surface((w, h))
        self.surface.fill(color)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy


class SpatialGrid:
    def __init__(self, cell_size=128):
        self.cell_size = cell_size
        self.cells = {}

    def clear(self):
        self.cells.clear()

    def insert(self, sprite):
        # Calculate cell range the sprite occupies
        x_start = int(sprite.frect.left // self.cell_size)
        x_end = int(sprite.frect.right // self.cell_size)
        y_start = int(sprite.frect.top // self.cell_size)
        y_end = int(sprite.frect.bottom // self.cell_size)

        for x in range(x_start, x_end + 1):
            for y in range(y_start, y_end + 1):
                self.cells.setdefault((x, y), []).append(sprite)

    def get_nearby(self, frect):
        x_start = int(frect.left // self.cell_size)
        x_end = int(frect.right // self.cell_size)
        y_start = int(frect.top // self.cell_size)
        y_end = int(frect.bottom // self.cell_size)

        nearby = []
        for x in range(x_start, x_end + 1):
            for y in range(y_start, y_end + 1):
                if (x, y) in self.cells:
                    nearby.extend(self.cells[(x, y)])
        return nearby


class SolidSprite(Sprite):
    def __init__(self, x, y, w, h, color=(255, 255, 255)):
        super().__init__(x, y, w, h, color)
        self.frect = pygame.FRect(x, y, w, h)

    def move(self, dx, dy, grid):
        # Horizontal Move & Resolve
        self.frect.x += dx
        for other in grid.get_nearby(self.frect):
            if other is not self and self.frect.colliderect(other.frect):
                if dx > 0: self.frect.right = other.frect.left
                if dx < 0: self.frect.left = other.frect.right

        # Vertical Move & Resolve
        self.frect.y += dy
        for other in grid.get_nearby(self.frect):
            if other is not self and self.frect.colliderect(other.frect):
                if dy > 0: self.frect.bottom = other.frect.top
                if dy < 0: self.frect.top = other.frect.bottom

        # Sync back to base Sprite properties for rendering
        self.x, self.y = self.frect.x, self.frect.y


class Layer:
    def __init__(self, name: str, parallax: float = 1.0):
        self.name = name
        self.parallax = parallax
        self.sprites: List[Sprite] = []
        self.effects: List[Tuple[Any, tuple]] = []
        self.visible = True

    def add_effect(self, effect_fn, *args):
        self.effects.append((effect_fn, args))

    def render(self, screen: pygame.Surface, camera: Camera, emitters=None):
        if not self.visible: return

        if self.parallax == 0.0 and not self.effects:
            for s in self.sprites:
                screen.blit(s.surface, (int(s.x), int(s.y)))
        else:
            layer_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            layer_surf.fill((0, 0, 0, 0))
            cx, cy = camera.x * self.parallax, camera.y * self.parallax

            for s in self.sprites:
                sx, sy = s.x - cx, s.y - cy
                if -s.width < sx < screen.get_width() and -s.height < sy < screen.get_height():
                    layer_surf.blit(s.surface, (int(sx), int(sy)))

            if emitters is not None:
                for emitter in emitters:
                    emitter.draw(layer_surf, camera)

            for effect_fn, args in self.effects:
                effect_fn(layer_surf, *args)

            screen.blit(layer_surf, (0, 0))


class Game:
    def __init__(self, w=200, h=300, title="Ortensia Engine"):
        pygame.init()
        self.screen = pygame.display.set_mode((w, h), pygame.SCALED | pygame.DOUBLEBUF)
        pygame.display.set_caption(title)

        self.clock = pygame.time.Clock()
        self.camera = Camera(width=w, height=h)
        self.layers: List[Layer] = []
        self.particle_emitters = []
        self.particle_layer_idx = -1
        self.solids = []
        self.grid = SpatialGrid(cell_size=160)
        self.running = True
        self.max_fps = 600
        self.game_div = 1000.0
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

            self.grid.clear()
            for obj in self.solids:
                self.grid.insert(obj)

            # Update Logic
            update_callback(self, dt)
            self.camera.update()

            for emitter in self.particle_emitters:
                emitter.update(dt)

            self.screen.fill((20, 20, 30))

            for i, layer in enumerate(self.layers):
                if self.particle_layer_idx != -1 and self.particle_layer_idx == i:
                    layer.render(self.screen, self.camera, emitters=self.particle_emitters)
                else:
                    layer.render(self.screen, self.camera)

            if self.particle_layer_idx == -1:
                for emitter in self.particle_emitters:
                    emitter.draw(self.screen, self.camera)

            fps = self.clock.get_fps()
            pygame.display.set_caption(f"Ortensia | FPS: {int(fps)}")

            pygame.display.flip()


if __name__ == "__main__":
    def s(x):
        return int(x * 0.5)


    game = Game(s(1000), s(600))
    bg2 = game.add_layer("Background2", 0.2)
    bg = game.add_layer("Background", 0.5)
    fg = game.add_layer("Foreground", 1.0)

    # fg.add_effect(PostProcessing.bloom, 20, 8, 3.0)
    fg.add_effect(PostProcessing.bloom, 60, 0, 1.0)
    # fg.add_effect(PostProcessing.blur, 3)

    player = SolidSprite(s(400), s(300), s(40), s(40), (255, 255, 255))
    fg.sprites.append(player)
    game.solids.append(player)
    game.camera.target = player
    emitter1 = ParticleEmitter(size=0.75, sparsity=0.2, g=40)
    game.particle_emitters.append(emitter1)
    game.particle_layer_idx = 2

    for i in range(15):
        bg.sprites.append(Sprite(i * s(250), s(200), s(100), s(400), (40, 40, 80)))

    for i in range(10):
        bg2.sprites.append(Sprite(i * s(400), s(400), s(100), s(200), (70, 70, 100)))

    for i in range(15):
        if i % 3 == 0:
            wall = SolidSprite(i * s(400), s(450), s(60), s(150), (255, 20, 50))
            fg.sprites.append(wall)
            game.solids.append(wall)  # Register as a solid obstacle
        else:
            wall = Sprite(i * s(400), s(450), s(60), s(150), (30, 70, 40))
            fg.sprites.append(wall)


    def my_update(game_inst, dt):
        keys = pygame.key.get_pressed()
        speed = 500 * dt * 1

        dx, dy = 0, 0
        if keys[pygame.K_LEFT]:  dx -= speed
        if keys[pygame.K_RIGHT]: dx += speed
        if keys[pygame.K_UP]:    dy -= speed
        if keys[pygame.K_DOWN]:  dy += speed

        player.move(dx, dy, game_inst.grid)

        emitter1.emit(player.x + 10, player.y + 10)


    game.run(my_update)
