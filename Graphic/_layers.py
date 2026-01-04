from typing import Optional, Tuple, Any
from dataclasses import dataclass
import random
from Graphic._sprites import *
from Graphic.functions import scale_color
import math


@dataclass
class Camera:
    x: float = 0.0
    y: float = 0.0
    width: int = 800
    height: int = 600
    target: Optional[Any] = None
    smooth: float = 0.1

    shake_intensity: float = 0.0
    shake_decay: float = 0.9  # How fast the shake stops (0.9 = 10% per frame)
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    def apply_shake(self, intensity: float):
        self.shake_intensity = intensity

    def update(self):
        prev_x, prev_y = self.x, self.y
        if self.target:
            tx = self.target.x - self.width // 2
            ty = self.target.y - self.height // 2
            self.x += (tx - self.x) * self.smooth
            self.y += (ty - self.y) * self.smooth

        if self.shake_intensity > 0.1:
            self.x += random.uniform(-self.shake_intensity, self.shake_intensity)
            self.y += random.uniform(-self.shake_intensity, self.shake_intensity)
            self.shake_intensity *= self.shake_decay
        else:
            self.shake_intensity = 0

        self.velocity_x = self.x - prev_x
        self.velocity_y = self.y - prev_y


class Layer:
    def __init__(self, name: str, parallax: float = 1.0, **kwargs):
        self.name = name
        self.parallax = parallax
        self.sprites: List[Sprite] = []
        self.effects: List[Tuple[Any, tuple]] = []
        self.visible = True

        self._cached_surf = None

    def add_effect(self, effect_fn, *args):
        self.effects.append((effect_fn, args))

    def _get_layer_surf(self, size: Tuple[int, int]) -> pygame.Surface:
        if self._cached_surf is None or self._cached_surf.get_size() != size:
            self._cached_surf = pygame.Surface(size, pygame.SRCALPHA)
        return self._cached_surf

    def render(self, screen: pygame.Surface, camera: Camera, emitters=None):
        if not self.visible: return

        if self.parallax == 0.0 and not self.effects and not emitters:
            for s in self.sprites:
                screen.blit(s.surface, (int(s.x), int(s.y)))
            return

        screen_size = screen.get_size()
        layer_surf = self._get_layer_surf(screen_size)

        layer_surf.fill((0, 0, 0, 0))

        cx, cy = camera.x * self.parallax, camera.y * self.parallax
        screen_w, screen_h = screen_size

        for s in self.sprites:
            sx, sy = s.x - cx, s.y - cy

            if -s.width < sx < screen_w and -s.height < sy < screen_h:
                if hasattr(s, 'update'):
                    s.update()
                layer_surf.blit(s.surface, (int(sx), int(sy)))

        if emitters is not None:
            for emitter in emitters:
                emitter.draw(layer_surf, camera)

        for effect_fn, args in self.effects:
            effect_fn(layer_surf, *args)

        screen.blit(layer_surf, (0, 0))


class ChunkedLayer:
    def __init__(self, name: str, parallax: float = 1.0, chunk_size: int = 500):
        self.name = name
        self.parallax = parallax
        self.visible = True
        self.chunk_size = chunk_size

        self.chunks = {}
        self.sprites = []

        self.effects: List[Tuple[Any, tuple]] = []
        self._cached_surf = None

    def add_effect(self, effect_fn, *args):
        self.effects.append((effect_fn, args))

    def add_static(self, sprite):
        cx = int(sprite.x // self.chunk_size)
        cy = int(sprite.y // self.chunk_size)

        if (cx, cy) not in self.chunks:
            self.chunks[(cx, cy)] = []
        self.chunks[(cx, cy)].append(sprite)

    def add_dynamic(self, sprite):
        self.sprites.append(sprite)

    def remove_static(self, sprite):
        cx = int(sprite.x // self.chunk_size)
        cy = int(sprite.y // self.chunk_size)

        if (cx, cy) in self.chunks:
            if sprite in self.chunks[(cx, cy)]:
                self.chunks[(cx, cy)].remove(sprite)
                return True
        return False

    def _get_layer_surf(self, size: Tuple[int, int]) -> pygame.Surface:
        if self._cached_surf is None or self._cached_surf.get_size() != size:
            self._cached_surf = pygame.Surface(size, pygame.SRCALPHA)
        return self._cached_surf

    def render(self, screen: pygame.Surface, camera: Camera, emitters=None):
        if not self.visible: return

        screen_size = screen.get_size()
        screen_w, screen_h = screen_size

        cx, cy = camera.x * self.parallax, camera.y * self.parallax

        layer_surf = self._get_layer_surf(screen_size)
        layer_surf.fill((0, 0, 0, 0))

        start_chunk_x = int(cx // self.chunk_size)
        end_chunk_x = int((cx + screen_w) // self.chunk_size) + 1

        start_chunk_y = int(cy // self.chunk_size)
        end_chunk_y = int((cy + screen_h) // self.chunk_size) + 1

        for x in range(start_chunk_x - 1, end_chunk_x + 1):
            for y in range(start_chunk_y - 1, end_chunk_y + 1):
                chunk_key = (x, y)
                if chunk_key in self.chunks:
                    for s in self.chunks[chunk_key]:
                        sx = s.x - cx
                        sy = s.y - cy
                        if -s.width < sx < screen_w and -s.height < sy < screen_h:
                            layer_surf.blit(s.surface, (int(sx), int(sy)))

        for s in self.sprites:
            sx = s.x - cx
            sy = s.y - cy
            if -s.width < sx < screen_w and -s.height < sy < screen_h:
                if hasattr(s, 'update'):
                    s.update()
                layer_surf.blit(s.surface, (int(sx), int(sy)))

        if emitters is not None:
            for emitter in emitters:
                emitter.draw(layer_surf, camera)

        for effect_fn, args in self.effects:
            effect_fn(layer_surf, *args)

        screen.blit(layer_surf, (0, 0))


class LightSource:
    _cache = {}

    def __init__(self, x, y, radius=100, color=(255, 255, 200), brightness=1.0,
                 diffusion_style='misty', diffusion_quality=2, falloff=1, diffusion_function=lambda i, fall: 1-fall**i, steps=10):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.brightness = brightness
        self.diffusion_style = diffusion_style
        self.diffusion_quality = diffusion_quality
        self.falloff = falloff
        self.steps = steps
        self.diffusion_function = diffusion_function
        self.surface = self._get_surface()


    def _get_surface(self):
        key = (self.radius, self.color, self.brightness, self.diffusion_style, self.diffusion_quality, self.falloff)

        if key in LightSource._cache:
            return LightSource._cache[key]

        size = self.radius * 2
        surf = pygame.Surface((size, size), pygame.SRCALPHA)

        center = (self.radius, self.radius)
        steps = self.steps
        for i in range(1, steps):
            intensity = self.diffusion_function(i, self.falloff)
            current_radius = int(self.radius - i*steps/self.radius)
            if intensity <= 0: continue

            draw_color = (*scale_color(self.color, intensity), 255)
            pygame.draw.circle(surf, draw_color, center, current_radius)

        if self.diffusion_quality > 0 and self.diffusion_style != 'sharp':
            surf = self._apply_diffusion(surf)

        LightSource._cache[key] = surf
        return surf

    def _apply_diffusion(self, surface):
        w, h = surface.get_size()

        # Stronger Scaling for 'misty'
        scale_divisor = 6 if self.diffusion_style == 'misty' else 4
        if self.diffusion_quality == 1: scale_divisor = 8  # Lower quality = more blur/blocky

        small_w, small_h = max(1, w // scale_divisor), max(1, h // scale_divisor)

        # Multi-pass blur for High Quality
        # We blur it, then blur the blurred version to remove "box" artifacts
        passes = 2 if self.diffusion_quality >= 2 else 1

        temp_surf = surface
        for _ in range(passes):
            # Downscale
            small = pygame.transform.smoothscale(temp_surf, (small_w, small_h))
            # Upscale
            temp_surf = pygame.transform.smoothscale(small, (w, h))

        return temp_surf


class LitLayer(ChunkedLayer):
    def __init__(self, name, parallax=1, ambient_color=(20, 20, 30)):
        super().__init__(name, parallax)
        self.lights = []
        self.ambient_color = ambient_color

    def add_light(self, light: LightSource):
        self.lights.append(light)

    def render(self, screen, camera, emitters=None):
        if not self.visible: return

        screen_size = screen.get_size()
        w_screen, h_screen = screen_size
        cx, cy = camera.x * self.parallax, camera.y * self.parallax

        layer_surf = self._get_layer_surf(screen_size)
        layer_surf.fill((0, 0, 0, 0))

        render_candidates = []

        start_chunk_x = int(cx // self.chunk_size)
        end_chunk_x = int((cx + w_screen) // self.chunk_size) + 1
        start_chunk_y = int(cy // self.chunk_size)
        end_chunk_y = int((cy + h_screen) // self.chunk_size) + 1

        for x in range(start_chunk_x - 1, end_chunk_x + 1):
            for y in range(start_chunk_y - 1, end_chunk_y + 1):
                if (x, y) in self.chunks:
                    render_candidates.extend(self.chunks[(x, y)])

        render_candidates.extend(self.sprites)

        visible_lights = []
        for l in self.lights:
            lx, ly = l.x - cx, l.y - cy
            if (lx + l.radius > 0 and lx - l.radius < w_screen and
                    ly + l.radius > 0 and ly - l.radius < h_screen):
                visible_lights.append(l)

        for s in render_candidates:
            sx = s.x - cx
            sy = s.y - cy

            if not (-s.width < sx < w_screen and -s.height < sy < h_screen):
                continue

            w, h = s.surface.get_size()

            light_map = pygame.Surface((w, h), 0)
            light_map.fill(self.ambient_color)
            if hasattr(s, 'update'):
                s.update()
            s_center_x = s.x + w / 2
            s_center_y = s.y + h / 2

            for l in visible_lights:
                dist_x = abs(l.x - s_center_x)
                dist_y = abs(l.y - s_center_y)

                if dist_x < (w / 2 + l.radius) and dist_y < (h / 2 + l.radius):
                    lx_local = (l.x - l.radius) - s.x
                    ly_local = (l.y - l.radius) - s.y

                    light_map.blit(l.surface, (lx_local, ly_local), special_flags=pygame.BLEND_ADD)

            final_sprite = s.surface.copy()
            final_sprite.blit(light_map, (0, 0), special_flags=pygame.BLEND_MULT)

            layer_surf.blit(final_sprite, (int(sx), int(sy)))

        if emitters:
            for em in emitters:
                em.draw(layer_surf, camera)

        for effect_fn, args in self.effects:
            effect_fn(layer_surf, *args)

        screen.blit(layer_surf, (0, 0))


class TileMap:
    def __init__(self, game, layer: ChunkedLayer, tile_size=40, texture=None):
        self.game = game
        self.layer = layer
        self.tile_size = tile_size
        self.texture = texture
        self.data = {}

    def get_grid_pos(self, screen_x, screen_y):
        cam_x = self.game.main_camera.x * self.layer.parallax
        cam_y = self.game.main_camera.y * self.layer.parallax

        world_x = screen_x + cam_x
        world_y = screen_y + cam_y

        gx = int(world_x // self.tile_size)
        gy = int(world_y // self.tile_size)
        return gx, gy

    def place_tile(self, screen_x, screen_y, color=(200, 200, 200)):
        gx, gy = self.get_grid_pos(screen_x, screen_y)
        if (gx, gy) in self.data:
            return

        world_x = gx * self.tile_size
        world_y = gy * self.tile_size

        tile_sprite = SolidSprite(world_x, world_y, self.tile_size, self.tile_size, color, texture=self.texture, alpha=True)

        self.layer.add_static(tile_sprite)
        self.game.solids.append(tile_sprite)
        self.data[(gx, gy)] = tile_sprite

    def remove_tile(self, screen_x, screen_y):
        gx, gy = self.get_grid_pos(screen_x, screen_y)

        if (gx, gy) in self.data:
            sprite = self.data[(gx, gy)]

            self.layer.remove_static(sprite)
            if sprite in self.game.solids:
                self.game.solids.remove(sprite)

            del self.data[(gx, gy)]


class BlockMap:
    def __init__(self, game, layer: ChunkedLayer, tile_size=40, texture=None):
        self.game = game
        self.layer = layer
        self.tile_size = tile_size
        self.texture = texture
        self.data = {}

    def get_grid_pos(self, screen_x, screen_y):
        cam_x = self.game.main_camera.x * self.layer.parallax
        cam_y = self.game.main_camera.y * self.layer.parallax

        world_x = screen_x + cam_x
        world_y = screen_y + cam_y

        gx = int(world_x // self.tile_size)
        gy = int(world_y // self.tile_size)
        return gx, gy

    def place_tile(self, screen_x, screen_y, block):
        gx, gy = self.get_grid_pos(screen_x, screen_y)
        if (gx, gy) in self.data:
            return

        world_x = gx * self.tile_size
        world_y = gy * self.tile_size

        tile_sprite = Block(world_x, world_y, self.tile_size, self.tile_size, 0, texture=self.texture, alpha=True)

        self.layer.add_static(tile_sprite)
        self.game.solids.append(tile_sprite)
        self.data[(gx, gy)] = tile_sprite

    def remove_tile(self, screen_x, screen_y):
        gx, gy = self.get_grid_pos(screen_x, screen_y)

        if (gx, gy) in self.data:
            sprite = self.data[(gx, gy)]

            self.layer.remove_static(sprite)
            if sprite in self.game.solids:
                self.game.solids.remove(sprite)

            del self.data[(gx, gy)]