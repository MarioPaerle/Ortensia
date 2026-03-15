from typing import Optional, Tuple, Any
from dataclasses import dataclass
import random
from Graphic._sprites import *
from Graphic.functions import scale_color
import math
from Graphic.functions import flag
from Graphic.gui import *
import json


@dataclass
class Camera:
    x: float = 0.0
    y: float = 0.0
    width: int = 800
    height: int = 600
    zoom: float = 1.0
    target_zoom: float = 1.0

    scroll_x: float = 0.0
    scroll_y: float = 0.0

    target: Optional[Any] = None
    smooth: float = 0.1
    zoom_smooth: float = 0.1

    shake_intensity: float = 0.0
    shake_decay: float = 0.9
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    target_delta_x = 0.0
    target_delta_y = -100.0

    def __post_init__(self):
        self.scroll_x = self.x + self.width // 2
        self.scroll_y = self.y + self.height // 2

    def apply_zoom(self, amount):
        self.target_zoom = max(0.1, min(5.0, self.target_zoom + amount))

    def snap_to_target(self):
        if self.target:
            t_w = getattr(self.target, 'width', 0)
            t_h = getattr(self.target, 'height', 0)
            self.scroll_x = self.target.x + t_w / 2
            self.scroll_y = self.target.y + t_h / 2
            self.update()

    def update(self):
        diff = self.target_zoom - self.zoom
        if abs(diff) < 0.001:
            self.zoom = self.target_zoom
        else:
            self.zoom += diff * self.zoom_smooth

        view_w = self.width / self.zoom
        view_h = self.height / self.zoom

        if self.target:
            t_w = getattr(self.target, 'width', 0)
            t_h = getattr(self.target, 'height', 0)

            target_center_x = self.target.x + t_w / 2 + self.target_delta_x
            target_center_y = self.target.y + t_h / 2 + self.target_delta_y

            self.scroll_x += (target_center_x - self.scroll_x) * self.smooth
            self.scroll_y += (target_center_y - self.scroll_y) * self.smooth

        shake_x, shake_y = 0, 0
        if self.shake_intensity > 0.1:
            shake_x = random.uniform(-self.shake_intensity, self.shake_intensity)
            shake_y = random.uniform(-self.shake_intensity, self.shake_intensity)
            self.shake_intensity *= self.shake_decay
        else:
            self.shake_intensity = 0

        prev_x, prev_y = self.x, self.y

        self.x = (self.scroll_x - view_w // 2) + shake_x
        self.y = (self.scroll_y - view_h // 2) + shake_y

        self.velocity_x = self.x - prev_x
        self.velocity_y = self.y - prev_y


class Layer:
    def __init__(self, name: str, parallax: float = 1.0, realized_parallax=None, **kwargs):
        self.name = name
        self.parallax = parallax
        self.realized_parallax = realized_parallax if realized_parallax is not None else self.parallax
        self.sprites: List[Sprite] = []
        self.effects: List[Tuple[Any, tuple]] = []
        self.emitters = []
        self.visible = True

        self._cached_surf = None

    def update(self, dt):
        pass

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
                emitter.draw(layer_surf, camera, self.parallax)
                emitter.update()

        for effect_fn, args in self.effects:
            effect_fn(layer_surf, *args)

        screen.blit(layer_surf, (0, 0))

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_cached_surf' in state:
            del state['_cached_surf']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cached_surf = None


class BakedLayer(Layer):

    def __init__(self, name: str, parallax: float = 1.0, width: int = 5000, height: int = 2000, realized_parallax=None,
                 **kwargs):
        super().__init__(name, parallax, realized_parallax)
        self.world_width = width
        self.world_height = height
        self.baked_surface = None
        self.has_baked = False

        # We keep sprites list only until we bake
        self.sprites = []

    def add_static(self, sprite):
        self.sprites.append(sprite)

    def add_dynamic(self, sprite):
        flag("No Dynamic add in BakedLayer", 2)
        return self.add_static(sprite)

    def bake(self):
        # 1. Create the massive canvas
        # flag(f"Baking layer '{self.name}' ({self.world_width}x{self.world_height})...")
        self.baked_surface = pygame.Surface((self.world_width, self.world_height), pygame.SRCALPHA)

        # 2. Draw everything onto it once
        # We sort by Y or creation order if needed, assuming painter's algorithm here
        for s in self.sprites:
            # We assume sprite positions are world coordinates relative to this layer
            # If your sprites have negative coordinates, you might need an offset here
            self.baked_surface.blit(s.surface, (int(s.x), int(s.y)))

        # 3. Clear memory
        self.sprites.clear()
        self.has_baked = True
        # flag(f"Layer '{self.name}' baked successfully.")

    def render(self, screen: pygame.Surface, camera: Camera, emitters=None):
        if not self.visible or not self.has_baked: return

        # Calculate where the camera is looking relative to this parallax layer
        cx = camera.x * self.parallax
        cy = camera.y * self.parallax

        screen_w, screen_h = screen.get_size()

        # Handle Zoom
        zoom = camera.zoom
        if abs(zoom - camera.target_zoom) < 0.001:
            zoom = camera.target_zoom

        view_w = int(screen_w / zoom)
        view_h = int(screen_h / zoom)

        # Optimization: Don't use subsurface if we are going out of bounds
        # Instead, we calculate the overlap and blit directly

        # Source rectangle (What part of the big image do we want?)
        src_rect = pygame.Rect(cx, cy, view_w, view_h)

        # Screen destination (Where does it go?)
        dest_pos = (0, 0)

        # Simple bounds handling:
        # If the camera looks beyond the baked surface, pygame clips it automatically,
        # but we need to calculate the negative offset if cx < 0

        draw_x = 0
        draw_y = 0

        # If using zoom, we need an intermediate surface or just scale the final result
        # Since this is a specialized fast layer, let's just grab the subsection

        try:
            sub = self.baked_surface.subsurface(src_rect.clip(self.baked_surface.get_rect()))

            # If we clipped, we need to adjust position on screen
            # (This logic handles edges of the world correctly)
            blit_x = 0
            blit_y = 0
            if cx < 0: blit_x = -cx
            if cy < 0: blit_y = -cy

            # Draw the baked chunk
            if zoom == 1.0:
                screen.blit(sub, (blit_x, blit_y))
            else:
                scaled = pygame.transform.scale(sub, (int(sub.get_width() * zoom), int(sub.get_height() * zoom)))
                screen.blit(scaled, (int(blit_x * zoom), int(blit_y * zoom)))

        except ValueError:
            # This happens if the rect is completely outside the surface
            pass

        # Draw dynamic emitters on top if needed
        if emitters:
            for emitter in emitters:
                emitter.draw(screen, camera, self.parallax)


class ParticleLayer:
    def __init__(self, name: str, parallax: float = 1.0, realized_parallax=None, **kwargs):
        self.name = name
        self.parallax = parallax
        self.realized_parallax = realized_parallax if realized_parallax is not None else self.parallax
        self.sprites: List[Sprite] = []
        self.effects: List[Tuple[Any, tuple]] = []
        self.emitters = []
        self.visible = True

        self._cached_surf = None

    def update(self, dt):
        pass

    def add_effect(self, effect_fn, *args):
        self.effects.append((effect_fn, args))

    def _get_layer_surf(self, size: Tuple[int, int]) -> pygame.Surface:
        if self._cached_surf is None or self._cached_surf.get_size() != size:
            self._cached_surf = pygame.Surface(size, pygame.SRCALPHA)
        return self._cached_surf

    def render(self, screen: pygame.Surface, camera: Camera, emitters=None):
        if not self.visible: return

        screen_size = screen.get_size()
        layer_surf = self._get_layer_surf(screen_size)

        layer_surf.fill((0, 0, 0, 0))

        for emitter in self.emitters:
            emitter.draw(layer_surf, camera, self.parallax)

        for effect_fn, args in self.effects:
            effect_fn(layer_surf, *args)

        screen.blit(layer_surf, (0, 0))

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_cached_surf' in state:
            del state['_cached_surf']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cached_surf = None


class ChunkedLayer:
    def __init__(self, name: str, parallax: float = 1.0, chunk_size: int = 15, realized_parallax=None):
        self.name = name
        self.parallax = parallax
        self.realized_parallax = parallax if realized_parallax is None else realized_parallax
        self.visible = True
        self.chunk_size = chunk_size

        self.chunks = {}
        self.sprites = []
        self.large_sprites = []

        self.effects: List[Tuple[Any, tuple]] = []
        self.emitters: List[Tuple[Any, tuple]] = []

        # Optimization: Reuse surfaces
        self._cached_surf = None
        self._cached_view_w = 0
        self._cached_view_h = 0

    def add_effect(self, effect_fn, *args):
        self.effects.append((effect_fn, args))

    def add_static(self, sprite):
        # Optimization: pre-calculate chunk keys to avoid doing it every frame
        if sprite.width > self.chunk_size or sprite.height > self.chunk_size:
            self.large_sprites.append(sprite)
            return

        cx = int(sprite.x // self.chunk_size)
        cy = int(sprite.y // self.chunk_size)

        if (cx, cy) not in self.chunks:
            self.chunks[(cx, cy)] = []
        self.chunks[(cx, cy)].append(sprite)

    def add_dynamic(self, sprite):
        self.sprites.append(sprite)

    def remove_static(self, sprite, x=None, y=None):
        if sprite in self.large_sprites:
            self.large_sprites.remove(sprite)
            return True

        # Use provided coordinates for faster lookup if available
        check_x = x if x is not None else sprite.x
        check_y = y if y is not None else sprite.y

        cx = int(check_x // self.chunk_size)
        cy = int(check_y // self.chunk_size)

        if (cx, cy) in self.chunks:
            if sprite in self.chunks[(cx, cy)]:
                self.chunks[(cx, cy)].remove(sprite)
                # Cleanup empty chunks to keep iteration fast
                if not self.chunks[(cx, cy)]:
                    del self.chunks[(cx, cy)]
                return True

        # Fallback search (slower)
        for ox in [-1, 0, 1]:
            for oy in [-1, 0, 1]:
                ncx, ncy = cx + ox, cy + oy
                if (ncx, ncy) in self.chunks and sprite in self.chunks[(ncx, ncy)]:
                    self.chunks[(ncx, ncy)].remove(sprite)
                    if not self.chunks[(ncx, ncy)]:
                        del self.chunks[(ncx, ncy)]
                    return True

        return False

    def _get_view_surface(self, view_w, view_h):
        # Optimization: Only recreate surface if size changes significantly
        # or if it hasn't been created yet.
        if (self._cached_surf is None or
                view_w > self._cached_surf.get_width() or
                view_h > self._cached_surf.get_height()):
            # Allocate 20% extra to prevent constant reallocation during small resizes
            new_w = int(view_w * 1.2)
            new_h = int(view_h * 1.2)
            self._cached_surf = pygame.Surface((new_w, new_h), pygame.SRCALPHA)

        # Clear only the area we are going to use (faster than filling the whole buffer)
        # However, due to scrolling, we usually just clear the sub-rect we need.
        surf = self._cached_surf.subsurface((0, 0, view_w, view_h))
        surf.fill((0, 0, 0, 0))
        return surf

    def render(self, screen: pygame.Surface, camera: Camera, emitters=None):
        if not self.visible: return

        screen_w, screen_h = screen.get_size()

        zoom = camera.zoom
        if abs(zoom - camera.target_zoom) < 0.001:
            zoom = camera.target_zoom

        view_w = int(screen_w / zoom)
        view_h = int(screen_h / zoom)

        layer_surf = self._get_view_surface(view_w, view_h)

        cx = camera.x * self.parallax
        cy = camera.y * self.parallax

        start_chunk_x = int(cx // self.chunk_size)
        end_chunk_x = int((cx + view_w) // self.chunk_size) + 1
        start_chunk_y = int(cy // self.chunk_size)
        end_chunk_y = int((cy + view_h) // self.chunk_size) + 1

        # Optimization: Collect blits for batch processing (optional, but cleaner)
        # Using direct blits here as it's often faster than constructing a list for fblits
        # when we need to do math on coordinates (subtracting cx, cy).

        # 1. Static Chunks
        # We iterate only the necessary range.
        for x in range(start_chunk_x - 1, end_chunk_x + 1):
            for y in range(start_chunk_y - 1, end_chunk_y + 1):
                chunk_key = (x, y)
                # Dictionary lookup is O(1)
                if chunk_key in self.chunks:
                    for s in self.chunks[chunk_key]:
                        # Culling: fast rect check
                        # We do the bounds check relative to camera here
                        sx = int(s.x - cx)
                        sy = int(s.y - cy)
                        if -s.width < sx < view_w and -s.height < sy < view_h:
                            layer_surf.blit(s.surface, (sx, sy))


        for s in self.large_sprites:
            sx = int(s.x - cx)
            sy = int(s.y - cy)
            if -s.width < sx < view_w and -s.height < sy < view_h:
                layer_surf.blit(s.surface, (sx, sy))

        for s in self.sprites:
            sx = int(s.x - cx)
            sy = int(s.y - cy)
            if -s.width < sx < view_w and -s.height < sy < view_h:
                # self.tick_update(s)
                layer_surf.blit(s.surface, (sx, sy))

        actual_emitters = emitters if emitters is not None else self.emitters
        if actual_emitters:
            for emitter in actual_emitters:
                emitter.draw(layer_surf, camera, self.parallax)

        for effect_fn, args in self.effects:
            effect_fn(layer_surf, *args)

        if zoom != 1.0:
            # Optimization: Use scale (nearest neighbor) if smoothscale is too slow.
            # Smoothscale is heavy. If you want retro look, scale is better anyway.
            # Using smoothscale to maintain your original style.
            scaled_output = pygame.transform.smoothscale(layer_surf, (screen_w, screen_h))
            screen.blit(scaled_output, (0, 0))
        else:
            screen.blit(layer_surf, (0, 0))

    def update(self, dt):
        pass

    def tick_update(self, sprite, dt):
        if hasattr(sprite, 'update'):
            if sprite.tick_rate > 0:
                sprite._tick_timer += dt
                interval = 1.0 / sprite.tick_rate

                if sprite._tick_timer >= interval:
                    sprite.update(sprite._tick_timer)
                    sprite._tick_timer %= interval
            else:
                # Comportamento normale
                sprite.update(dt)

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_cached_surf' in state: del state['_cached_surf']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cached_surf = None


class LightSource:
    _cache = {}

    def __init__(self, x, y, radius=100, color=(255, 255, 200), brightness=1.0,
                 diffusion_style='misty', diffusion_quality=2, falloff=1,
                 diffusion_function=lambda i, fall: 1 - fall ** i, steps=10):
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
            current_radius = int(self.radius - i * steps / self.radius)
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

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'surface' in state:
            del state['surface']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.surface = self._get_surface()

    def update(self, dt):
        pass


class LitLayer(ChunkedLayer):
    def __init__(self, name, parallax=1, ambient_color=(20, 20, 30), realized_parallax=None):
        super().__init__(name, parallax, realized_parallax=realized_parallax)
        self.lights = []
        self.ambient_color = ambient_color
        # Optimization: Reuse the light map surface
        self._cached_light_map = None

    def add_light(self, light: LightSource):
        self.lights.append(light)
        return light

    def render(self, screen, camera, emitters=None):
        if not self.visible: return

        screen_w, screen_h = screen.get_size()
        zoom = camera.zoom
        if abs(zoom - camera.target_zoom) < 0.001:
            zoom = camera.target_zoom

        view_w = int(screen_w / zoom)
        view_h = int(screen_h / zoom)

        # 1. Render all geometry normally first
        # We call the superclass logic manually to avoid redundant checks,
        # but for simplicity and maintenance, we can just reuse the drawing logic
        # or copy the relevant optimized parts.

        layer_surf = self._get_view_surface(view_w, view_h)
        cx, cy = camera.x * self.parallax, camera.y * self.parallax

        # --- DRAW SPRITES (Geometry) ---
        start_chunk_x = int(cx // self.chunk_size)
        end_chunk_x = int((cx + view_w) // self.chunk_size) + 1
        start_chunk_y = int(cy // self.chunk_size)
        end_chunk_y = int((cy + view_h) // self.chunk_size) + 1

        # Use a blit sequence for potential speedup in Pygame 2+
        blits_sequence = []

        # Chunks
        for x in range(start_chunk_x - 1, end_chunk_x + 1):
            for y in range(start_chunk_y - 1, end_chunk_y + 1):
                chunk_key = (x, y)
                if chunk_key in self.chunks:
                    for s in self.chunks[chunk_key]:
                        sx = int(s.x - cx)
                        sy = int(s.y - cy)
                        if -s.width < sx < view_w and -s.height < sy < view_h:
                            layer_surf.blit(s.surface, (sx, sy))

        # Large & Dynamic
        for s in self.large_sprites + self.sprites:
            sx = int(s.x - cx)
            sy = int(s.y - cy)
            if -s.width < sx < view_w and -s.height < sy < view_h:
                layer_surf.blit(s.surface, (sx, sy))

        # --- APPLY LIGHTING (Global, not per-sprite) ---
        # This is 100x faster than creating a surface for every block.

        # 1. Prepare Light Map
        if (self._cached_light_map is None or
                self._cached_light_map.get_size() != (view_w, view_h)):
            self._cached_light_map = pygame.Surface((view_w, view_h), 0)  # No alpha needed for map

        light_map = self._cached_light_map
        # Fill with darkness (Ambient)
        light_map.fill(self.ambient_color)

        # 2. Add Lights (Additive)
        # We only draw lights that are visible
        view_rect = pygame.Rect(0, 0, view_w, view_h)

        for l in self.lights:
            lx = int(l.x - cx - l.radius)
            ly = int(l.y - cy - l.radius)

            # Fast culling: Check if light rect intersects view
            # Light dimensions are radius*2
            diam = l.radius * 2
            if lx + diam > 0 and lx < view_w and ly + diam > 0 and ly < view_h:
                # BLEND_ADD adds the light color to the ambient darkness
                light_map.blit(l.surface, (lx, ly), special_flags=pygame.BLEND_ADD)

        # 3. Apply Light Map to Geometry (Multiplicative)
        # BLEND_MULT: Layer Color * Light Map Color.
        # Transparent pixels (alpha 0) in layer_surf remain transparent.
        # Visible pixels get tinted by the light map.
        layer_surf.blit(light_map, (0, 0), special_flags=pygame.BLEND_MULT)

        # --- Emitters & Effects (Post-lighting usually, or pre-lighting?) ---
        # Usually particles glow, so they might be drawn *after* lighting or
        # drawn *onto* the layer_surf before lighting.
        # Your original code drew them after per-sprite lighting, effectively making them "lit"
        # if they were blitted, or unlit if drawn on top.
        # Standard: Particles are often self-illuminated. We draw them AFTER multiply.

        actual_emitters = emitters if emitters is not None else self.emitters
        if actual_emitters:
            for emitter in actual_emitters:
                emitter.draw(layer_surf, camera, self.parallax)

        for effect_fn, args in self.effects:
            effect_fn(layer_surf, *args)

        # --- Final Render ---
        if zoom != 1.0:
            # Use smoothscale for quality, or scale for speed
            scaled_output = pygame.transform.smoothscale(layer_surf, (screen_w, screen_h))
            screen.blit(scaled_output, (0, 0))
        else:
            screen.blit(layer_surf, (0, 0))

    def update(self, dt):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_cached_surf' in state: del state['_cached_surf']
        if '_cached_light_map' in state: del state['_cached_light_map']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cached_surf = None
        self._cached_light_map = None


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

        tile_sprite = SolidSprite(world_x, world_y, self.tile_size, self.tile_size, color, texture=self.texture,
                                  alpha=True)

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

    def update(self, dt):
        pass


LAYER_BACK = 'back'
LAYER_MID = 'middle'
LAYER_FRONT = 'front'
LAYER_ORDER = (LAYER_BACK, LAYER_MID, LAYER_FRONT)


class BlockMap:
    def __init__(self, level, layers: dict, tile_size=40):
        """
        layers = {
            'back':   <ChunkedLayer or LitLayer>,
            'middle': <LitLayer>,
            'front':  <ChunkedLayer or LitLayer>,
        }
        """
        self.level = level
        self.tile_size = tile_size
        self.render_layers = layers

        self.data = {k: {} for k in LAYER_ORDER}
        self.lights = {k: {} for k in LAYER_ORDER}
        self.physics_blocks = {k: [] for k in LAYER_ORDER}
        self.tick_rate = 0
        self._tick_timer = 0

        self.hover_timer = 0.0
        self.hover_default_color = (190, 190, 255)
        self.hover_color = (190, 190, 255)
        self.cursor_gx = 0
        self.cursor_gy = 0
        self.active_layer = LAYER_MID

        self._waila_target = None
        self._waila_alpha = 0.0
        self._waila_offset_x = 0.0
        self._waila_target_alpha = 0.0
        self._waila_target_offset = 0.0
        self._waila_font_name = "PixeloidSans.ttf"
        self._waila_font_desc = None
        self._waila_surf = None

        self.depth_effects = {
            'front_fade': True,
            'back_desaturate': False,
            'back_darken': False,
            'front_shadow': False,
            'back_blur': False,
        }

        self._shadow_surf_cache = {}

    def _layer(self, name):
        return self.render_layers[name]

    def get_grid_pos(self, screen_x, screen_y, layer_name=LAYER_MID):
        cam_x = self.level.main_camera.x * self._layer(layer_name).parallax
        cam_y = self.level.main_camera.y * self._layer(layer_name).parallax
        return int((screen_x + cam_x) // self.tile_size), int((screen_y + cam_y) // self.tile_size)

    def get_grid_distance(self, p, q, layer_name=LAYER_MID):
        p = self.get_grid_pos(*p, layer_name)
        q = self.get_grid_pos(*q, layer_name)
        return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

    def get_grid_distance2(self, p, q, layer_name=LAYER_MID):
        if hasattr(p, 'x') and hasattr(p, 'y'):
            px_world, py_world = p.x, p.y
        else:
            px_world, py_world = p
        gx_p = int(px_world // self.tile_size)
        gy_p = int(py_world // self.tile_size)
        gx_q, gy_q = self.get_grid_pos(*q, layer_name)
        return math.sqrt((gx_p - gx_q) ** 2 + (gy_p - gy_q) ** 2)

    def _place_internal(self, gx, gy, block, layer_name, placer=None):
        world_x = gx * self.tile_size
        world_y = gy * self.tile_size
        tile_sprite = block.place(world_x, world_y)
        block.on_place(other=placer, layer=self._layer(layer_name))

        rl = self._layer(layer_name)
        if block.light_emission_intensity > 0 and isinstance(rl, LitLayer) and not block.physics_block:
            self.lights[layer_name][(gx, gy)] = rl.add_light(
                LightSource(world_x, world_y,
                            radius=block.light_emission_intensity * 100,
                            color=block.light_emission_color, falloff=0.99, steps=200))
        if block.light_emission_intensity > 0 and isinstance(rl, LitLayer) and block.physics_block:
            flag("Light Emitting Blocks do not support Physics as for now", level=2)

        rl.add_static(tile_sprite)
        self.data[layer_name][(gx, gy)] = tile_sprite

        if layer_name == LAYER_MID:
            if hasattr(self.level.grid, 'static_grid'):
                self.level.grid.static_grid.insert(tile_sprite)
            else:
                self.level.solids.append(tile_sprite)  # Fallback se usi ancora SpatialGrid

                # Solo i blocchi che cadono e i giocatori vanno nella lista che si aggiorna a ogni frame
            if tile_sprite.physics_block:
                self.physics_blocks[layer_name].append(tile_sprite)
                self.level.solids.append(tile_sprite)

    def refresh_static_grid(self):
        if hasattr(self.level.grid, 'static_grid'):
            self.level.grid.static_grid.clear()
            for (gx, gy), sprite in self.data[LAYER_MID].items():
                if not getattr(sprite, 'physics_block', False):
                    self.level.grid.static_grid.insert(sprite)

    def place_tile(self, screen_x, screen_y, block, layer_name=None, placer=None):
        layer_name = layer_name or self.active_layer
        gx, gy = self.get_grid_pos(screen_x, screen_y, layer_name)
        if (gx, gy) in self.data[layer_name]:
            return
        self._place_internal(gx, gy, block, layer_name, placer=placer)

    def set_tile(self, gx, gy, block, layer_name=None, overwrite=False):
        layer_name = layer_name or self.active_layer
        if (gx, gy) in self.data[layer_name] and not overwrite:
            flag(f"Position ({gx}, {gy}) is already occupied on '{layer_name}'. Use overwrite=True")
            return
        if overwrite and (gx, gy) in self.data[layer_name]:
            self.del_tile(gx, gy, layer_name)
        self._place_internal(gx, gy, block, layer_name)

        self.refresh_static_grid()

    def remove_tile(self, screen_x, screen_y, layer_name=None):
        layer_name = layer_name or self.active_layer
        gx, gy = self.get_grid_pos(screen_x, screen_y, layer_name)
        self.del_tile(gx, gy, layer_name)

    def has_tile(self, screen_x, screen_y, layer_name=None):
        l = layer_name if layer_name is not None else 'middle'
        return True if self.data[l].get(self.get_grid_pos(screen_x, screen_y, layer_name=l),
                                        None) is not None else False

    def get_tile(self, screen_x, screen_y, layer_name=None):
        l = layer_name if layer_name is not None else 'middle'
        return self.data[l].get(self.get_grid_pos(screen_x, screen_y, layer_name=l),
                                None)

    def del_tile(self, gx, gy, layer_name=None):
        layer_name = layer_name or self.active_layer
        if (gx, gy) in self.data[layer_name]:
            sprite = self.data[layer_name][(gx, gy)]
            self._layer(layer_name).remove_static(sprite)
            if layer_name == LAYER_MID:
                if sprite in self.level.solids:
                    self.level.solids.remove(sprite)
                if sprite in self.physics_blocks[layer_name]:
                    self.physics_blocks[layer_name].remove(sprite)
            del self.data[layer_name][(gx, gy)]

        if (gx, gy) in self.lights[layer_name]:
            self._layer(layer_name).lights.remove(self.lights[layer_name][(gx, gy)])
            del self.lights[layer_name][(gx, gy)]

        self.refresh_static_grid()

    def placer_light(self, screen_x, screen_y, color=None):
        self.hover_color = color if color else self.hover_default_color
        self.cursor_gx, self.cursor_gy = self.get_grid_pos(screen_x, screen_y, self.active_layer)

    def _waila_lookup(self, screen_x, screen_y):
        for ln in (LAYER_MID, LAYER_FRONT, LAYER_BACK):
            gx, gy = self.get_grid_pos(screen_x, screen_y, ln)
            if (gx, gy) in self.data[ln]:
                return self.data[ln][(gx, gy)]
        return None

    def _waila_render_panel(self, block):
        from Graphic.gui import FontManager
        name_font = FontManager.get(self._waila_font_name, 18, bold=True)
        desc_font = FontManager.get(self._waila_font_desc, 14)

        name_surf = name_font.render(block.name, True, (255, 255, 255))
        desc_text = getattr(block, 'description', '') or block.id
        desc_surf = desc_font.render(desc_text, True, (180, 180, 180))

        pad = 10
        w = max(name_surf.get_width(), desc_surf.get_width()) + pad * 2
        h = name_surf.get_height() + desc_surf.get_height() + pad * 3

        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((20, 20, 30, 180))
        pygame.draw.rect(panel, (100, 100, 140, 200), (0, 0, w, h), 2, border_radius=4)

        panel.blit(name_surf, (pad, pad))
        panel.blit(desc_surf, (pad, pad + name_surf.get_height() + pad))

        self._waila_surf = panel

    def _waila_update(self, dt, screen_x, screen_y):
        block = self._waila_lookup(screen_x, screen_y)

        if block is not self._waila_target:
            self._waila_target = block
            self._waila_surf = None  # invalidate cache

        if block:
            self._waila_target_alpha = 1.0
            self._waila_target_offset = 0.0
        else:
            self._waila_target_alpha = 0.0
            self._waila_target_offset = 60.0

        k = 12.0
        self._waila_alpha += (self._waila_target_alpha - self._waila_alpha) * min(1.0, k * dt)
        self._waila_offset_x += (self._waila_target_offset - self._waila_offset_x) * min(1.0, k * dt)

    def _waila_draw(self, surface):
        if self._waila_alpha < 0.01 or self._waila_target is None:
            return
        if self._waila_surf is None:
            self._waila_render_panel(self._waila_target)

        panel = self._waila_surf.copy()
        panel.set_alpha(int(self._waila_alpha * 220))

        mx, my = pygame.mouse.get_pos()
        sw, sh = surface.get_size()
        pw, ph = panel.get_size()

        draw_x = mx + 14 + self._waila_offset_x
        draw_y = my - ph - 6

        if draw_x + pw > sw:
            draw_x = mx - pw - 14 - self._waila_offset_x
        if draw_y < 0:
            draw_y = my + 20

        surface.blit(panel, (draw_x, draw_y))

    def update(self, dt, mouse_pos=None):
        self.hover_timer += dt * 2.5

        if mouse_pos:
            self._waila_update(dt, *mouse_pos)

        if self.depth_effects['front_fade']:
            self._update_front_layer_fade(dt)

        if self.depth_effects['back_desaturate'] or self.depth_effects['back_darken']:
            self._update_back_layer_effects()

        if not self.physics_blocks[LAYER_MID]:
            return

        blocks_to_reassign = []
        for block in self.physics_blocks[LAYER_MID]:
            moved = block.update_physics(dt, self.level.grid)
            if moved and block.is_grounded:
                new_gx = int(block.x // self.tile_size)
                new_gy = int(block.y // self.tile_size)
                old_key = None
                for key, sprite in self.data[LAYER_MID].items():
                    if sprite is block:
                        old_key = key
                        break
                if old_key and old_key != (new_gx, new_gy):
                    blocks_to_reassign.append((block, old_key, (new_gx, new_gy)))

        for block, old_key, new_key in blocks_to_reassign:
            old_world_x = old_key[0] * self.tile_size
            old_world_y = old_key[1] * self.tile_size
            self._layer(LAYER_MID).remove_static(block, x=old_world_x, y=old_world_y)
            del self.data[LAYER_MID][old_key]
            self._layer(LAYER_MID).add_static(block)
            self.data[LAYER_MID][new_key] = block
            self.level.grid.clear()
            for obj in self.level.solids:
                self.level.grid.insert(obj)

    def _update_front_layer_fade(self, dt):
        if not hasattr(self.level, 'player') or self.level.player is None:
            return

        player = self.level.player
        fade_distance = self.tile_size * 3
        k = 8.0

        # 1. Trova le coordinate griglia del player
        player_gx = int((player.x + player.width / 2) // self.tile_size)
        player_gy = int((player.y + player.height / 2) // self.tile_size)

        # 2. Ripristina l'opacità per TUTTI i blocchi modificati di recente
        # (dovrai tenere traccia dei blocchi faddati in una lista a parte come self.faded_blocks)

        # 3. Itera SOLO in un quadrato attorno al player (es: raggio di 4 blocchi)
        for x in range(player_gx - 4, player_gx + 5):
            for y in range(player_gy - 4, player_gy + 5):
                sprite = self.data[LAYER_FRONT].get((x, y))
                if sprite:
                    # ORA fai il tuo calcolo sqrt e l'alpha, ma lo farai massimo 81 volte,
                    # non importa se la mappa ha 2 milioni di blocchi!
                    pass

    def _update_back_layer_effects(self):
        import pygame
        for (gx, gy), sprite in self.data[LAYER_BACK].items():
            if not hasattr(sprite, '_depth_modified'):
                original_surf = sprite.surface.copy()

                if self.depth_effects['back_darken']:
                    dark_overlay = pygame.Surface(original_surf.get_size(), pygame.SRCALPHA)
                    dark_overlay.fill((0, 0, 0, 60))
                    original_surf.blit(dark_overlay, (0, 0))

                if self.depth_effects['back_desaturate']:
                    pixels = pygame.surfarray.pixels3d(original_surf)
                    gray = (pixels[:, :, 0] * 0.299 + pixels[:, :, 1] * 0.587 + pixels[:, :, 2] * 0.114).astype(
                        np.uint8)
                    pixels[:, :, 0] = pixels[:, :, 0] * 0.6 + gray * 0.4
                    pixels[:, :, 1] = pixels[:, :, 1] * 0.6 + gray * 0.4
                    pixels[:, :, 2] = pixels[:, :, 2] * 0.6 + gray * 0.4
                    del pixels

                if self.depth_effects['back_blur']:
                    w, h = original_surf.get_size()
                    small = pygame.transform.smoothscale(original_surf, (w // 2, h // 2))
                    original_surf = pygame.transform.smoothscale(small, (w, h))

                sprite.surface = original_surf
                sprite._depth_modified = True

    def _create_shadow(self, width, height):
        cache_key = (width, height)
        if cache_key in self._shadow_surf_cache:
            return self._shadow_surf_cache[cache_key]

        shadow = pygame.Surface((width + 8, height + 8), pygame.SRCALPHA)
        for i in range(4):
            alpha = 100
            offset = i * 2
            pygame.draw.rect(shadow, (0, 0, 0, alpha),
                             (offset, offset, width, height), border_radius=2)

        self._shadow_surf_cache[cache_key] = shadow
        return shadow

    def render(self, surface, camera):
        cam_x = camera.x * self._layer(self.active_layer).parallax
        cam_y = camera.y * self._layer(self.active_layer).parallax

        if self.depth_effects['front_shadow']:
            self._render_front_shadows(surface, camera)

        rect_x = (self.cursor_gx * self.tile_size) - cam_x
        rect_y = (self.cursor_gy * self.tile_size) - cam_y

        pulse_alpha = int(30 + math.sin(self.hover_timer) * 15)

        cursor_surf = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        cursor_surf.fill((*self.hover_color, pulse_alpha))
        pygame.draw.rect(cursor_surf, (*self.hover_color, 70), (0, 0, self.tile_size, self.tile_size), 1)
        surface.blit(cursor_surf, (rect_x, rect_y))

    def _render_front_shadows(self, surface, camera):
        cam_x = camera.x * self._layer(LAYER_MID).parallax
        cam_y = camera.y * self._layer(LAYER_MID).parallax

        player = self.level.player
        player_cy = player.y + player.height / 2

        for (gx, gy), sprite in self.data[LAYER_MID].items():
            block_cy = sprite.y + sprite.height / 2
            if (gx - 1, gy) in self.data[LAYER_BACK]:
                if block_cy < player_cy:
                    shadow = self._create_shadow(sprite.width, sprite.height)
                    sx = sprite.x - cam_x - 4
                    sy = sprite.y - cam_y - 4

                    current_alpha = getattr(sprite, '_current_alpha', 255)
                    shadow_alpha = int((current_alpha / 255) * 0.8 * 255)
                    shadow.set_alpha(shadow_alpha)

                    surface.blit(shadow, (int(sx), int(sy)))

    def __getstate__(self):
        state = self.__dict__.copy()
        for k in ('render_layers', 'level', '_waila_surf'):
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save(self, path=''):
        metadata = {str(s): self.data['middle'][s].get_metadata() for s in self.data['middle']}
        f2 = json.dumps(metadata)
        with open(path + f'/-blockmap_metadata.json', 'w') as file:
            file.write(f2)

        for ln in LAYER_ORDER:
            data = {str(s): self.data[ln][s].id for s in self.data[ln]}
            f = json.dumps(data)
            with open(path + f'/-blockmap_{ln}.json', 'w') as file:
                file.write(f)
        flag("Saved BlockMap")

    def load(self, level, path=''):
        found_any = False
        loaded = {}
        fname = path + f'/-blockmap_metadata.json'
        with open(fname) as file:
            loaded_metadata = json.loads(file.read())

        for ln in LAYER_ORDER:
            fname = path + f'/-blockmap_{ln}.json'
            try:
                with open(fname) as file:
                    loaded[ln] = json.loads(file.read())
                found_any = True
            except FileNotFoundError:
                flag(f"Not Found -blockmap_{ln}.json")

        if not found_any:
            try:
                with open(path + '/-blockmap.json') as file:
                    loaded[LAYER_MID] = json.loads(file.read())
                found_any = True
                flag("Migrated legacy blockmap -> middle layer", level=0)
            except FileNotFoundError:
                pass

        if not found_any:
            flag(f"No BlockMap files found at {path}", level=2)
            return

        self.reset(level)

        for ln, datas in loaded.items():
            for block in datas:
                if datas[block] not in level.registered_blocks:
                    print(datas[block])

                self.set_tile(*eval(block),
                              level.registered_blocks.get(datas[block], level.registered_blocks['_None']),
                              layer_name=ln
                              )
                if block in loaded_metadata:
                    self.data[ln][*eval(block)].set_metadata(loaded_metadata[block])

        self.refresh_static_grid()

    def loadstruct(self, level, path=''):
        for ln in LAYER_ORDER:
            fname = path + f'/am-blockmap_{ln}.json'
            try:
                with open(fname) as file:
                    datas = json.loads(file.read())
            except FileNotFoundError:
                continue
            for block in datas:
                self.set_tile(*eval(block), level.registered_blocks[datas[block]], layer_name=ln)

    def reset(self, level):
        for ln in LAYER_ORDER:
            for tile in list(self.data[ln].keys()):
                self.del_tile(*tile, ln)
            self.data[ln] = {}
            self.physics_blocks[ln] = []
            self.lights[ln] = {}


class UILayer(Layer):
    def __init__(self, name="UI", parallax=0):
        super().__init__(name, parallax=parallax)
        self.elements: List[UIElement] = []

    def add_element(self, element: UIElement):
        self.elements.append(element)
        return element

    def process_events(self, event):
        """Must be called inside the main game loop event pump"""
        if not self.visible: return

        for el in reversed(self.elements):
            if el.handle_event(event):
                break

    def update(self, dt):
        if not self.visible: return
        for el in self.elements:
            el.update()

    def render(self, screen: pygame.Surface, camera: Camera = None, emitters=None):
        if not self.visible: return
        super().render(screen, camera)

        for el in self.elements:
            el.draw(screen)
