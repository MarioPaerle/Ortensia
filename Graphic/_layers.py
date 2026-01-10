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
    target_delta_y = -200.0

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
        # 1. Update Zoom Smoothly
        diff = self.target_zoom - self.zoom
        if abs(diff) < 0.001:
            self.zoom = self.target_zoom
        else:
            self.zoom += diff * self.zoom_smooth

        view_w = self.width / self.zoom
        view_h = self.height / self.zoom

        # 3. Track Target CENTER
        if self.target:
            t_w = getattr(self.target, 'width', 0)
            t_h = getattr(self.target, 'height', 0)

            target_center_x = self.target.x + t_w / 2 + self.target_delta_x
            target_center_y = self.target.y + t_h / 2 + self.target_delta_y

            self.scroll_x += (target_center_x - self.scroll_x) * self.smooth
            self.scroll_y += (target_center_y - self.scroll_y) * self.smooth

        # 4. Shake Logic
        shake_x, shake_y = 0, 0
        if self.shake_intensity > 0.1:
            shake_x = random.uniform(-self.shake_intensity, self.shake_intensity)
            shake_y = random.uniform(-self.shake_intensity, self.shake_intensity)
            self.shake_intensity *= self.shake_decay
        else:
            self.shake_intensity = 0

        # 5. Calculate Final Top-Left Coordinate
        # TopLeft = Center - Half_View_Size
        prev_x, prev_y = self.x, self.y

        self.x = (self.scroll_x - view_w // 2) + shake_x
        self.y = (self.scroll_y - view_h // 2) + shake_y

        # Update velocity (useful for parallax or motion blur effects)
        self.velocity_x = self.x - prev_x
        self.velocity_y = self.y - prev_y


class Layer:
    def __init__(self, name: str, parallax: float = 1.0, **kwargs):
        self.name = name
        self.parallax = parallax
        self.sprites: List[Sprite] = []
        self.effects: List[Tuple[Any, tuple]] = []
        self.emitters = []
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
                emitter.draw(layer_surf, camera, self.parallax)
                emitter.update()

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
        self.sprites = []  # Dynamic objects
        self.large_sprites = []  # NEW: Large static objects (Backgrounds, big trees)

        self.effects: List[Tuple[Any, tuple]] = []
        self.emitters: List[Tuple[Any, tuple]] = []
        self._cached_surf = None

    def add_effect(self, effect_fn, *args):
        self.effects.append((effect_fn, args))

    def add_static(self, sprite):
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

    def remove_static(self, sprite):
        # Check large sprites first
        if sprite in self.large_sprites:
            self.large_sprites.remove(sprite)
            return True

        cx = int(sprite.x // self.chunk_size)
        cy = int(sprite.y // self.chunk_size)

        if (cx, cy) in self.chunks:
            if sprite in self.chunks[(cx, cy)]:
                self.chunks[(cx, cy)].remove(sprite)
                return True
        return False

    def _get_view_surface(self, view_w, view_h):
        """Smart allocation for the render surface."""
        alloc_w, alloc_h = 0, 0
        if self._cached_surf:
            alloc_w, alloc_h = self._cached_surf.get_size()

        if view_w > alloc_w or view_h > alloc_h:
            # Create a buffer 20% larger than needed to prevent constant resizing
            new_w = int(view_w * 1.2)
            new_h = int(view_h * 1.2)
            self._cached_surf = pygame.Surface((new_w, new_h), pygame.SRCALPHA)

        elif view_w < alloc_w * 0.5:
            self._cached_surf = pygame.Surface((view_w, view_h), pygame.SRCALPHA)

        surf = self._cached_surf.subsurface((0, 0, view_w, view_h))
        surf.fill((0, 0, 0, 0))
        return surf

    def render(self, screen: pygame.Surface, camera: Camera, emitters=None):
        emitters = self.emitters if emitters is None else emitters
        if not self.visible: return

        screen_w, screen_h = screen.get_size()

        zoom = camera.zoom
        if abs(zoom - camera.target_zoom) < 0.001:
            zoom = camera.target_zoom

        view_w = int(screen_w / zoom)
        view_h = int(screen_h / zoom)

        layer_surf = self._get_view_surface(view_w, view_h)

        cx, cy = camera.x * self.parallax, camera.y * self.parallax

        start_chunk_x = int(cx // self.chunk_size)
        end_chunk_x = int((cx + view_w) // self.chunk_size) + 1
        start_chunk_y = int(cy // self.chunk_size)
        end_chunk_y = int((cy + view_h) // self.chunk_size) + 1

        for x in range(start_chunk_x - 1, end_chunk_x + 1):
            for y in range(start_chunk_y - 1, end_chunk_y + 1):
                chunk_key = (x, y)
                if chunk_key in self.chunks:
                    for s in self.chunks[chunk_key]:
                        sx = s.x - cx
                        sy = s.y - cy
                        if -s.width < sx < view_w and -s.height < sy < view_h:
                            layer_surf.blit(s.surface, (int(sx), int(sy)))

        for s in self.large_sprites:
            sx = s.x - cx
            sy = s.y - cy
            if -s.width < sx < view_w and -s.height < sy < view_h:
                layer_surf.blit(s.surface, (int(sx), int(sy)))

        for s in self.sprites:
            sx = s.x - cx
            sy = s.y - cy
            if -s.width < sx < view_w and -s.height < sy < view_h:
                if hasattr(s, 'update'):
                    s.update()
                layer_surf.blit(s.surface, (int(sx), int(sy)))

        """if len(self.sprites) != 0:
            print(self.sprites)"""

        if emitters is not None:
            for emitter in emitters:
                emitter.draw(layer_surf, camera, self.parallax)

        for effect_fn, args in self.effects:
            effect_fn(layer_surf, *args)

        if zoom != 1.0:
            scaled_output = pygame.transform.smoothscale(layer_surf, (screen_w, screen_h))
            screen.blit(scaled_output, (0, 0))
        else:
            screen.blit(layer_surf, (0, 0))


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


class LitLayer(ChunkedLayer):
    def __init__(self, name, parallax=1, ambient_color=(20, 20, 30)):
        super().__init__(name, parallax)
        self.lights = []
        self.ambient_color = ambient_color

    def add_light(self, light: LightSource):
        self.lights.append(light)

    def render(self, screen, camera, emitters=None):
        if not self.visible: return

        screen_w, screen_h = screen.get_size()

        # Stabilize Zoom
        zoom = camera.zoom
        if abs(zoom - camera.target_zoom) < 0.001:
            zoom = camera.target_zoom

        view_w = int(screen_w / zoom)
        view_h = int(screen_h / zoom)

        # Get surface (Lazy Alloc)
        layer_surf = self._get_view_surface(view_w, view_h)

        cx, cy = camera.x * self.parallax, camera.y * self.parallax

        # --- 1. Collect Candidates ---
        render_candidates = []

        start_chunk_x = int(cx // self.chunk_size)
        end_chunk_x = int((cx + view_w) // self.chunk_size) + 1
        start_chunk_y = int(cy // self.chunk_size)
        end_chunk_y = int((cy + view_h) // self.chunk_size) + 1

        # A. Chunks
        for x in range(start_chunk_x - 1, end_chunk_x + 1):
            for y in range(start_chunk_y - 1, end_chunk_y + 1):
                if (x, y) in self.chunks:
                    render_candidates.extend(self.chunks[(x, y)])

        # B. Large Sprites
        render_candidates.extend(self.large_sprites)

        # C. Dynamic Sprites
        render_candidates.extend(self.sprites)

        # --- 2. Collect Lights ---
        visible_lights = []
        for l in self.lights:
            lx, ly = l.x - cx, l.y - cy
            # Check overlap with View
            if (lx + l.radius > 0 and lx - l.radius < view_w and
                    ly + l.radius > 0 and ly - l.radius < view_h):
                visible_lights.append(l)

        # --- 3. Draw Loop ---
        for s in render_candidates:
            sx = s.x - cx
            sy = s.y - cy

            # Culling Check
            if not (-s.width < sx < view_w and -s.height < sy < view_h):
                continue

            # Lighting Logic
            w, h = s.surface.get_size()

            # Create lightmap for this sprite
            light_map = pygame.Surface((w, h), 0)
            light_map.fill(self.ambient_color)

            if hasattr(s, 'update'): s.update()

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

        # --- 4. Post-Processing ---
        if emitters:
            for em in emitters:
                em.draw(layer_surf, camera, self.parallax)

        for effect_fn, args in self.effects:
            effect_fn(layer_surf, *args)

        # --- 5. Final Scale ---
        if zoom != 1.0:
            scaled_output = pygame.transform.smoothscale(layer_surf, (screen_w, screen_h))
            screen.blit(scaled_output, (0, 0))
        else:
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


class BlockMap:
    def __init__(self, game, layer: ChunkedLayer, tile_size=40, texture=None):
        self.game = game
        self.layer = layer
        self.tile_size = tile_size
        self.texture = texture
        self.data = {}
        self.physics_blocks = []  # Track physics-enabled blocks

    def get_grid_pos(self, screen_x, screen_y):
        cam_x = self.game.main_camera.x * self.layer.parallax
        cam_y = self.game.main_camera.y * self.layer.parallax

        world_x = screen_x + cam_x
        world_y = screen_y + cam_y

        gx = int(world_x // self.tile_size)
        gy = int(world_y // self.tile_size)
        return gx, gy

    def place_tile(self, screen_x, screen_y, block: Block):
        gx, gy = self.get_grid_pos(screen_x, screen_y)
        if (gx, gy) in self.data:
            return

        world_x = gx * self.tile_size
        world_y = gy * self.tile_size

        tile_sprite = block.place(world_x, world_y)

        self.layer.add_static(tile_sprite)
        self.game.solids.append(tile_sprite)
        self.data[(gx, gy)] = tile_sprite

        # Track physics blocks separately
        if tile_sprite.physics_block:
            self.physics_blocks.append(tile_sprite)

    def remove_tile(self, screen_x, screen_y):
        gx, gy = self.get_grid_pos(screen_x, screen_y)

        if (gx, gy) in self.data:
            sprite = self.data[(gx, gy)]

            self.layer.remove_static(sprite)
            if sprite in self.game.solids:
                self.game.solids.remove(sprite)
            if sprite in self.physics_blocks:
                self.physics_blocks.remove(sprite)

            del self.data[(gx, gy)]

    def update(self, dt):
        """
        Update all physics-enabled blocks.
        Call this in your game loop after refresh_grid().
        """
        if not self.physics_blocks:
            return

        # Track blocks that need chunk reassignment
        blocks_to_reassign = []

        for block in self.physics_blocks:
            moved = block.update_physics(dt, self.game.grid)

            # If block moved significantly, we may need to update its chunk
            if moved and block.is_grounded:
                # Calculate new grid position
                new_gx = int(block.x // self.tile_size)
                new_gy = int(block.y // self.tile_size)

                # Find old position in data
                old_key = None
                for key, sprite in self.data.items():
                    if sprite is block:
                        old_key = key
                        break

                # If position changed, update data dictionary
                if old_key and old_key != (new_gx, new_gy):
                    blocks_to_reassign.append((block, old_key, (new_gx, new_gy)))

        # Reassign blocks to new grid positions
        for block, old_key, new_key in blocks_to_reassign:
            # Remove from old chunk
            self.layer.remove_static(block)
            del self.data[old_key]

            # Add to new chunk
            self.layer.add_static(block)
            self.data[new_key] = block

            # Refresh grid to ensure collision detection works
            self.game.grid.clear()
            for obj in self.game.solids:
                self.game.grid.insert(obj)
