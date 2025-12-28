import pygame
import numpy as np
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
import random
import math

PARTICLE_EMISSION_QUALITY = 1
EFFECTS_QUALITY = 2


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
    def linear_bloom(surface: pygame.Surface, strength=10.0, quality=EFFECTS_QUALITY):
        if quality <= 0: return

        scale = [0, 8, 6, 4, 2, 1][quality]
        size = surface.get_size()
        small_size = (max(1, size[0] // scale), max(1, size[1] // scale))

        small_surf = pygame.transform.smoothscale(surface, small_size)

        pixels = pygame.surfarray.array3d(small_surf).astype(np.float16)
        alphas = pygame.surfarray.array_alpha(small_surf).astype(np.float16)

        luma = np.max(pixels, axis=2) / 255.0
        alpha_weight = alphas / 255.0

        weight = luma * alpha_weight * strength

        processed = pixels * weight[..., None]
        np.clip(processed, 0, 255, out=processed)

        bloom_small = pygame.Surface(small_size, pygame.SRCALPHA)
        pygame.surfarray.blit_array(bloom_small, processed.astype(np.uint8))

        pygame.surfarray.pixels_alpha(bloom_small)[...] = np.clip(np.max(processed, axis=2) * 2, 0, 255).astype(
            np.uint8)

        final_glow = pygame.transform.smoothscale(bloom_small, size)
        surface.blit(final_glow, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    @staticmethod
    def halo(surface: pygame.Surface, radius=4, strength=1.5, tint=(255, 255, 255), quality=EFFECTS_QUALITY):
        if quality <= 0: return

        # 1. Determine Scale
        # We use a single, larger downscale factor based on radius to simulate blur
        scale = [0, 12, 10, 8, 4, 1][quality]
        orig_size = surface.get_size()

        # Calculate a size that accounts for the "radius" (blur intensity)
        # This prevents the drift caused by repeated // 2 operations
        blur_divisor = max(1, scale + (radius * 2))
        small_size = (max(1, orig_size[0] // blur_divisor),
                      max(1, orig_size[1] // blur_divisor))

        # 2. Extract and Process
        small_surf = pygame.transform.smoothscale(surface, small_size)
        pixels = pygame.surfarray.array3d(small_surf).astype(np.float16)
        alphas = pygame.surfarray.array_alpha(small_surf).astype(np.float16)

        # Luma weight logic
        weight = (np.max(pixels, axis=2) / 255.0) * (alphas / 255.0) * strength
        processed = weight[..., None] * np.array(tint, dtype=np.float16)
        np.clip(processed, 0, 255, out=processed)

        # 3. Create Halo Surface
        halo_small = pygame.Surface(small_size, pygame.SRCALPHA)
        pygame.surfarray.blit_array(halo_small, processed.astype(np.uint8))

        # Proportional Alpha
        new_alphas = np.clip(np.max(processed, axis=2) * 2, 0, 255).astype(np.uint8)
        pygame.surfarray.pixels_alpha(halo_small)[...] = new_alphas

        # 4. Final Upscale (The "Blur" effect)
        # By scaling directly back to orig_size, we avoid coordinate drift
        final_halo = pygame.transform.smoothscale(halo_small, orig_size)

        # 5. PINNED BLIT
        # Blit at (0, 0) because final_halo is a full-screen representation of the layer
        surface.blit(final_halo, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    @staticmethod
    def _obs_lumen(surface: pygame.Surface, threshold=100, intensity=2.5, quality=EFFECTS_QUALITY):
        if quality <= 0: return

        size = surface.get_size()
        scale = [0, 32, 16, 8, 4, 1][quality]

        small_size = (max(1, size[0] // scale), max(1, size[1] // scale))
        source_surf = pygame.transform.smoothscale(surface, small_size)

        pixels = pygame.surfarray.array3d(source_surf).astype(np.float16)

        luma = (pixels[..., 0] * 0.299 + pixels[..., 1] * 0.587 + pixels[..., 2] * 0.114)

        mask = luma > threshold
        emissive_pixels = np.zeros_like(pixels)
        emissive_pixels[mask] = pixels[mask] * intensity

        glow_buffer = pygame.Surface(small_size, pygame.SRCALPHA)
        np.clip(emissive_pixels, 0, 255, out=emissive_pixels)
        pygame.surfarray.blit_array(glow_buffer, emissive_pixels.astype(np.uint8))

        new_alphas = np.clip(np.max(emissive_pixels, axis=2) * 1.5, 0, 255).astype(np.uint8)
        pygame.surfarray.pixels_alpha(glow_buffer)[...] = new_alphas

        final_lumen = pygame.transform.smoothscale(glow_buffer, size)

        surface.blit(final_lumen, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    @staticmethod
    def lumen(surface: pygame.Surface, threshold=100, intensity=2.5, tint=(255, 255, 255), quality=EFFECTS_QUALITY):
        if quality <= 0: return

        size = surface.get_size()

        # --- BRANCH: FAST LOWER QUALITY (quality == 1) ---
        if quality == 1:
            # Scale down aggressively (32x) for speed
            small_size = (max(1, size[0] // 32), max(1, size[1] // 32))
            # 1. Capture and downscale (creates natural blur)
            glow_buffer = pygame.transform.smoothscale(surface, small_size)

            # 2. Fast Tint & Intensity (No numpy used here)
            tint_color = tuple(min(255, int(c * intensity)) for c in tint)
            glow_buffer.fill(tint_color, special_flags=pygame.BLEND_RGB_MULT)

            # 3. Upscale and Add
            final_lumen = pygame.transform.smoothscale(glow_buffer, size)
            surface.blit(final_lumen, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
            return

        # --- BRANCH: STANDARD/HIGH QUALITY (quality > 1) ---
        scale = [0, 32, 16, 8, 4, 1][quality]
        small_size = (max(1, size[0] // scale), max(1, size[1] // scale))
        source_surf = pygame.transform.smoothscale(surface, small_size)

        pixels = pygame.surfarray.array3d(source_surf).astype(np.float16)

        # Calculate luma to isolate the 'hot' spots
        luma = (pixels[..., 0] * 0.299 + pixels[..., 1] * 0.587 + pixels[..., 2] * 0.114)
        mask = luma > threshold

        # Apply Intensity AND Tint
        emissive_pixels = np.zeros_like(pixels)
        tint_array = np.array(tint, dtype=np.float16) / 255.0
        emissive_pixels[mask] = pixels[mask] * intensity * tint_array

        # Create the glow buffer
        glow_buffer = pygame.Surface(small_size, pygame.SRCALPHA)
        np.clip(emissive_pixels, 0, 255, out=emissive_pixels)
        pygame.surfarray.blit_array(glow_buffer, emissive_pixels.astype(np.uint8))

        # Generate alpha based on brightness to ensure smooth light falloff
        new_alphas = np.clip(np.max(emissive_pixels, axis=2) * 1.5, 0, 255).astype(np.uint8)
        pygame.surfarray.pixels_alpha(glow_buffer)[...] = new_alphas

        # Final upscale (the 'Lumen' bleed)
        final_lumen = pygame.transform.smoothscale(glow_buffer, size)
        surface.blit(final_lumen, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

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
    def light_shader(surface: pygame.Surface, amount=4, strength=1.5, quality=EFFECTS_QUALITY):
        """
        Designed for a dedicated 'Light Layer'.
        Blurs the emissive sources and adds them back to create a volumetric glow.
        """
        if quality <= 0 or amount <= 0: return

        size = surface.get_size()

        # 1. Generate the blur (the f(x) part)
        # We use a lower shrink factor for 'light' to keep the glow wide and soft
        shrink = 1 / (amount * [0, 8, 4, 2, 1, 1, 1][quality])
        small_size = (max(1, int(size[0] * shrink)), max(1, int(size[1] * shrink)))

        # Create the glow buffer
        # Note: We don't copy the whole surface if it's already a dedicated light layer
        glow_buffer = pygame.transform.smoothscale(surface, small_size)
        glow_buffer = pygame.transform.smoothscale(glow_buffer, size)

        # 2. Apply Strength/Gain
        if strength != 1.0:
            # This amplifies the light intensity before adding it back
            glow_buffer.fill((min(255, 255 * strength),) * 3, special_flags=pygame.BLEND_RGB_MULT)

        # 3. y = x + f(x)
        # The original sharp lights (x) + the blurred glow (f(x))
        surface.blit(glow_buffer, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

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

    @staticmethod
    def black_and_white(surface: pygame.Surface, intensity=1.0, quality=EFFECTS_QUALITY):
        if quality <= 0 or intensity <= 0: return

        # Use pixels3d to get a direct view (no copying)
        pixels = pygame.surfarray.pixels3d(surface)

        # Fast Integer Luma: (R+G+B) // 3 is much faster than dot products
        # We use a bit-shift for the average to stay on the CPU's fast path
        r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]

        # Standard weighted luma without float conversion
        # 0.299R + 0.587G + 0.114B approximated with integers:
        luma = (r.astype(np.uint16) * 77 + g.astype(np.uint16) * 150 + b.astype(np.uint16) * 29) >> 8

        if intensity >= 1.0:
            # Full B&W replacement is nearly instant in-place
            pixels[..., 0] = luma
            pixels[..., 1] = luma
            pixels[..., 2] = luma
        else:
            # Partial lerp (only used if intensity < 1.0)
            pixels[..., 0] = pixels[..., 0] + ((luma - pixels[..., 0]) * intensity).astype(np.uint8)
            pixels[..., 1] = pixels[..., 1] + ((luma - pixels[..., 1]) * intensity).astype(np.uint8)
            pixels[..., 2] = pixels[..., 2] + ((luma - pixels[..., 2]) * intensity).astype(np.uint8)

        del pixels

    @staticmethod
    def contrast(surface: pygame.Surface, contrast=1.2, quality=EFFECTS_QUALITY):
        if quality <= 0 or contrast == 1.0: return

        pixels = pygame.surfarray.pixels3d(surface)

        # Pre-compute a 256-value Lookup Table (LUT)
        # This is the secret to speed: we calculate the math 256 times instead of 1,000,000 times
        full_range = np.arange(256, dtype=np.float16)
        lut = ((full_range - 128) * contrast + 128)
        lut = np.clip(lut, 0, 255).astype(np.uint8)

        # Apply the LUT to the whole array at once
        pixels[...] = lut[pixels]

        del pixels

    @staticmethod
    def fcontrast(surface: pygame.Surface, contrast=1.2, quality=EFFECTS_QUALITY):
        if quality <= 0 or contrast == 1.0: return

        pixels = pygame.surfarray.pixels3d(surface)

        full_range = np.arange(256, dtype=np.float16)
        lut = ((full_range - 128.0) * contrast + 128.0)

        lut = np.clip(lut, 0, 255).astype(np.uint8)
        flat_pixels = pixels.reshape(-1)

        flat_pixels[...] = lut[flat_pixels]

        del pixels


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
            color = self.color if self.color != 'random' else (
            random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            pygame.draw.circle(surface, color, (px, py), max(1, int(size * self.size)))


class FireflyEmitter(ParticleEmitter):
    def __init__(self, color=(190, 200, 190), count=50, size=1):
        super().__init__(color, count, size, g=0, sparsity=1.0)
        self.firefly_data = np.random.uniform(0, 5, (int(count * PARTICLE_EMISSION_QUALITY), 5))

    def update(self, dt):
        if not np.any(self.active): return

        indices = np.where(self.active)[0]
        for i in indices:
            self.firefly_data[i, 1] += random.uniform(-2, 2) * dt
            angle = self.firefly_data[i, 1]

            speed = 20.0
            self.data[i, 2] = math.cos(angle) * speed
            self.data[i, 3] = math.sin(angle) * speed

        self.data[self.active, 0:2] += self.data[self.active, 2:4] * dt
        self.firefly_data[self.active, 0] += dt  # Timer

    def draw(self, surface, camera):
        indices = np.where(self.active)[0]
        for i in indices:
            px = int(self.data[i, 0] - camera.x)
            py = int(self.data[i, 1] - camera.y)

            # Pulse the alpha/size using a sine wave
            timer = self.firefly_data[i, 0]
            pulse = (math.sin(timer * 2.0 + self.firefly_data[i, 3]) + 1) / 2

            # Draw the core
            current_size = max(1, int(self.size * (1 + pulse / 4)))

            # For fireflies, we draw two circles: a bright core and a soft glow
            # Note: Pygame-ce draw.circle is fast, but for 100+ fireflies this is fine
            color = self.color
            pygame.draw.circle(surface, color, (px, py), current_size)

            # Add a subtle bloom/glow if we are on a transparent layer
            if pulse > 0.7:
                glow_color = (min(255, color[0] + 50), min(255, color[1] + 50), color[2], 100)
                pygame.draw.circle(surface, glow_color, (px, py), current_size * 2)


@dataclass
class Camera:
    x: float = 0.0
    y: float = 0.0
    width: int = 800
    height: int = 600
    target: Optional[Any] = None
    smooth: float = 0.1

    # New Shake Properties
    shake_intensity: float = 0.0
    shake_decay: float = 0.9  # How fast the shake stops (0.9 = 10% per frame)

    def apply_shake(self, intensity: float):
        self.shake_intensity = intensity

    def update(self):
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


class Sprite:
    def __init__(self, x, y, w, h, color=(255, 255, 255)):
        self.x, self.y, self.width, self.height = x, y, w, h
        self.surface = pygame.Surface((w, h))
        self.surface.fill(color)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy


class AnimatedSprite(Sprite):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)
        self.animations = {}  # Store lists of frames: {"idle": [surf1, surf2], "walk": [...]}
        self.current_state = "idle"
        self.frame_index = 0.0
        self.animation_speed = 10.0  # Frames per second
        self.frect = pygame.FRect(x, y, w, h)  # For compatibility with your new collision system

    def add_animation(self, name: str, frames: List[pygame.Surface]):
        """Register a list of surfaces for a specific state."""
        self.animations[name] = frames

    def update_animation(self, dt):
        """Advances the frame index based on time."""
        if self.current_state in self.animations:
            self.frame_index += self.animation_speed * dt

            if self.frame_index >= len(self.animations[self.current_state]):
                self.frame_index = 0

            current_frame = int(self.frame_index)
            self.surface = self.animations[self.current_state][current_frame]

    def set_state(self, state: str):
        if self.current_state != state:
            self.current_state = state
            self.frame_index = 0.0

    def move(self, dx, dy, grid):
        """Integrated with your SolidSprite logic."""
        # Horizontal
        self.frect.x += dx
        for other in grid.get_nearby(self.frect):
            if other is not self and self.frect.colliderect(other.frect):
                if dx > 0: self.frect.right = other.frect.left
                if dx < 0: self.frect.left = other.frect.right

        # Vertical
        self.frect.y += dy
        for other in grid.get_nearby(self.frect):
            if other is not self and self.frect.colliderect(other.frect):
                if dy > 0: self.frect.bottom = other.frect.top
                if dy < 0: self.frect.top = other.frect.bottom

        self.x, self.y = self.frect.x, self.frect.y

        # Logic to pick state based on movement
        if dx != 0 or dy != 0:
            self.set_state("walk")
        else:
            self.set_state("idle")


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


class AnimatedSolidSprite(SolidSprite):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h, (0, 0, 0, 0))
        self.animations = {}
        self.current_state = 'idle'
        self.frame_index = 0.0
        self.animation_speed = 12.0
        self.frect = pygame.FRect(x, y, w, h)

    def add_animation(self, name, frames):
        scaled_frames = []
        for f in frames:
            scaled_f = pygame.transform.scale(f, (self.width, self.height))
            scaled_frames.append(scaled_f.convert_alpha())
        self.animations[name] = scaled_frames

        if name == self.current_state:
            self.surface = self.animations[name][0]

    def update_animation(self, dt):
        if self.current_state not in self.animations: return

        frames = self.animations[self.current_state]
        self.frame_index += self.animation_speed * dt

        idx = int(self.frame_index) % len(frames)
        self.surface = frames[idx]


class Layer:
    def __init__(self, name: str, parallax: float = 1.0):
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
                layer_surf.blit(s.surface, (int(sx), int(sy)))

        if emitters is not None:
            for emitter in emitters:
                emitter.draw(layer_surf, camera)

        for effect_fn, args in self.effects:
            effect_fn(layer_surf, *args)

        screen.blit(layer_surf, (0, 0))


class Game:
    def __init__(self, w=200, h=300, title="Ortensia Engine", flag=pygame.RESIZABLE | pygame.SCALED | pygame.DOUBLEBUF):
        pygame.init()
        self.screen = pygame.display.set_mode((w, h), flag)
        pygame.display.set_caption(title)

        self.clock = pygame.time.Clock()
        self.camera = Camera(width=w, height=h)
        self.layers: List[Layer] = []
        self.particle_emitters = []
        self.particle_layer_idx = -1
        self.solids = []
        self.grid = SpatialGrid(cell_size=200)
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
        return int(x * 1)


    game = Game(s(1000), s(600), flag=pygame.SCALED | pygame.RESIZABLE)
    bg2 = game.add_layer("Background2", 0.2)
    bg = game.add_layer("Background", 0.5)
    particles = game.add_layer("particles", 1.0)
    fg = game.add_layer("Foreground", 1.0)

    particles.add_effect(PostProcessing.lumen, 10, 2)
    fg.add_effect(PostProcessing.black_and_white)
    from functions import *

    # player = SolidSprite(s(400), s(300), s(40), s(40), (255, 255, 255))
    player = AnimatedSolidSprite(s(400), s(300), s(40), s(40))
    player.add_animation('idle', load_spritesheet("examples/Hare_Run.png", 32, 32, row=3))
    fg.sprites.append(player)
    game.solids.append(player)
    game.camera.target = player

    emitter1 = ParticleEmitter(size=s(1.5), sparsity=0.2, g=40, color='random')
    ff = FireflyEmitter(count=100, size=1)
    game.particle_emitters.append(ff)
    game.particle_emitters.append(emitter1)

    for _ in range(30):
        ff.emit(random.randint(0, 1000), random.randint(200, 500), 1)
        pass
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
        speed = 500 * dt * 0.5

        dx, dy = 0, 0
        if keys[pygame.K_LEFT]:  dx -= speed
        if keys[pygame.K_RIGHT]: dx += speed
        if keys[pygame.K_UP]:    dy -= speed
        if keys[pygame.K_DOWN]:  dy += speed
        if keys[pygame.K_SPACE]:
            game.camera.shake_intensity = 5

        player.move(dx, dy, game_inst.grid)
        player.update_animation(dt)

        emitter1.emit(player.x + s(20), player.y + s(20))


    game.run(my_update)
