from Graphic._layers import *
import numpy as np
import math
import datetime

PARTICLE_EMISSION_QUALITY = 1
EFFECTS_QUALITY = 4


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

        scale = [0, 32, 16, 8, 4, 1][quality]
        intensity = int(intensity * 6 / (quality + 1))
        small_size = (max(1, size[0] // scale), max(1, size[1] // scale))
        source_surf = pygame.transform.smoothscale(surface, small_size)

        pixels = pygame.surfarray.array3d(source_surf).astype(np.float16)

        luma = (pixels[..., 0] * 0.299 + pixels[..., 1] * 0.587 + pixels[..., 2] * 0.114)
        mask = luma > threshold

        emissive_pixels = np.zeros_like(pixels)
        tint_array = np.array(tint, dtype=np.float16) / 255.0
        emissive_pixels[mask] = pixels[mask] * intensity * tint_array

        glow_buffer = pygame.Surface(small_size, pygame.SRCALPHA)
        np.clip(emissive_pixels, 0, 255, out=emissive_pixels)
        pygame.surfarray.blit_array(glow_buffer, emissive_pixels.astype(np.uint8))

        new_alphas = np.clip(np.max(emissive_pixels, axis=2) * 1.5, 0, 255).astype(np.uint8)
        pygame.surfarray.pixels_alpha(glow_buffer)[...] = new_alphas

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

        pixels = pygame.surfarray.pixels3d(surface)

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

    @staticmethod
    def underwater_distortion(surface: pygame.Surface, amplitude=5, frequency=0.05,
                              quality=EFFECTS_QUALITY):
        if quality <= 0: return
        time_val = datetime.datetime.now().microsecond * 1e-6 * 2 * np.pi

        source = surface.copy()
        surface.fill((0, 0, 0, 0))

        width, height = surface.get_size()

        slice_h = [0, 16, 8, 4, 2, 1][quality]

        for y in range(0, height, slice_h):
            shift = int(math.sin(y * frequency + time_val) * amplitude)

            src_rect = pygame.Rect(0, y, width, slice_h)

            surface.blit(source, (shift, y), src_rect)
            if shift > 0:
                surface.blit(source, (shift - width, y), src_rect)
            elif shift < 0:
                surface.blit(source, (shift + width, y), src_rect)

    @staticmethod
    def motion_blur(surface: pygame.Surface, camera: Camera, strength_scale=1.0, quality=EFFECTS_QUALITY):
        if quality <= 0: return

        vx = camera.velocity_x * strength_scale
        vy = camera.velocity_y * strength_scale

        magnitude = abs(vx ** 2 + vy ** 2)

        if magnitude < 0.5:
            return

        samples = [1, 1, 2, 3, 5, 10][min(quality, 5)]

        max_blur = 60
        if magnitude > max_blur:
            scale_factor = max_blur / magnitude
            vx *= scale_factor
            vy *= scale_factor

        ghost = surface.copy()

        alpha = max(40, int(150 / samples))
        ghost.set_alpha(alpha)
        # -------------------

        step_x = vx / samples
        step_y = vy / samples

        for i in range(1, samples + 1):
            offset_x = -step_x * i
            offset_y = -step_y * i
            surface.blit(ghost, (offset_x, offset_y))

    @staticmethod
    def fast_motion_blur(surface: pygame.Surface, camera: Camera, strength_scale=1.0, quality=EFFECTS_QUALITY):
        """
        Optimized Motion Blur using Downsampling.
        """
        if quality <= 0: return

        vx = camera.velocity_x * strength_scale
        vy = camera.velocity_y * strength_scale
        magnitude = math.sqrt(vx ** 2 + vy ** 2)

        if magnitude < 1.0: return

        downscale_factor = 4 if quality == 1 else (2 if quality == 2 else 1)

        w, h = surface.get_size()
        small_w = max(1, w // downscale_factor)
        small_h = max(1, h // downscale_factor)

        work_surf = pygame.transform.smoothscale(surface, (small_w, small_h))

        samples = [2, 3, 5, 1][min(quality, 3)]

        vx /= downscale_factor
        vy /= downscale_factor

        max_blur = 40 / downscale_factor
        curr_mag = math.sqrt(vx ** 2 + vy ** 2)
        if curr_mag > max_blur:
            scale = max_blur / curr_mag
            vx *= scale
            vy *= scale

        ghost = work_surf.copy()

        alpha = max(50, int(180 / samples))
        ghost.set_alpha(alpha)

        step_x = vx / samples
        step_y = vy / samples

        for i in range(1, samples + 1):
            offset_x = -step_x * i
            offset_y = -step_y * i
            work_surf.blit(ghost, (offset_x, offset_y))

        final_blur = pygame.transform.smoothscale(work_surf, (w, h))

        final_blur.set_alpha(150)
        surface.blit(final_blur, (0, 0))

    _fog_noise = None

    @staticmethod
    def _generate_fog_noise(size):
        """Generates a simple cloud/noise texture using numpy."""
        width, height = size
        x = np.linspace(0, 1 * np.pi, width)
        y = np.linspace(0, 3 * np.pi, height)
        X, Y = np.meshgrid(x, y)

        noise = (np.sin(X) + np.sin(Y) + np.sin(X * 2 + Y) * 0.5)
        noise = noise + np.random.randn(*noise.shape) / 30

        noise = ((noise - noise.min()) / (noise.max() - noise.min()) * 255).astype(np.uint8) // 2

        fog_surf = pygame.Surface(size, pygame.SRCALPHA)
        fog_surf.fill((200, 200, 220))

        pygame.surfarray.pixels_alpha(fog_surf)[...] = noise.transpose(-1, -2)
        return fog_surf

    @staticmethod
    def fog(surface: pygame.Surface, density=0.4, speed=0.5, quality=EFFECTS_QUALITY):
        if quality <= 0: return

        w, h = surface.get_size()

        if PostProcessing._fog_noise is None or PostProcessing._fog_noise.get_size() != (w, h):
            PostProcessing._fog_noise = PostProcessing._generate_fog_noise((w, h))

        time_val = pygame.time.get_ticks() / 400.0 * speed

        offset_x1 = int(time_val * 50) % w
        offset_x2 = int(time_val * -20) % w

        fog_layer = PostProcessing._fog_noise
        fog_layer.set_alpha(int(255 * density))

        surface.blit(fog_layer, (offset_x1 - w, 0))
        surface.blit(fog_layer, (offset_x1, 0))

        # fog_layer_2 = pygame.transform.flip(fog_layer, True, False)
        # surface.blit(fog_layer_2, (offset_x2 - w, 0))
        # surface.blit(fog_layer_2, (offset_x2, 0))

    _grain_surface = None
    _grain_params = None

    @staticmethod
    def grain(surface: pygame.Surface, intensity=20, dynamic=True, quality=EFFECTS_QUALITY):
        """
        Adds a film grain / noise overlay.
        intensity: 0-255 (15-30 is usually the 'sweet spot' for 2000s film).
        dynamic: If True, the grain jitters every frame.
        """
        if quality <= 0 or intensity <= 0: return

        size = surface.get_size()
        current_params = (size, intensity)

        # 1. Cache the grain surface so we don't regenerate noise every frame
        if PostProcessing._grain_surface is None or PostProcessing._grain_params != current_params:
            # We create the grain at a lower resolution for performance if quality is low
            scale = [0, 4, 2, 1, 1, 1][quality]
            grain_size = (size[0] // scale, size[1] // scale)

            # Create a grayscale noise array
            # We use 128 as the neutral point so BLEND_RGB_ADD/SUB balances out
            noise = np.random.randint(-intensity, intensity, (grain_size[0], grain_size[1], 3), dtype=np.int16)

            # Create the surface
            grain_surf = pygame.Surface(grain_size)
            # Use a neutral gray background
            grain_surf.fill((128, 128, 128))

            # Add the noise to the gray
            arr = pygame.surfarray.pixels3d(grain_surf).astype(np.int16)
            arr += noise
            np.clip(arr, 0, 255, out=arr)
            pygame.surfarray.blit_array(grain_surf, arr.astype(np.uint8))

            # Scale back up to full size if we downsampled
            if scale > 1:
                PostProcessing._grain_surface = pygame.transform.scale(grain_surf, size)
            else:
                PostProcessing._grain_surface = grain_surf

            PostProcessing._grain_params = current_params

        # 2. Apply the grain
        offset = (0, 0)
        if dynamic:
            # Instead of regenerating noise (slow), we just jitter the existing texture
            # 2000s film grain often "flashes" slightly.
            offset = (np.random.randint(-10, 10), np.random.randint(-10, 10))

        # BLEND_RGB_SUB or BLEND_RGB_MULT works, but for that 'film' look,
        # BLEND_RGB_ADD with a slightly darker base or OVERLAY is best.
        # Here we use standard alpha or soft light simulation:
        surface.blit(PostProcessing._grain_surface, offset, special_flags=pygame.BLEND_RGB_SUB)


class ParticleEmitter:
    def __init__(self, color=(0, 255, 255), count=1000, size=1, g=10, sparsity=0.75, deltax=0, deltay=0):
        self.color = color
        self.data = np.zeros((int(count * PARTICLE_EMISSION_QUALITY), 5))  # [x, y, vx, vy, life]
        self.active = np.zeros(int(count * PARTICLE_EMISSION_QUALITY), dtype=bool)
        self.g = g
        self.sparsity = sparsity
        self.size = size
        self.tracking = None
        self.deltax = deltax
        self.deltay = deltay

    def track(self, obj):
        assert hasattr(obj, 'x') and hasattr(obj, 'y')
        self.tracking = obj

    def emit(self, x=None, y=None, amount=5):
        if x is None and y is None and self.tracking is not None:
            x, y = self.tracking.x + self.deltax, self.tracking.y + self.deltay
        inactive = np.where(~self.active)[0]
        if len(inactive) == 0:
            return

        idx = inactive[:amount]
        actual_count = len(idx)

        jitter = np.random.uniform(-3, 3, (actual_count, 2))
        self.data[idx, 0:2] = [x, y] + jitter
        self.data[idx, 2:4] = np.random.uniform(-150, 150, (actual_count, 2)) * self.sparsity
        self.data[idx, 4] = np.random.uniform(0.5, 1.5, actual_count)
        self.active[idx] = True

    def update(self, dt):
        if self.tracking is not None:
            self.emit()
        if not np.any(self.active): return

        self.data[self.active, 3] += self.g * dt
        self.data[self.active, 0:2] += self.data[self.active, 2:4] * dt

        self.data[self.active, 4] -= dt
        self.active &= (self.data[:, 4] > 0)

    def draw(self, surface, camera, parallax=1):
        indices = np.where(self.active)[0]
        screen_w, screen_h = surface.get_size()
        for i in indices:
            px = int(self.data[i, 0] - camera.x) * parallax
            py = int(self.data[i, 1] - camera.y) * parallax
            size = max(1, int(self.data[i, 4] * 5))

            radius = max(1, int(size * self.size))
            if -radius <= px <= screen_w + radius and -radius <= py <= screen_h + radius:
                color = self.color if self.color != 'random' else (
                    random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                pygame.draw.circle(surface, color, (px * parallax, py * parallax), radius)


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

    def draw(self, surface, camera, parallax=1):
        indices = np.where(self.active)[0]
        screen_w, screen_h = surface.get_size()

        for i in indices:
            px = int(self.data[i, 0] - camera.x) * parallax
            py = int(self.data[i, 1] - camera.y) * parallax

            timer = self.firefly_data[i, 0]
            pulse = (math.sin(timer * 2.0 + self.firefly_data[i, 3]) + 1) / 2

            current_size = max(1, int(self.size * (1 + pulse / 4)))

            color = self.color
            radius = current_size * parallax
            if -radius <= px <= screen_w + radius and -radius <= py <= screen_h + radius:
                color = self.color if self.color != 'random' else (
                    random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                pygame.draw.circle(surface, color, (px * parallax, py * parallax), radius)

            if pulse > 0.7:
                glow_color = (min(255, color[0] + 50), min(255, color[1] + 50), color[2], 100)
                pygame.draw.circle(surface, glow_color, (px, py), current_size * 2)


class Root:
    def __init__(self, w=200, h=300, title="Ortensia Engine", flag=pygame.RESIZABLE | pygame.SCALED | pygame.DOUBLEBUF,
                 icon=None):
        self.screen = pygame.display.set_mode((w, h), flag)
        self.w = w
        self.h = h
        self.title = title
        self.flag = flag
        self.icon = icon
        pygame.display.set_caption(title)
        if icon is not None:
            icon_surf = pygame.image.load(icon).convert_alpha()
            pygame.display.set_icon(icon_surf)

        self.clock = pygame.time.Clock()


class Scene:
    def __init__(self, root):
        self.screen = root.screen

        self.clock = pygame.time.Clock()
        self.main_camera = Camera(width=root.w, height=root.h)
        self.layers: List[Layer] = []
        self.centerx = root.w // 2
        self.centery = root.h // 2
        self.particle_emitters = []
        self.particle_layer_idx = -1
        self.solids = []
        self.grid = SpatialGrid(cell_size=500)
        self.running = True
        self.max_fps = 600
        self.game_div = 1000.0
        self.scale = 1
        self.layer_type = ChunkedLayer
        self.updatables = []
        self.mechaniques = []
        self.renderables = {}  # must be a dict of tuples with (index of layer on top to blit, surf)

        # Simulated 3D
        self.anaglyph_mode = False
        self.stereo_separation = 6.0
        self._left_buffer = None
        self._right_buffer = None

    def c_justified_pos(self, w, h, dx=0, dy=0):
        return (self.centerx - w // 2 + dx, self.centery - h // 2 + dy)

    def add_create_layer(self, name, parallax=1.0, chunk_size=2000, layertype=ChunkedLayer) -> Layer:
        if layertype is None:
            layertype = self.layer_type
        l = layertype(name, parallax, chunk_size=chunk_size)
        self.layers.append(l)
        return l

    def add_layer(self, l):
        self.layers.append(l)

    def _render_anaglyph(self, render_target, layers_to_draw, emitters):
        w, h = render_target.get_size()

        if self._left_buffer is None or self._left_buffer.get_size() != (w, h):
            self._left_buffer = pygame.Surface((w, h))
            self._right_buffer = pygame.Surface((w, h))

        original_cam_x = self.main_camera.x

        self.main_camera.x = original_cam_x - (self.stereo_separation / 2)

        self._left_buffer.fill((20, 20, 30))
        for i, layer in enumerate(layers_to_draw):
            if self.particle_layer_idx != -1 and self.particle_layer_idx == i:
                layer.render(self._left_buffer, self.main_camera, emitters=emitters)
            else:
                layer.render(self._left_buffer, self.main_camera)

        if self.particle_layer_idx == -1:
            for emitter in emitters:
                emitter.draw(self._left_buffer, self.main_camera)

        self._left_buffer.fill((255, 0, 0), special_flags=pygame.BLEND_MULT)
        self.main_camera.x = original_cam_x + (self.stereo_separation / 2)

        self._right_buffer.fill((20, 20, 30))
        for i, layer in enumerate(layers_to_draw):
            if self.particle_layer_idx != -1 and self.particle_layer_idx == i:
                layer.render(self._right_buffer, self.main_camera, emitters=emitters)
            else:
                layer.render(self._right_buffer, self.main_camera)

        if self.particle_layer_idx == -1:
            for emitter in emitters:
                emitter.draw(self._right_buffer, self.main_camera)

        self._right_buffer.fill((0, 255, 255), special_flags=pygame.BLEND_MULT)

        render_target.blit(self._left_buffer, (0, 0))
        render_target.blit(self._right_buffer, (0, 0), special_flags=pygame.BLEND_ADD)

        self.main_camera.x = original_cam_x

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
            self.main_camera.update()

            for emitter in self.particle_emitters:
                emitter.update(dt)

            self.screen.fill((20, 20, 30))

            for i, layer in enumerate(self.layers):
                if self.particle_layer_idx != -1 and self.particle_layer_idx == i:
                    layer.render(self.screen, self.main_camera, emitters=self.particle_emitters)
                else:
                    layer.render(self.screen, self.main_camera)

                if hasattr(layer, 'process_events'):
                    layer.process_events(event)

            if self.particle_layer_idx == -1:
                for emitter in self.particle_emitters:
                    emitter.draw(self.screen, self.main_camera)

            fps = self.clock.get_fps()
            pygame.display.set_caption(f"Ortensia | FPS: {int(fps)}")

            pygame.display.flip()

    def add_renderable(self, renderable, index):
        if index in self.renderables:
            self.renderables[index].append(renderable)
        else:
            self.renderables[index] = [renderable]

    def update(self):
        dt = self.clock.tick(self.max_fps) / self.game_div
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                self.anaglyph_mode = not self.anaglyph_mode
                flag("Activated Anaglyph 3d mode")

            for layer in self.layers:  # TODO: This can be probably optimized
                if hasattr(layer, 'process_events'):
                    layer.process_events(event)

        self.grid.clear()
        for obj in self.solids:
            self.grid.insert(obj)

        self.main_camera.update()
        for updt in self.mechaniques:
            updt.mechaniches(pygame.key.get_pressed(), dt)
        for updt in self.updatables:
            updt.update(dt)

        for emitter in self.particle_emitters:
            emitter.update(dt)

        if self.anaglyph_mode:
            self._render_anaglyph(self.screen, self.layers, self.particle_emitters)
        else:
            self.screen.fill((20, 20, 30))

            for i, layer in enumerate(self.layers):
                if self.particle_layer_idx != -1 and self.particle_layer_idx == i:
                    layer.render(self.screen, self.main_camera, emitters=self.particle_emitters)
                else:
                    layer.render(self.screen, self.main_camera)

                if hasattr(layer, 'update'):
                    layer.update(dt)

                for renderable in self.renderables.get(i, []):
                    renderable.render(self.screen, self.main_camera)
                for renderable in self.renderables.get(-len(self.layers) + i, []):
                    renderable.render(self.screen, self.main_camera)

        if self.particle_layer_idx == -1:
            for emitter in self.particle_emitters:
                emitter.draw(self.screen, self.main_camera)

        fps = self.clock.get_fps()
        pygame.display.set_caption(f"Ortensia | FPS: {int(fps)}")


if __name__ == "__main__":
    def s(x):
        return int(x * 1)


    game = Scene(s(1000), s(600), flag=pygame.SCALED | pygame.RESIZABLE)
    bg2 = game.add_create_layer("Background2", 0.2)
    bg = game.add_create_layer("Background", 0.5)
    particles = game.add_create_layer("particles", 1.0)
    fg = game.add_create_layer("Foreground", 1.0)

    # fg.add_effect(PostProcessing.underwater_distortion, 5)
    # particles.add_effect(PostProcessing.lumen, 20, 3)
    fg.add_effect(PostProcessing.fog, 0.1, 10)
    from functions import *

    # player = SolidSprite(s(400), s(300), s(40), s(40), (255, 255, 255))
    player = AnimatedSolidSprite(s(400), s(300), s(64), s(64))
    player.add_animation('idle', load_spritesheet("examples/AuryRunning.png", 64, 64, row=0))
    fg.sprites.append(player)
    water = FluidSprite(s(200), s(500), s(600), s(100), color=(50, 100, 255, 120))
    fg.sprites.append(water)
    game.solids.append(player)
    game.main_camera.target = player

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
            game.solids.append(wall)
        else:
            wall = Sprite(i * s(400), s(450), s(60), s(150), (30, 70, 40))
            fg.sprites.append(wall)

    terrain_layer = game.add_create_layer("Terrain", 1.0)
    map_system = TileMap(game, terrain_layer, tile_size=s(40), texture='../examples/Ortensia1.png')


    def draw_builder_cursor(screen, tile_map, camera):
        mx, my = pygame.mouse.get_pos()
        gx, gy = tile_map.get_grid_pos(mx, my)

        cam_x = camera.x * tile_map.layer.parallax
        cam_y = camera.y * tile_map.layer.parallax

        rect_x = (gx * tile_map.tile_size) - cam_x
        rect_y = (gy * tile_map.tile_size) - cam_y

        cursor_surf = pygame.Surface((tile_map.tile_size, tile_map.tile_size), pygame.SRCALPHA)
        cursor_surf.fill((255, 255, 255, 100))
        pygame.draw.rect(cursor_surf, (255, 255, 255), (0, 0, tile_map.tile_size, tile_map.tile_size), 2)  # Border

        screen.blit(cursor_surf, (rect_x, rect_y))


    def my_update(game_inst, dt):
        keys = pygame.key.get_pressed()
        speed = 900 * dt * 0.5

        dx, dy = 0, 0
        if keys[pygame.K_LEFT]:  dx -= speed
        if keys[pygame.K_RIGHT]: dx += speed
        if keys[pygame.K_UP]:    dy -= speed
        if keys[pygame.K_DOWN]:  dy += speed
        if keys[pygame.K_SPACE]:
            game.main_camera.shake_intensity = 5

        player.move(dx, dy, game_inst.grid)
        player.update_animation(dt)
        water.update(interactors=[player])

        emitter1.emit(player.x + s(20), player.y + s(20))

        mouse_buttons = pygame.mouse.get_pressed()
        mx, my = pygame.mouse.get_pos()

        if mouse_buttons[0]:
            map_system.place_tile(mx, my, color=(100, 200, 100))

        if mouse_buttons[2]:
            map_system.remove_tile(mx, my)

        # draw_builder_cursor(game_inst.screen, map_system, game_inst.main_camera)


    game.run(my_update)
