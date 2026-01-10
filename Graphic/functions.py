import pygame
import numpy as np
from colorama import Fore

RED = Fore.RED
BLUE = Fore.BLUE
GREEN = Fore.LIGHTGREEN_EX
RES = Fore.RESET
YEL = Fore.YELLOW


def flag(message, level=1):
    if level == 0:
        print(f"{BLUE}|Success|: {message}{RES}")
    if level == 1:
        print(f"{BLUE}|Ortensia|: {message}{RES}")
    if level == 2:
        print(f"{YEL}|Warning|: {message}{RES}")
    if level == 3:
        print(f"{RED}|Error|: {message}{RES}")

def load_spritesheet(filename, frame_w, frame_h, row='all', scale=(1, 1)):
    sheet = pygame.image.load(filename).convert_alpha()
    frames = []

    if row == 'all':
        for y in range(0, sheet.get_height(), frame_h):
            for x in range(0, sheet.get_width(), frame_w):
                frame = sheet.subsurface(pygame.Rect(x, y, frame_w, frame_h))
                frames.append(frame)
    else:
        for x in range(0, sheet.get_width(), frame_w):
            frame = sheet.subsurface(pygame.Rect(x, row * frame_h, frame_w, frame_h))
            frames.append(frame)

    return [pygame.transform.scale_by(f, factor=scale) for f in frames]


def scale_color(color, factor):
    return [min(255, max(0, int(c * factor))) for c in color]


def show_surface(surface):
    from PIL import Image
    img_data = pygame.image.tobytes(surface, 'RGBA')
    img = Image.frombytes('RGBA', surface.get_size(), img_data)
    img.show()


# Cache so we don't regenerate the noise every frame
_grain_cache = None


def add_grain(surface, intensity=12, dynamic=True, color=(255, 255, 255)):
    global _grain_cache

    w, h = surface.get_size()
    cache_w, cache_h = w + 20, h + 20  # Oversized for shake effect

    # Generate grain texture once
    if _grain_cache is None or _grain_cache.get_size() != (cache_w, cache_h):
        grain_surf = pygame.Surface((cache_w, cache_h), pygame.SRCALPHA)
        noise = np.random.randint(0, intensity, (cache_w, cache_h), dtype=np.uint8)

        pygame.surfarray.pixels3d(grain_surf)[...] = color
        pygame.surfarray.pixels_alpha(grain_surf)[...] = noise
        _grain_cache = grain_surf

    # Random offset creates the film flicker
    if dynamic:
        x_off = np.random.randint(-20, 0)
        y_off = np.random.randint(-20, 0)
        surface.blit(_grain_cache, (x_off, y_off))
    else:
        surface.blit(_grain_cache, (-10, -10))


_vignette_cache = None


def add_vignette(surface, intensity=0.5):
    global _vignette_cache
    w, h = surface.get_size()

    if _vignette_cache is None or _vignette_cache.get_size() != (w, h):
        # Build radial gradient
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        x, y = np.meshgrid(x, y)

        dist = np.sqrt(x ** 2 + y ** 2)
        mask = np.clip(1 - dist * intensity, 0, 1)
        mask = (mask * 255).astype(np.uint8).T

        vignette_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        alpha_channel = 255 - mask
        pygame.surfarray.pixels3d(vignette_surf)[...] = (20, 10, 20)
        pygame.surfarray.pixels_alpha(vignette_surf)[...] = alpha_channel

        _vignette_cache = vignette_surf

    surface.blit(_vignette_cache, (0, 0))


def apply_fisheye(surface, strength=0.1):
    """Wide-angle lens distortion effect"""
    width, height = surface.get_size()
    src = pygame.surfarray.array3d(surface)

    # Build coordinate grid
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xv, yv = np.meshgrid(x, y)

    r = np.sqrt(xv ** 2 + yv ** 2)
    dr = 1 + strength * r ** 2

    new_x = (xv * dr + 1) * width / 2
    new_y = (yv * dr + 1) * height / 2

    new_x = np.clip(new_x, 0, width - 1).astype(int).T
    new_y = np.clip(new_y, 0, height - 1).astype(int).T

    dst = src[new_x, new_y]
    pygame.surfarray.blit_array(surface, dst)


_scanline_surf = None


def apply_scanlines(surface, opacity=40):
    global _scanline_surf
    w, h = surface.get_size()

    if _scanline_surf is None:
        _scanline_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        for y in range(0, h, 2):
            pygame.draw.line(_scanline_surf, (0, 0, 0, opacity), (0, y), (w, y))

    surface.blit(_scanline_surf, (0, 0))


def apply_aberration(surface, offset=2):
    """Chromatic aberration - color fringing at edges"""
    arr = pygame.surfarray.pixels3d(surface)

    red = arr[:, :, 0]
    blue = arr[:, :, 2]

    # Shift red and blue channels in opposite directions
    arr[:, :, 0] = np.roll(red, -offset, axis=0)
    arr[:, :, 2] = np.roll(blue, offset, axis=0)

    del arr


_lut_cache = None


def create_lut_map(filter_type="warm"):
    """Pre-compute color grading lookup table"""
    vals = np.arange(256, dtype=np.uint8)

    if filter_type == "warm":
        r = np.clip(vals * 1.1, 0, 255).astype(np.uint8)
        g = np.clip(vals * 1.05, 0, 255).astype(np.uint8)
        b = np.clip(vals * 0.9, 0, 255).astype(np.uint8)
    elif filter_type == "faded_pastel":
        r = np.clip(vals * 0.8 + 30, 0, 255).astype(np.uint8)
        g = np.clip(vals * 0.8 + 35, 0, 255).astype(np.uint8)
        b = np.clip(vals * 0.8 + 40, 0, 255).astype(np.uint8)

    return r, g, b


def add_red_blue_shifts(surface, offset):
    """
    Anaglyphic 3D effect
    offset > 0: pushes layer back
    offset < 0: pulls layer forward
    offset = 0: neutral plane
    """
    target = surface.copy()
    arr = pygame.surfarray.pixels3d(target)

    if offset != 0:
        arr[:, :, 0] = np.roll(arr[:, :, 0], -offset, axis=0)
        arr[:, :, 1:] = np.roll(arr[:, :, 1:], offset, axis=0)

    del arr
    return target


def apply_lut(surface, lut_maps):
    """Apply color grading using lookup tables"""
    r_map, g_map, b_map = lut_maps
    arr = pygame.surfarray.pixels3d(surface)

    arr[:, :, 0] = r_map[arr[:, :, 0]]
    arr[:, :, 1] = g_map[arr[:, :, 1]]
    arr[:, :, 2] = b_map[arr[:, :, 2]]

    del arr
