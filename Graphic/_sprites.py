import pygame
from typing import List, Any, Optional
import numpy as np


class Sprite:
    def __init__(self, x, y, w, h, color=(255, 255, 255), texture=None, alpha=False):
        self.x, self.y, self.width, self.height = x, y, w, h

        flags = pygame.SRCALPHA if alpha else 0
        self.surface = pygame.Surface((w, h), flags)

        if texture is not None and alpha:
            self.texture = pygame.image.load(texture).convert_alpha()
            self.texture = pygame.transform.scale(self.texture, (w, h))
            self.surface.blit(self.texture, (0, 0))

        elif texture is not None and not alpha:
            self.texture = pygame.image.load(texture).convert()
            self.surface.blit(self.texture, (0, 0))
        else:
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
        self.frect.x += dx
        for other in grid.get_nearby(self.frect):
            if other is not self and self.frect.colliderect(other.frect):
                if dx > 0: self.frect.right = other.frect.left
                if dx < 0: self.frect.left = other.frect.right

        self.frect.y += dy
        for other in grid.get_nearby(self.frect):
            if other is not self and self.frect.colliderect(other.frect):
                if dy > 0: self.frect.bottom = other.frect.top
                if dy < 0: self.frect.top = other.frect.bottom

        self.x, self.y = self.frect.x, self.frect.y

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
    def __init__(self, x, y, w, h, color=(255, 255, 255), texture=None, alpha=False):
        super().__init__(x, y, w, h, color, texture=texture, alpha=alpha)
        self.frect = pygame.FRect(x, y, w, h)
        
    def move(self, dx, dy, grid):
        self.frect.x += dx
        for other in grid.get_nearby(self.frect):
            if other is not self and self.frect.colliderect(other.frect):
                if dx > 0: self.frect.right = other.frect.left
                if dx < 0: self.frect.left = other.frect.right

        self.frect.y += dy
        for other in grid.get_nearby(self.frect):
            if other is not self and self.frect.colliderect(other.frect):
                if dy > 0: self.frect.bottom = other.frect.top
                if dy < 0: self.frect.top = other.frect.bottom

        self.x, self.y = self.frect.x, self.frect.y


class FluidSprite(Sprite):
    def __init__(self, x, y, w, h, resolution=4, color=(50, 150, 255, 150)):
        self.headroom = h // 2
        super().__init__(x, y - self.headroom, w, h + self.headroom)

        self.color = color
        self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        self.k = 0.015
        self.damp = 0.015
        self.spread = 0.15

        self.count = w // resolution
        self.column_width = resolution

        self.sea_level = self.headroom
        self.target_y = np.full(self.count, self.sea_level, dtype=np.float16)
        self.curr_y = np.full(self.count, self.sea_level, dtype=np.float16)
        self.vel = np.zeros(self.count, dtype=np.float16)

    def splash(self, x_pos, velocity):
        local_x = x_pos - self.x
        idx = int(local_x // self.column_width)
        if 0 <= idx < self.count:
            self.vel[idx] += velocity

    def get_height_at(self, world_x):
        """Returns the world-Y coordinate of the water surface at a given X."""
        local_x = world_x - self.x
        idx = int(local_x // self.column_width)
        if 0 <= idx < self.count:
            return self.curr_y[idx] + self.y
        return self.y + self.sea_level

    def update(self, interactors: List[Any] = None):
        force = -self.k * (self.curr_y - self.target_y) - self.damp * self.vel
        self.vel += force
        self.curr_y += self.vel

        left_deltas = np.zeros_like(self.curr_y)
        right_deltas = np.zeros_like(self.curr_y)
        left_deltas[1:] = self.spread * (self.curr_y[1:] - self.curr_y[:-1])
        right_deltas[:-1] = self.spread * (self.curr_y[:-1] - self.curr_y[1:])
        self.vel -= left_deltas
        self.vel -= right_deltas

        if interactors:
            for obj in interactors:
                if self.x < obj.x + obj.width / 2 < self.x + self.width:
                    water_line = self.get_height_at(obj.x + obj.width / 2)
                    obj_bottom = obj.y + obj.height

                    if obj_bottom > water_line:
                        if abs(obj_bottom - water_line) < 10:
                            self.splash(obj.x + obj.width / 2, 5)

                        depth = obj_bottom - water_line
                        buoyancy_force = depth * 0.5

                        if hasattr(obj, 'frect'):
                            pass

        self._draw_fluid()

    def _draw_fluid(self):
        self.surface.fill((0, 0, 0, 0))
        x_coords = np.arange(0, self.count) * self.column_width

        top_points = np.stack((x_coords, self.curr_y), axis=-1)
        bottom_right = [[self.width, self.height]]
        bottom_left = [[0, self.height]]
        full_poly = np.concatenate([top_points, bottom_right, bottom_left])

        pygame.draw.polygon(self.surface, self.color, full_poly.tolist())

        if len(top_points) > 1:
            pygame.draw.lines(self.surface, (255, 255, 255), False, top_points.tolist(), 2)


class AnimatedSolidSprite(SolidSprite):
    def __init__(self, x, y, w, h, gw=None, gh=None):
        super().__init__(x, y, w, h, (0, 0, 0, 0))
        self.animations = {}
        self.current_state = 'idle'
        self.frame_index = 0.0
        self.animation_speed = 12.0
        self.frect = pygame.FRect(x, y, w, h)
        self.gw = gw if gw is not None else w
        self.gh = gh if gh is not None else h

    def add_animation(self, name, frames):
        scaled_frames = []
        for f in frames:
            scaled_f = pygame.transform.scale(f, (self.gw, self.gh))
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
