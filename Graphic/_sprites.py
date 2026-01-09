import pygame
from typing import List, Any, Optional
import numpy as np
import copy


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
            self.texture = pygame.transform.scale(self.texture, (w, h))

            self.surface.blit(self.texture, (0, 0))
        else:
            self.surface.fill(color)

    def move(self, dx, dy):
        """Simple movement without collision detection"""
        self.x += dx
        self.y += dy


class AnimatedSprite(Sprite):
    """Non-solid animated sprite that doesn't interact with physics"""

    def __init__(self, x, y, w, h, cw=None, ch=None):
        super().__init__(x, y, w, h)
        self.animations = {}
        self.current_state = "idle"
        self.frame_index = 0.0
        self.animation_speed = 10.0
        self.show_hitboxes = False

        # Create a frect for visual reference, but it won't be used for collision
        self.frect = pygame.FRect(x, y, w if cw is None else cw, h if ch is None else ch)

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
            base_frame = self.animations[self.current_state][current_frame]

            if self.show_hitboxes:
                # Create a copy so we don't modify the original animation frame
                self.surface = base_frame.copy()
                hitbox_rect = pygame.Rect(
                    self.frect.x - self.x,
                    self.frect.y - self.y,
                    self.frect.width,
                    self.frect.height
                )
                pygame.draw.rect(self.surface, (255, 255, 255), hitbox_rect, width=2)
            else:
                self.surface = base_frame

    def set_state(self, state: str):
        """Change animation state"""
        if self.current_state != state:
            self.current_state = state
            self.frame_index = 0.0

    def move(self, dx, dy):
        """Simple movement without collision - updates both x,y and frect"""
        self.x += dx
        self.y += dy
        self.frect.x = self.x
        self.frect.y = self.y


class SpatialGrid:
    def __init__(self, cell_size=128):
        self.cell_size = cell_size
        self.cells = {}

    def clear(self):
        self.cells.clear()

    def insert(self, sprite):
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
    """Sprite with collision detection"""

    def __init__(self, x, y, w, h, color=(255, 255, 255), texture=None, alpha=False,
                 cw=None, ch=None, coffset_x=0, coffset_y=0):
        super().__init__(x, y, w, h, color, texture=texture, alpha=alpha)

        # Collision box dimensions
        self.cw = w if cw is None else cw
        self.ch = h if ch is None else ch

        # Collision box offset from sprite position
        self.coffset_x = coffset_x
        self.coffset_y = coffset_y

        # Create collision rect with offset
        self.frect = pygame.FRect(x + coffset_x, y + coffset_y, self.cw, self.ch)

        # Visual hitbox settings
        self.show_hitboxes = False
        self.hitbox_color = (255, 255, 255)
        self.hitbox_width = 2

    def set_hitbox(self, width=None, height=None, offset_x=None, offset_y=None):
        """Dynamically adjust hitbox size and offset"""
        if width is not None:
            self.cw = width
        if height is not None:
            self.ch = height
        if offset_x is not None:
            self.coffset_x = offset_x
        if offset_y is not None:
            self.coffset_y = offset_y

        # Update frect with new dimensions and offset
        self.frect.width = self.cw
        self.frect.height = self.ch
        self.frect.x = self.x + self.coffset_x
        self.frect.y = self.y + self.coffset_y

    def draw_hitbox(self, surface=None):
        """Draw hitbox visualization on surface (or self.surface if none provided)"""
        if not self.show_hitboxes:
            return

        target = surface if surface is not None else self.surface

        # Calculate hitbox position relative to sprite surface
        hitbox_rect = pygame.Rect(
            self.coffset_x,
            self.coffset_y,
            self.cw,
            self.ch
        )
        pygame.draw.rect(target, self.hitbox_color, hitbox_rect, width=self.hitbox_width)

    def move(self, dx, dy, grid):
        """Movement with collision detection"""
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

        # Update sprite position to match collision box (accounting for offset)
        self.x = self.frect.x - self.coffset_x
        self.y = self.frect.y - self.coffset_y


class AnimatedSolidSprite(SolidSprite):
    """Solid sprite with animations - combines physics with animation"""

    def __init__(self, x, y, w, h, color=(255, 255, 255), texture=None, alpha=False,
                 cw=None, ch=None, coffset_x=0, coffset_y=0):

        super().__init__(x, y, w, h, color, texture=texture, alpha=alpha,
                         cw=cw, ch=ch, coffset_x=coffset_x, coffset_y=coffset_y)
        self.animations = {}
        self.current_state = "idle"
        self.frame_index = 0.0
        self.animation_speed = 10.0

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
            base_frame = self.animations[self.current_state][current_frame]

            if self.show_hitboxes:
                # Create a copy so we don't modify the original animation frame
                self.surface = base_frame.copy()
                self.draw_hitbox()
            else:
                self.surface = base_frame

    def set_state(self, state: str):
        """Change animation state"""
        if self.current_state != state:
            self.current_state = state
            self.frame_index = 0.0

    def move(self, dx, dy, grid):
        """Movement with collision detection (inherited from SolidSprite)"""
        # Store collision info
        collide = 'none'

        self.frect.x += dx
        for other in grid.get_nearby(self.frect):
            if other is not self and self.frect.colliderect(other.frect):
                if dx > 0:
                    self.frect.right = other.frect.left
                    collide = 'r'
                if dx < 0:
                    self.frect.left = other.frect.right
                    collide = 'l'

        self.frect.y += dy
        for other in grid.get_nearby(self.frect):
            if other is not self and self.frect.colliderect(other.frect):
                if dy > 0:
                    self.frect.bottom = other.frect.top
                    collide = 'b'
                if dy < 0:
                    self.frect.top = other.frect.bottom
                    collide = 'u'

        # Update sprite position to match collision box (accounting for offset)
        self.x = self.frect.x - self.coffset_x
        self.y = self.frect.y - self.coffset_y
        return collide


class Block(SolidSprite):
    """
    A template block that clones itself.
    Inherits 'move' from SolidSprite, so it has physics too!
    """

    def __init__(self, w, h, id, texture=None, alpha=False):
        super().__init__(-100, -100, w, h, (100, 100, 100), texture=texture, alpha=alpha)

        self.id = id
        self.name = 'Generic Block'

        self.emitter = None
        self.phisic_block = False
        self.speed_multiplier = 1
        self.bounce_multiplier = 1
        self.stickyness = 0
        self.hardness = 0
        self.light_emission_intensity = 0
        self.light_emission_color = (255, 255, 255)

    def clone(self):
        new_obj = copy.copy(self)
        new_obj.frect = self.frect.copy()

        return new_obj

    def place(self, x, y):
        """
        Clones the template and teleports it to (x, y).
        """
        new_block = self.clone()

        new_block.frect.topleft = (x, y)
        new_block.x = x
        new_block.y = y

        return new_block

    def __repr__(self):
        return f'<Block {self.id} at {self.frect.topleft}>'


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