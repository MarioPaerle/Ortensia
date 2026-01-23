import pygame
from typing import List, Any, Optional
import numpy as np
import copy
from Graphic.functions import load_spritesheet


class Sprite:
    def __init__(self, x, y, w, h, color=(255, 255, 255), texture=None, alpha=False):
        self.x, self.y, self.width, self.height = x, y, w, h
        self.color = color
        self.texture_path = texture
        self.alpha = alpha

        self._load_surface()

    def _load_surface(self):
        flags = pygame.SRCALPHA if self.alpha else 0
        self.surface = pygame.Surface((self.width, self.height), flags)

        if self.texture_path:
            if self.alpha:
                tex = pygame.image.load(self.texture_path).convert_alpha()
            else:
                tex = pygame.image.load(self.texture_path).convert()

            tex = pygame.transform.scale(tex, (self.width, self.height))
            self.surface.blit(tex, (0, 0))
        else:
            self.surface.fill(self.color)

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'surface' in state:
            del state['surface']
        if 'texture' in state:
            del state['texture']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load_surface()


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

        self.cw = w if cw is None else cw
        self.ch = h if ch is None else ch

        self.coffset_x = coffset_x
        self.coffset_y = coffset_y

        self.frect = pygame.FRect(x + coffset_x, y + coffset_y, self.cw, self.ch)

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

        self.x = self.frect.x - self.coffset_x
        self.y = self.frect.y - self.coffset_y


class AnimationLoader:
    def __init__(self, file, w, h, row=0, scale=(1, 1)):
        self.file = file
        self.w = w
        self.h = h
        self.row = row
        self.scale = scale
        self.frames = None

    def get_frames(self):
        if self.frames is None:
            self.frames = load_spritesheet(self.file, self.w, self.h, row=self.row, scale=self.scale)
        return self.frames

    def __getstate__(self):
        state = self.__dict__.copy()
        state['frames'] = None
        return state


class AnimatedSolidSprite(SolidSprite):
    """Solid sprite with animations - combines physics with animation and supports pickling."""

    def __init__(self, x, y, w, h, color=(255, 255, 255), texture=None, alpha=False,
                 cw=None, ch=None, coffset_x=0, coffset_y=0):

        super().__init__(x, y, w, h, color, texture, alpha,
                         cw=cw, ch=ch, coffset_x=coffset_x, coffset_y=coffset_y)

        self.animations = {}
        self.anim_loaders = {}

        # --- ORIGINAL STATE ---
        self.current_state = "idle"
        self.frame_index = 0.0
        self.animation_speed = 10.0

    def add_animation(self, name: str, loader):
        """
        Registers an animation using a loader.
        The loader is saved; the surfaces are generated immediately for use.
        """
        self.anim_loaders[name] = loader
        self.animations[name] = loader.get_frames()

    def update_animation(self, dt):
        """Advances the frame index based on time."""
        if self.current_state in self.animations:
            self.frame_index += self.animation_speed * dt

            frames = self.animations[self.current_state]
            if self.frame_index >= len(frames):
                self.frame_index = 0

            current_frame = int(self.frame_index)
            current_frame = current_frame % len(frames)

            base_frame = frames[current_frame]

            if hasattr(self, 'show_hitboxes') and self.show_hitboxes:
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

        self.x = self.frect.x - self.coffset_x
        self.y = self.frect.y - self.coffset_y
        return collide

    def setpos(self, x, y):
        """Movement with collision detection (inherited from SolidSprite)"""
        collide = 'none'

        self.frect.x = x + self.coffset_x
        self.frect.y = y + self.coffset_y

        self.x = x
        self.y = y
        return collide

    def __getstate__(self):
        state = super().__getstate__() if hasattr(super(), '__getstate__') else self.__dict__.copy()
        if 'animations' in state:
            del state['animations']

        if 'surface' in state:
            del state['surface']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.animations = {}

        for name, loader in self.anim_loaders.items():
            self.animations[name] = loader.get_frames()

        if self.current_state in self.animations:
            frames = self.animations[self.current_state]
            if frames:
                idx = int(self.frame_index) % len(frames)
                base_frame = frames[idx]

                if hasattr(self, 'show_hitboxes') and self.show_hitboxes:
                    self.surface = base_frame.copy()
                    self.draw_hitbox()
                else:
                    self.surface = base_frame
            else:
                self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        else:
            self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)


class Block(SolidSprite):
    """
    A template block that clones itself.
    Inherits 'move' from SolidSprite, so it has physics too!
    """

    def __init__(
            self,
            w,
            h,
            id,
            texture=None,
            alpha=False,
            emitter=None,
            physics=False,
            speed_multiplier=1.0,
            bounce_multiplier=1.0,
            stickyness=0.0,
            hardness=0.0,
            light_emission_intensity=0.0,
            light_emission_color=(255, 255, 255),
            gravity=980,
            max_fall_speed=1000,
    ):
        super().__init__(
            -100,
            -100,
            w,
            h,
            (100, 100, 100),
            texture=texture,
            alpha=alpha,
        )

        # Identity
        self.id = id
        self.name = "Generic Block"

        self.emitter = emitter
        self.physics_block = physics

        self.speed_multiplier = speed_multiplier
        self.bounce_multiplier = bounce_multiplier
        self.stickyness = stickyness
        self.hardness = hardness

        # Lighting
        self.light_emission_intensity = light_emission_intensity
        self.light_emission_color = light_emission_color

        # Physics state
        self.velocity_y = 0.0
        self.gravity = gravity
        self.is_grounded = False
        self.max_fall_speed = max_fall_speed

    def clone(self):
        new_obj = copy.copy(self)
        new_obj.frect = self.frect.copy()

        new_obj.velocity_y = 0
        new_obj.is_grounded = False
        return new_obj

    def place(self, x, y):
        new_block = self.clone()

        new_block.frect.topleft = (x, y)
        new_block.x = x
        new_block.y = y

        return new_block

    def update_physics(self, dt, grid):
        if not self.physics_block:
            return False

        self.velocity_y += self.gravity * dt

        if self.velocity_y > self.max_fall_speed:
            self.velocity_y = self.max_fall_speed

        dy = self.velocity_y * dt

        if abs(dy) < 0.01:
            self.is_grounded = True
            return False

        old_y = self.frect.y

        self.frect.y += dy

        collision = False
        for other in grid.get_nearby(self.frect):
            if other is not self and self.frect.colliderect(other.frect):
                if dy > 0:
                    self.frect.bottom = other.frect.top
                    self.velocity_y = 0
                    self.is_grounded = True
                    collision = True
                elif dy < 0:
                    self.frect.top = other.frect.bottom
                    self.velocity_y = 0
                    collision = True

        self.y = self.frect.y

        moved = abs(self.y - old_y) > 0.01

        if not collision and moved:
            self.is_grounded = False

        return moved

    def __repr__(self):
        return f'<Block {self.id} at ({self.x}, {self.y})>'


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
