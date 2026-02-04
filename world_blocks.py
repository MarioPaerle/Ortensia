from basic import Block
import pygame


class Plant(Block):
    pass


class GuiBlock(Block):
    pass


class Deathblock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def on_touch(self, other, dt=1):
        if hasattr(other, 'life'):
            other.take_damage(float('inf'))


class Chest(Block):
    def __init__(self, w, h, id, textures, *args, **kwargs):
        self.textures = [pygame.transform.scale(pygame.image.load(t).convert_alpha(), (w, h)) for t in textures]
        self.state = False # Closed
        super().__init__(w, h, id, texture=textures[1], *args, **kwargs)
        self.surface = self.textures[0]

    def on_click(self, player):
        self.state = not self.state
        self.surface = self.textures[self.state]



