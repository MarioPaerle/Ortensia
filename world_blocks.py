from basic import Block


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



