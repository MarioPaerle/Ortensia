import pygame

pygame.init()

from Graphic.base import *
from Graphic.functions import *
import random


def default_scaler(x):
    return int(x * 1)


class Game(Root):
    def __init__(self, w, h, title, flag=pygame.RESIZABLE | pygame.DOUBLEBUF, icon=None):
        super().__init__(w, h, title=title, flag=flag, icon=icon)
        if True:
            gui = UILayer()
            _scene0 = Scene(self)
            _scene0.add_layer(gui)

            def close_game():
                _scene0.running = False
                exit()

            def play_game():
                # self.set_scene('1')
                self.set_scene('save_selector')

            playbutton = UIButton(*_scene0.c_justified_pos(150, 50, dy=100), text='Close Game')
            playbutton2 = UIButton(*_scene0.c_justified_pos(150, 50, dy=0), text='Singleplayer')
            playbutton.on_click = close_game
            playbutton2.on_click = play_game
            gui.add_element(playbutton)
            gui.add_element(playbutton2)

        self.loaded_scenes = {'0': _scene0}
        self.active_scene_name = '0'
        self.state = 0

    def update(self):
        self.loaded_scenes[self.active_scene_name].update()
        if not self.loaded_scenes[self.active_scene_name].running:
            exit()
        pygame.display.flip()

    def addscene(self, scene, name):
        self.loaded_scenes[name] = scene

    def set_scene(self, name):
        if name in self.loaded_scenes:
            self.active_scene_name = name
            self.loaded_scenes[name].clock.tick()
        else:
            flag("Trying to load a non existing scene", level=3)


class Engine(Scene):
    def __init__(
            self,
            name='Ortensia',
            base_size=(1000, 600),
            flag=pygame.SCALED | pygame.RESIZABLE | pygame.HWSURFACE,
            scaler=default_scaler,

    ):
        self.name = name
        self.base_size = base_size
        self.flag = flag
        self.scaler = scaler
        self.w, self.h = base_size[0], base_size[1]
        self.sw, self.sh = scaler(self.w), scaler(self.h)
        self.asset_folder = ''
        root = Root(self.sw, self.sh, title=name, flag=flag)
        super().__init__(root)
        self.cameras = [self.main_camera]
        self.camera_id = 0
        self.updaters = []
        self.player = None
        self.g = 9.81
        self.lut = create_lut_map("warm")
        self.map_system = None

    def dt(self):
        return self.clock.tick(self.max_fps) / self.game_div

    def refresh_grid(self):
        self.grid.clear()
        for obj in self.solids:
            self.grid.insert(obj)

    def update(self):
        dt = self.dt()
        camera = self.cameras[self.camera_id]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            for layer in self.layers:  # TODO: This can be probably optimized
                if hasattr(layer, 'process_events'):
                    layer.process_events(event)

        self.mechaniches(dt)
        camera.update()

        self.screen.fill((20, 20, 30))

        for i, layer in enumerate(self.layers):
            if self.particle_layer_idx != -1 and self.particle_layer_idx == i:
                layer.render(self.screen, camera, emitters=self.particle_emitters)
            else:
                layer.render(self.screen, camera)

            if layer.emitters is not None:
                for emitter in layer.emitters:
                    emitter.update(dt)

            if hasattr(layer, 'update'):
                layer.update(dt)

        fps = self.clock.get_fps()
        # apply_lut(self.screen, self.lut)
        add_grain(self.screen, 10, dynamic=True)
        add_vignette(self.screen, 0.33)
        pygame.display.set_caption(f"{self.name} | FPS: {int(fps)}")
        pygame.display.flip()

    def mechaniches(self, dt):
        keys = pygame.key.get_pressed()
        if self.player is not None:
            self.player.mechaniches(keys, dt)
        for emitter in self.particle_emitters:
            emitter.update(dt)


class Player(AnimatedSolidSprite):
    def __init__(self, x, y, w, h, level, cw=None, ch=None, coffset_x=0, coffset_y=0):
        super().__init__(x, y, w, h, cw=cw, ch=ch, coffset_x=coffset_x, coffset_y=coffset_y)
        self.level = level
        self.vx = 0
        self.vy = 0
        self.added = False
        self.on_floor = False
        self.inventory = {}
        self.slotbar = SlotBar(x=260, y=0, level=level, slot_count=9)
        self.uilayer = UILayer()
        self.uilayer.add_element(self.slotbar)
        level.add_layer(self.uilayer)

        self.SPEED_X = 200
        self.JUMP_FORCE = 240
        self.def_GRAVITY = 500
        self.GRAVITY = 500
        self.FLYING = False
        self.mode = 0

    def physics(self, dt, dx, dy=0):
        self.vy += self.GRAVITY * dt
        dy = self.vy * dt + dy

        collide = self.move(dx, dy, self.level.grid)

        self.on_floor = False
        if collide == 'b':
            self.vy = 0
            self.on_floor = True
        elif collide == 'u':
            self.vy = 0

    def switch_mode(self):
        if self.mode == 0:
            self.mode = 1
            # self.GRAVITY = 0
        else:
            self.mode = 0
            self.GRAVITY = self.def_GRAVITY

    def mechaniches(self, keys, dt):
        current_vx = 0
        dy = 0
        if keys[pygame.K_a]:  current_vx -= self.SPEED_X
        if keys[pygame.K_d]: current_vx += self.SPEED_X

        if keys[pygame.K_SPACE] and self.on_floor:
            self.vy = -self.JUMP_FORCE

        elif keys[pygame.K_w] and self.mode == 1 and (not self.on_floor or self.FLYING):
            self.FLYING = True
            self.vy = 0
            dy = -300 * dt

        if keys[pygame.K_s] and self.mode == 1:
            dy = 300 * dt

        if keys[pygame.K_q]:  self.level.main_camera.apply_zoom(0.5 * dt)
        if keys[pygame.K_e]:  self.level.main_camera.apply_zoom(-0.5 * dt)

        if self.FLYING:
            self.GRAVITY = 0
        else:
            self.GRAVITY = self.def_GRAVITY
        if self.FLYING and self.on_floor:
            self.FLYING = False

        dx = current_vx * dt
        self.physics(dt, dx, dy)

        if dx != 0:
            self.set_state("walk")
        else:
            self.set_state("idle")

        self.update_animation(dt)
        ms = self.level.map_system
        mouse_buttons = pygame.mouse.get_pressed()
        mx, my = pygame.mouse.get_pos()
        ms.placer_light(mx, my)
        if mouse_buttons[0]:
            block = self.slotbar.get_and_use_selected(consume=(1 if self.mode == 0 else 0))
            if block.id is not '_None':
                ms.place_tile(mx, my, block=block)
                self.level.refresh_grid()

        if mouse_buttons[2]:
            ms.remove_tile(mx, my)
            self.level.refresh_grid()

    def save(self, path=''):
        f = {
            'x': self.x,
            'y': self.y,
            'slotbar': [[a.id, i] for a, i in self.slotbar.slots]
        }
        with open(path + '/-player.json', 'w') as file:
            file.write(json.dumps(f))

    def load(self, level, path=''):
        try:
            with open(path + '/-player.json') as file:
                datas = file.read()
        except FileNotFoundError:
            flag(f"No BlockMap found at {path}")
            return
        datas = json.loads(datas)
        x = datas['x']
        y = datas['y']
        self.setpos(x, y)
        self.slotbar.slots = [[level.registered_blocks[d], k] for d, k in datas['slotbar']]
        self.slotbar.render_bar()


class SlotBar(UIElement):
    def __init__(self, x, y, level, slot_count=9, scale=50):
        padding = 4
        width = (scale + padding) * slot_count + padding
        height = scale + padding * 2

        super().__init__(x, y, width, height)

        self.level = level
        self.slot_count = slot_count
        self.scale = scale
        self.padding = padding
        self.selected_index = 0

        self.slots = [[a, 1] for a in list(level.registered_blocks.values())[:slot_count]]

        while len(self.slots) < slot_count:
            self.slots.append( self.level.registered_blocks['_None'])

        self.font = pygame.font.SysFont("Arial", 12, bold=True)


    def get_selected(self):
        if 0 <= self.selected_index < len(self.slots):
            return self.slots[self.selected_index][0]
        return None

    def get_and_use_selected(self, consume=0):
        if consume == 0:
            return self.get_selected()
        if 0 <= self.selected_index < len(self.slots):
            self.slots[self.selected_index][1] -= 1
            ret = self.slots[self.selected_index][0]
            if self.slots[self.selected_index][1] == 0:
                self.slots[self.selected_index][0] = self.level.registered_blocks['_None']
                self.render_bar()
            return ret
        return None

    def render_bar(self):
        self.surface.fill((0, 0, 0, 0))

        for i in range(self.slot_count):
            px = self.padding + i * (self.scale + self.padding)
            py = self.padding
            slot_rect = pygame.Rect(px, py, self.scale, self.scale)

            pygame.draw.rect(self.surface, (50, 50, 50, 200), slot_rect)

            if i == self.selected_index:
                pygame.draw.rect(self.surface, (255, 255, 255), slot_rect, width=3)
            else:
                pygame.draw.rect(self.surface, (140, 140, 140), slot_rect, width=1)

            block = self.slots[i][0]
            if block.id is '_None':
                continue
            icon_size = self.scale - 12
            icon_surf = pygame.transform.scale(block.surface, (icon_size, icon_size))

            ix = px + (self.scale - icon_size) // 2
            iy = py + (self.scale - icon_size) // 2
            self.surface.blit(icon_surf, (ix, iy))

            num_surf = self.font.render(str(i + 1), True, (255, 255, 255))
            self.surface.blit(num_surf, (px + 3, py + 3))

            count_surf = self.font.render(str(self.slots[i][1]), True, (255, 255, 255))
            self.surface.blit(count_surf, (px + 3, py + self.scale - 15))

    def handle_event(self, event):
        if not self.visible:
            return False

        if event.type == pygame.KEYDOWN:
            if pygame.K_1 <= event.key <= pygame.K_9:
                new_index = event.key - pygame.K_1

                if new_index < self.slot_count:
                    self.selected_index = new_index
                    self.render_bar()
                    return True

        return False


class WorldLevel(Scene):
    def __init__(self, root, map_system=None):
        super().__init__(root)
        self.map_system = map_system
        self.registered_blocks = {}
        self.savable_objects = []
        self.level_name = ''

    def set_map(self, map_system):
        if self.map_system is not None:
            flag("map system already present, overwritten", level=2)
        self.map_system = map_system
        self.updatables.append(self.map_system)
        self.savable_objects.append(self.map_system)
        self.add_renderable(self.map_system, -2)

    def add_player(self, player):
        self.player = player
        self.solids.append(player)
        self.main_camera.target = player
        self.mechaniques.append(player)
        self.savable_objects.append(self.player)

    def refresh_grid(self):
        self.grid.clear()
        for obj in self.solids:
            self.grid.insert(obj)

    def save(self, path='saves'):
        os.makedirs(path, exist_ok=True)
        os.makedirs(path + '/' + self.level_name, exist_ok=True)

        for savable in self.savable_objects:
            savable.save(path=path + '/' + self.level_name)

    def load(self, path):
        self.level_name = path.split('/')[-1]
        for savable in self.savable_objects:
            savable.load(self, path=path)


import os
import json


def get_save_names(path='saves/'):
    return os.listdir(path)


class SaveMenu(Scene):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.ui = UILayer("SaveSelection")
        self.add_layer(self.ui)
        self.refresh_saves()
        self.root = game_root

    def refresh_saves(self):
        self.ui.elements.clear()

        saves = get_save_names()

        self.ui.add_element(UIText(x=self.centerx - 100, y=50, text="Select World", size=40))

        start_y = 120
        for i, save_name in enumerate(saves):
            btn = UIButton(
                x=self.centerx - 150,
                y=start_y + (i * 55),
                width=300,
                height=45,
                text=save_name
            )
            btn2 = UIButton(
                x=self.centerx + 160,
                y=start_y + (i * 55),
                width=50,
                height=45,
                text='Del'
            )
            btn.on_click = lambda name=save_name: self.load_and_play(name)
            btn2.on_click = lambda name=save_name: self.remove(name)
            self.ui.add_element(btn)
            self.ui.add_element(btn2)

        back_btn = UIButton(x=self.centerx - 75, y=500, width=150, height=40, text="Back to Title")
        back_btn.on_click = lambda: self.root.set_scene('0')
        self.ui.add_element(back_btn)

    def load_and_play(self, save_name):
        self.root.set_scene('1')
        world_level = self.root.loaded_scenes['1']
        world_level.load(f"saves/{save_name}")

    def remove(self, save_name):
        try:
            for file in os.listdir("saves"):
                if f"{save_name}-" in file:  # TODO: This is Brutally wrong
                    os.remove(f"saves/{file}")
        except PermissionError:
            flag(f"Impossible to delete {save_name} due to Lack of Permissions", 3)


class SaveAsMenu(Scene):
    def __init__(self, game_root, world_level):
        super().__init__(game_root)
        self.world_level = world_level
        self.root = game_root
        self.ui = UILayer("SaveAs")
        self.add_layer(self.ui)

        self.input_box = UITextInput(self.centerx - 100, self.centery - 50, 200, 40)
        self.ui.add_element(self.input_box)

        confirm_btn = UIButton(self.centerx - 60, self.centery + 20, 120, 40, text="Confirm Save")
        confirm_btn.on_click = self.perform_save
        self.ui.add_element(confirm_btn)

        cancel_btn = UIButton(self.centerx - 60, self.centery + 70, 120, 40, text="Cancel")
        cancel_btn.on_click = lambda: self.root.set_scene('1')  # Back to game
        self.ui.add_element(cancel_btn)

    def perform_save(self):
        filename = self.input_box.text.strip()
        if filename:
            self.world_level.level_name = filename
            self.world_level.save()
            flag(f"Created new save: {filename}", level=0)
            self.root.set_scene('1')


if __name__ == "__main__":
    game = Engine(name='Ortensia', base_size=(1000, 600), flag=pygame.SCALED | pygame.RESIZABLE)

    s = game.scaler
    # bg0 = game.add_create_layer("Background", 0.1)
    # bg1 = game.add_create_layer("Background", 0.2)
    # bg2 = game.add_create_layer("Background", 0.3)
    bg3 = game.add_create_layer("Background", 0.5)
    bg4 = game.add_create_layer("Background", 0.8)
    bg5 = game.add_create_layer("Background", 1)
    ui_layer = UILayer("HUD")


    def on_reset_click():
        game.running = False


    score_label = UIText(x=20, y=20, text="Stop", size=30, color=(255, 215, 0), shadow=True, font_name='minecraftia20')
    ui_layer.add_element(score_label)
    reset_btn = UIButton(x=20, y=60, width=120, height=40, text="Reset Cam",
                         bg_color=(200, 50, 50), on_click=on_reset_click)
    ui_layer.add_element(reset_btn)
    game.add_layer(ui_layer)

    fg = LitLayer("Foreground", 1, ambient_color=(100, 100, 100))  # Dark ambient
    game.add_layer(fg)
    """
    particles = game.add_create_layer("particles", 1.5)
    particles2 = game.add_create_layer("particles", 2.3)
    """
    terrain_layer = game.add_create_layer("Terrain", 1)

    map_system = BlockMap(game, fg, tile_size=s(32))
    game.map_system = map_system
    # fg = game.add_create_layer("Foreground", 1.0)

    wall_lamp4 = LightSource(s(250), s(200), radius=s(200), color=(20, 126, 126), falloff=0.99, steps=100)
    """wall_lamp3 = LightSource(s(130), s(400), radius=s(200), color=(255, 60, 60), falloff=0.1, steps=10)
    wall_lamp = LightSource(s(600), s(450), radius=s(200), color=(60, 255, 60), falloff=0.99, steps=100)
    wall_lamp2 = LightSource(s(900), s(450), radius=s(200), color=(60, 60, 255), falloff=0.99, steps=100)"""
    """fg.add_light(wall_lamp)
    fg.add_light(wall_lamp2)
    fg.add_light(wall_lamp3)"""

    # fg.add_effect(PostProcessing.fog, 1, 3)
    # particles.add_effect(PostProcessing.lumen, 4, 2)
    # particles2.add_effect(PostProcessing.lumen, 4, 2)
    # fg.add_effect(PostProcessing.motion_blur, game.main_camera, 3.5)
    # fg.add_effect(PostProcessing.black_and_white)
    # bg4.add_effect(PostProcessing.lumen, 60, 0.5)

    # player = SolidSprite(s(400), s(300), s(40), s(40), (255, 255, 255))
    from blocks import *

    player = Player(s(400), s(300), s(64), s(64), level=game, cw=16, coffset_x=23, coffset_y=-6)
    # player.add_animation('walk', load_spritesheet("Graphic/examples/AuryRunning.png", 64, 64, row=0, scale=(1, 1)))
    walk_loader = AnimationLoader("Graphic/examples/AuryRunning.png", 64, 64, row=0, scale=(1, 1))
    player.add_animation('walk', walk_loader)
    player.show_hitboxes = True
    fg.sprites.append(player)
    water = FluidSprite(s(500), s(650), s(600), s(100), color=(50, 100, 255, 120))
    # fg.sprites.append(water)
    game.solids.append(player)
    game.main_camera.target = player

    emitter1 = ParticleEmitter(size=s(1.5), sparsity=0.2, g=40, color='random', deltax=s(20), deltay=s(20))
    ff = FireflyEmitter(count=100, size=1)
    ff2 = FireflyEmitter(count=100, size=1)
    emitter1.track(player)
    """particles.emitters.append(ff)
    particles2.emitters.append(ff2)"""
    # game.particle_emitters.append(emitter1)

    for _ in range(50):
        ff.emit(random.randint(0, 1000), random.randint(200, 500), 1)
        pass
    for _ in range(50):
        ff2.emit(random.randint(0, 1000), random.randint(200, 500), 1)
        pass

    game.particle_layer_idx = 2
    fg.add_light(LightSource(900, 580, radius=200, color=(255, 0, 0), falloff=0.99, steps=200))
    fg.add_light(LightSource(650, 580, radius=200, color=(125, 255, 255), falloff=0.99, steps=200))
    fg.add_static(Sprite(650, 580, 16, 128, texture="assets/textures/blocks/lamp.png", alpha=True))
    fg.add_static(Sprite(900, 580, 16, 128, texture="assets/textures/blocks/lamp.png", alpha=True))
    # bg0.add_static(Sprite(-100, -220, 2304//2, 1396//2, (40, 40, 80), texture='assets/textures/backgrounds/1.png', alpha=True))
    # bg2.add_static(Sprite(-100, -150, 2304//2, 1396//2, (40, 40, 80), texture='assets/textures/backgrounds/2.png', alpha=True))
    # bg3.add_static(Sprite(-100, -120, 2304/60/2, 1396//2, (40, 40, 80), texture='assets/textures/backgrounds/3.png', alpha=True))
    bg4.add_static(
        Sprite(-100, -80, 2304 // 2, 1396 // 2, (40, 40, 80), texture='assets/textures/backgrounds/4.png', alpha=True))
    bg5.add_static(
        Sprite(-100, -20, 2304 // 2, 1396 // 2, (40, 40, 80), texture='assets/textures/backgrounds/5.png', alpha=True))
    base = SolidSprite(-100, 700, 3000, 10, (40, 40, 40))
    fg.add_static(base)
    game.solids.append(base)
    """for i in range(15):
        if i % 3 == 0:
            wall = SolidSprite(i * s(400), s(450), s(60), s(150), (255, 20, 50))
            fg.sprites.append(wall)
            game.solids.append(wall)
        else:
            wall = Sprite(i * s(400), s(450), s(60), s(150), (30, 70, 40))
            fg.sprites.append(wall)
    """

    game.player = player
    game.cameras.append(Camera)
    game.refresh_grid()
    from Graphic.save_system import WorldState

    while game.running:
        game.update()
