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
                self.active_scene_name = '1'

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
    def __init__(self, x, y, w, h, game, cw=None, ch=None, coffset_x=0, coffset_y=0):
        super().__init__(x, y, w, h, cw=cw, ch=ch, coffset_x=coffset_x, coffset_y=coffset_y)
        self.game = game
        self.vx = 0
        self.vy = 0
        self.added = False
        self.on_floor = False
        self.inventory = {}

    def physics(self, dt, dx, dy):
        self.vy += 9.81 * dt

        collide = self.move(dx, dy, self.game.grid)

        self.on_floor = False
        if collide == 'b':
            self.vy = 0
            self.on_floor = True

    def mechaniches(self, keys, dt):
        speed = 60 * dt
        dx, dy = self.vx, self.vy

        if keys[pygame.K_LEFT]:  dx -= speed
        if keys[pygame.K_RIGHT]: dx += speed
        if keys[pygame.K_UP] and self.on_floor:
            self.vy = - 4
        if keys[pygame.K_DOWN]:  dy += speed
        if keys[pygame.K_q]:  game.main_camera.apply_zoom(0.5 * dt)
        if keys[pygame.K_e]:  game.main_camera.apply_zoom(-0.5 * dt)
        self.physics(dt, dx, dy)

        if dx != 0 or dy != 0:
            self.set_state("walk")
        else:
            self.set_state("idle")

        self.update_animation(dt)
        # water.update(interactors=[self])

        mouse_buttons = pygame.mouse.get_pressed()
        mx, my = pygame.mouse.get_pos()
        """if mouse_buttons[0]:
            block = random.choice(list(RAPIDBLOCKS.values()))
            map_system.place_tile(mx, my, block=block)
            game.refresh_grid()

        if mouse_buttons[2]:
            map_system.remove_tile(mx, my)
            game.refresh_grid()"""


class WorldLevel(Scene):
    def __init__(self, root, map_system=None):
        super().__init__(root)
        self.map_system = map_system

    def set_map(self, map_system):
        if self.map_system is not None:
            flag("map system already present, overwritten", level=2)
        self.map_system = map_system

        

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

    player = Player(s(400), s(300), s(64), s(64), game=game, cw=16, coffset_x=23, coffset_y=-6)
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
