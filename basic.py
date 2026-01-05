import pygame
from Graphic.base import *
from Graphic.functions import *


class Map:
    def __init__(self):
        pass


class Engine(Game):
    def __init__(
            self,
            name='Ortensia',
            base_size=(1000, 600),
            flag=pygame.SCALED | pygame.RESIZABLE,
            scaler=lambda x: int(x * 1),

    ):
        self.name = name
        self.base_size = base_size
        self.flag = flag
        self.scaler = scaler
        self.w, self.h = base_size[0], base_size[1]
        self.sw, self.sh = scaler(self.w), scaler(self.h)
        self.asset_folder = ''
        super().__init__(self.sw, self.sh, title=name, flag=flag)
        self.cameras = [self.main_camera]
        self.camera_id = 0
        self.updaters = []
        self.player = None
        self.g = 9.81

    def dt(self):
        return self.clock.tick(self.max_fps) / self.game_div

    def refresh_grid(self):
        self.grid.clear()
        for obj in self.solids:
            self.grid.insert(obj)

    def update(self):
        while self.running:
            dt = self.dt()
            camera = self.cameras[self.camera_id]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

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

            fps = self.clock.get_fps()
            pygame.display.set_caption(f"{self.name} | FPS: {int(fps)}")
            pygame.display.flip()

    def mechaniches(self, dt):
        keys = pygame.key.get_pressed()
        self.player.mechaniches(keys, dt)
        for emitter in self.particle_emitters:
            emitter.update(dt)


class Player(AnimatedSprite):
    def __init__(self, x, y, w, h, game):
        AnimatedSprite.__init__(self, x, y, w, h)
        self.game = game
        self.vx = 0
        self.vy = 0

    def physics(self, dt):
        self.vy += 9.81 * dt

    def mechaniches(self, keys, dt):
        speed = 200 * dt
        dx, dy = self.vx, self.vy

        if keys[pygame.K_LEFT]:  dx -= speed
        if keys[pygame.K_RIGHT]: dx += speed
        if keys[pygame.K_UP]:    dy -= speed
        if keys[pygame.K_DOWN]:  dy += speed
        if keys[pygame.K_SPACE]:
            self.game.main_camera.shake_intensity = 1
            fg.add_light(wall_lamp4)

        self.physics(dt)

        if dx != 0 or dy != 0:
            self.set_state("walk")
        else:
            self.set_state("idle")

        collide = self.move(dx, dy, self.game.grid)

        if collide == 'b':
            self.vy = 0

        self.update_animation(dt)
        water.update(interactors=[self])

        mouse_buttons = pygame.mouse.get_pressed()
        mx, my = pygame.mouse.get_pos()
        if mouse_buttons[0]:
            map_system.place_tile(mx, my, color=(100, 200, 100))
            game.refresh_grid()

        if mouse_buttons[2]:
            map_system.remove_tile(mx, my)
            game.refresh_grid()


game = Engine(name='Ortensia', base_size=(1000, 600), flag=pygame.SCALED | pygame.RESIZABLE)
s = game.scaler
bg2 = game.add_create_layer("Background2", 0.2)
bg = game.add_create_layer("Background", 0.5)
particles = game.add_create_layer("particles", 1)
particles2 = game.add_create_layer("particles", 2)
terrain_layer = game.add_create_layer("Terrain", 1.0)
map_system = TileMap(game, terrain_layer, tile_size=s(40), texture='examples/Ortensia1.png')
# fg = game.add_create_layer("Foreground", 1.0)

fg = LitLayer("Foreground", 1.0, ambient_color=(100, 100, 100))  # Dark ambient
game.add_layer(fg)

wall_lamp3 = LightSource(s(130), s(400), radius=s(200), color=(255, 60, 60), falloff=0.1, steps=10)
wall_lamp4 = LightSource(s(250), s(200), radius=s(200), color=(255, 0, 60), falloff=0.99, steps=100)
wall_lamp = LightSource(s(600), s(450), radius=s(200), color=(60, 255, 60), falloff=0.99, steps=100)
wall_lamp2 = LightSource(s(900), s(450), radius=s(200), color=(60, 60, 255), falloff=0.99, steps=100)
"""fg.add_light(wall_lamp)
fg.add_light(wall_lamp2)
fg.add_light(wall_lamp3)"""

# fg.add_effect(PostProcessing.underwater_distortion, 5)
# particles.add_effect(PostProcessing.lumen, 4, 2)
# particles2.add_effect(PostProcessing.lumen, 4, 2)
# fg.add_effect(PostProcessing.motion_blur, game.main_camera, 3.5)
# fg.add_effect(PostProcessing.black_and_white)

# player = SolidSprite(s(400), s(300), s(40), s(40), (255, 255, 255))
player = Player(s(400), s(300), s(64), s(64), game=game)
player.add_animation('walk', load_spritesheet("Graphic/examples/AuryRunning.png", 64, 64, row=0, scale=(1, 1)))
fg.sprites.append(player)
water = FluidSprite(s(200), s(500), s(600), s(100), color=(50, 100, 255, 120))
fg.sprites.append(water)
game.solids.append(player)
game.main_camera.target = player

emitter1 = ParticleEmitter(size=s(1.5), sparsity=0.2, g=40, color='random', deltax=s(20), deltay=s(20))
ff = FireflyEmitter(count=100, size=1)
ff2 = FireflyEmitter(count=100, size=1)
emitter1.track(player)
particles.emitters.append(ff)
particles2.emitters.append(ff2)
# game.particle_emitters.append(emitter1)

for _ in range(50):
    ff.emit(random.randint(0, 1000), random.randint(200, 500), 1)
    pass
for _ in range(50):
    ff2.emit(random.randint(0, 1000), random.randint(200, 500), 1)
    pass

game.particle_layer_idx = 2

for i in range(15):
    bg.add_static(Sprite(i * s(250), s(200), s(100), s(400), (40, 40, 80)))

for i in range(10):
    bg2.add_static(Sprite(i * s(400), s(400), s(100), s(200), (70, 70, 100)))

for i in range(15):
    if i % 3 == 0:
        wall = SolidSprite(i * s(400), s(450), s(60), s(150), (255, 20, 50))
        fg.sprites.append(wall)
        game.solids.append(wall)
    else:
        wall = Sprite(i * s(400), s(450), s(60), s(150), (30, 70, 40))
        fg.sprites.append(wall)

game.player = player
game.cameras.append(Camera)
game.refresh_grid()
game.update()
