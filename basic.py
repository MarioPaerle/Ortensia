import pygame
from Graphic.base import *
from Graphic.functions import *


class BuildableLayer:
    def __init__(self):
        pass


class Engine(Game):
    def __init__(
            self,
            name='Ortensia',
            base_size=(1000, 600),
            flag=pygame.SCALED | pygame.RESIZABLE,
            scaler=lambda x: int(x*1),

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

    def dt(self):
        return self.clock.tick(self.max_fps) / self.game_div

    def update(self):
        while self.running:
            dt = self.dt()
            camera = self.cameras[self.camera_id]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.grid.clear()
            for obj in self.solids:
                self.grid.insert(obj)

            self.mechaniches(dt)
            camera.update()

            for emitter in self.particle_emitters:
                emitter.update(dt)

            self.screen.fill((20, 20, 30))

            for i, layer in enumerate(self.layers):
                if self.particle_layer_idx != -1 and self.particle_layer_idx == i:
                    layer.render(self.screen, camera, emitters=self.particle_emitters)
                else:
                    layer.render(self.screen, camera)

            if self.particle_layer_idx == -1:
                for emitter in self.particle_emitters:
                    emitter.draw(self.screen, camera)

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

    def mechaniches(self, keys, dt):
        speed = 200 * dt

        dx, dy = 0, 0
        if keys[pygame.K_LEFT]:  dx -= speed
        if keys[pygame.K_RIGHT]: dx += speed
        if keys[pygame.K_UP]:    dy -= speed
        if keys[pygame.K_DOWN]:  dy += speed
        if keys[pygame.K_SPACE]:
            self.game.main_camera.shake_intensity = 1

        self.move(dx, dy, self.game.grid)
        self.update_animation(dt)
        water.update(interactors=[self])


game = Engine(name='Ortensia', base_size=(1000, 600), flag=pygame.SCALED | pygame.RESIZABLE)
s = game.scaler
bg2 = game.add_create_layer("Background2", 0.2)
bg = game.add_create_layer("Background", 0.5)
particles = game.add_create_layer("particles", 1.0)
# fg = game.add_layer("Foreground", 1.0)

fg = LitLayer("Foreground", 1.0, ambient_color=(30, 30, 50)) # Dark ambient
game.add_layer(fg)

wall_lamp3 = LightSource(s(130), s(400), radius=s(200), color=(255, 60, 60), falloff=0.99, steps=100)
wall_lamp = LightSource(s(600), s(450), radius=s(200), color=(60, 255, 60), falloff=0.99, steps=100)
wall_lamp2 = LightSource(s(900), s(450), radius=s(200), color=(60, 60, 255), falloff=0.99, steps=100)
fg.add_light(wall_lamp)
fg.add_light(wall_lamp2)
fg.add_light(wall_lamp3)

# fg.add_effect(PostProcessing.underwater_distortion, 5)
particles.add_effect(PostProcessing.lumen, 40, 5)
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
emitter1.track(player)
game.particle_emitters.append(ff)
# game.particle_emitters.append(emitter1)

for _ in range(30):
    ff.emit(random.randint(0, 1000), random.randint(200, 500), 1)
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
game.update()
