import pygame

pygame.init()

from Graphic.base import *
from Graphic.functions import *
import random
from Graphic._audio import SoundEngine


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
        self.loaded_scenes[self.active_scene_name].on_stop()
        if name in self.loaded_scenes:
            self.active_scene_name = name
            self.loaded_scenes[name].clock.tick()
            self.loaded_scenes[name].on_run()
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
        # add_grain(self.screen, 10, dynamic=True)
        # add_vignette(self.screen, 0.33)
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
        self.x = self.frect.x
        self.y = self.frect.y
        self.added = False
        self.on_floor = False
        self.inventory = {}
        self.slotbar = SlotBar(x=260, y=0, level=level, slot_count=9)
        self.uilayer = UILayer()
        self.uilayer.add_element(self.slotbar)
        self.no_reachable_text = UIText(500, 570, "Not Reachable", shadow=False, font_name="ArcadeClassic", size=25)
        self.uilayer.add_element(self.no_reachable_text)
        level.add_layer(self.uilayer)
        self.max_life = 20
        self.life = 20

        self.SPEED_X = 200
        self.JUMP_FORCE = 260

        # --- GRAVITY SETTINGS (Celeste Style) ---
        self.GRAVITY_RISE = 620  # Standard gravity when going up
        self.GRAVITY_FALL = 800  # Heavier gravity when falling
        self.GRAVITY = self.GRAVITY_RISE

        self.FLYING = False
        self.mode = 0

        # --- JUMP BUFFER & AUTOJUMP ---
        self.jump_buffer_timer = 0.0
        self.JUMP_BUFFER_DURATION = 0.15  # 150ms forgiveness window
        self.prev_space_state = False

        # --- CLICK COOLDOWN ---
        self.click_cooldown_timer = 0.0
        self.CLICK_COOLDOWN_DURATION = 0.2

        # --- DASH & MOVEMENT STATE ---
        self.facing_right = True
        self.is_dashing = False
        self.dash_timer = 0.0
        self.dash_cooldown_timer = 0.0
        self.stand_still = False

        self.DASH_SPEED = 600
        self.DASH_DURATION = 0.13
        self.DASH_COOLDOWN = 0.6
        self.DOUBLE_TAP_WINDOW = 250  # ms

        # Tap detection (Generic for Keyboard & Controller)
        self.last_tap_time = 0
        self.last_tap_key = None
        self.prev_input_left = False
        self.prev_input_right = False

        # Inputs for slot switching logic
        self.prev_lb = False
        self.prev_rb = False

        # --- CONTROLLER INIT ---
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            try:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"Controller detected: {self.joystick.get_name()}")
            except Exception as e:
                print(f"Controller error: {e}")


        # ----- SOUNDS
        self.walking_timer = 100
        self.sound_engine = None

    def die(self, message=''):
        "Die animations and modalitiea"
        flag(message, 1)
        self.setpos(*self.level.spawnpoint)
        self.life = self.max_life
        self.vx = 0
        self.vy = 0

    def take_damage(self, quantity):
        "Take a quantity of Damage"
        if self.mode == 0:
            self.life -= quantity

    def physics(self, dt, dx, dy=0, holding_jump=False):
        if not self.FLYING and not self.is_dashing:
            if self.vy > 0 and holding_jump:
                self.GRAVITY = self.GRAVITY_RISE
            else:
                if holding_jump:
                    self.GRAVITY = self.GRAVITY_RISE
                else:
                    self.GRAVITY = self.GRAVITY_FALL
        elif self.is_dashing:
            self.GRAVITY = 0

        self.vy += self.GRAVITY * dt
        dy = self.vy * dt + dy

        old_x, old_y = self.x, self.y
        collide, collided_with = self.move(dx, dy, self.level.grid)

        if abs(self.x - old_x) > 0.01 or abs(self.y - old_y) > 0.01:
            self.level.grid.mark_dirty()

        self.on_floor = False
        if collide == 'b':
            collided_with.on_touch(self, dt)
            if self.walking_timer <= 0 and dx != 0:
                self.walking_timer = 22
                collided_with.play_r('touch')
            self.vy = 0
            self.on_floor = True
        elif collide == 'u':
            self.vy = 0

        if self.walking_timer > 0:
            self.walking_timer -= 1
        if dy < 0:
            self.walking_timer = 0

    def switch_mode(self):
        if self.mode == 0:
            self.mode = 1
        else:
            self.mode = 0
            self.GRAVITY = self.GRAVITY_RISE
            self.FLYING = False
            self.is_dashing = False

    def mechaniches(self, keys, dt):
        # Update Timers
        if self.jump_buffer_timer > 0:
            self.jump_buffer_timer -= dt
        if self.click_cooldown_timer > 0:
            self.click_cooldown_timer -= dt
        if self.dash_cooldown_timer > 0:
            self.dash_cooldown_timer -= dt

        curr_time_ms = pygame.time.get_ticks()

        # --- INPUT UNIFICATION ---
        input_left = keys[pygame.K_a]
        input_right = keys[pygame.K_d]
        input_up = keys[pygame.K_w]
        input_down = keys[pygame.K_s]
        input_jump = keys[pygame.K_SPACE]
        input_dash_btn = False

        mouse_buttons = pygame.mouse.get_pressed()
        input_place = mouse_buttons[2]
        input_destroy = mouse_buttons[0]

        input_lb = False
        input_rb = False

        if self.joystick:
            axis_x = self.joystick.get_axis(0)
            if axis_x < -0.4:
                input_left = True
            elif axis_x > 0.4:
                input_right = True

            axis_y = self.joystick.get_axis(1)
            if axis_y < -0.4:
                input_up = True
            elif axis_y > 0.4:
                input_down = True

            if self.joystick.get_button(0): input_jump = True

            if self.joystick.get_button(2): input_dash_btn = True

            if self.joystick.get_button(4): input_lb = True

            if self.joystick.get_button(5): input_rb = True

            rx = self.joystick.get_axis(2)
            ry = self.joystick.get_axis(3)
            if abs(rx) > 0.2 or abs(ry) > 0.2:
                mx, my = pygame.mouse.get_pos()
                sensitivity = 800 * dt
                pygame.mouse.set_pos(mx + rx * sensitivity, my + ry * sensitivity)

            if self.joystick.get_axis(4) > 0.5:
                input_place = True

            if self.joystick.get_axis(5) > 0.5:
                input_destroy = True

        if input_rb and not self.prev_rb:
            self.slotbar.selected_index = (self.slotbar.selected_index + 1) % 9
            self.slotbar.render_bar()
        self.prev_rb = input_rb

        if input_lb and not self.prev_lb:
            self.slotbar.selected_index = (self.slotbar.selected_index - 1) % 9
            self.slotbar.render_bar()

        self.prev_lb = input_lb

        if input_left and not self.prev_input_left:
            if self.last_tap_key == 'left' and (curr_time_ms - self.last_tap_time) < self.DOUBLE_TAP_WINDOW:
                if self.dash_cooldown_timer <= 0 and self.mode == 0:
                    self.is_dashing = True
                    self.dash_timer = self.DASH_DURATION
                    self.dash_cooldown_timer = self.DASH_COOLDOWN
                    self.facing_right = False
                    self.sound_engine.play_sfx("assets/sounds/dash1.wav")

            self.last_tap_key = 'left'
            self.last_tap_time = curr_time_ms
        self.prev_input_left = input_left

        if input_right and not self.prev_input_right:
            if self.last_tap_key == 'right' and (curr_time_ms - self.last_tap_time) < self.DOUBLE_TAP_WINDOW:
                if self.dash_cooldown_timer <= 0 and self.mode == 0:
                    self.is_dashing = True
                    self.dash_timer = self.DASH_DURATION
                    self.dash_cooldown_timer = self.DASH_COOLDOWN
                    self.facing_right = True
                    self.sound_engine.play_sfx("assets/sounds/dash1.wav")
            self.last_tap_key = 'right'
            self.last_tap_time = curr_time_ms


        self.prev_input_right = input_right

        if input_dash_btn and self.dash_cooldown_timer <= 0 and self.mode == 0 and not self.is_dashing:
            self.is_dashing = True
            self.dash_timer = self.DASH_DURATION
            self.dash_cooldown_timer = self.DASH_COOLDOWN

            if input_left:
                self.facing_right = False
            elif input_right:
                self.facing_right = True

        current_vx = 0
        dy = 0

        if self.is_dashing:
            self.dash_timer -= dt
            if self.dash_timer <= 0:
                self.is_dashing = False
                self.vy = 0
            else:
                current_vx = self.DASH_SPEED if self.facing_right else -self.DASH_SPEED
                self.vy = 0
        else:
            if input_left:
                current_vx -= self.SPEED_X
                self.facing_right = False
            if input_right:
                current_vx += self.SPEED_X
                self.facing_right = True

        if not self.is_dashing:
            if input_jump and not self.prev_space_state:
                self.jump_buffer_timer = self.JUMP_BUFFER_DURATION
            self.prev_space_state = input_jump

            wants_to_jump = (self.jump_buffer_timer > 0) or input_jump

            if self.on_floor and wants_to_jump:
                self.vy = -self.JUMP_FORCE
                self.jump_buffer_timer = 0

                # Flying Logic
            elif input_up and self.mode == 1 and (not self.on_floor or self.FLYING):
                self.FLYING = True
                self.vy = 0
                dy = -300 * dt
            if input_down and self.mode == 1:
                dy = 300 * dt

        if self.FLYING:
            self.GRAVITY = 0
        if self.FLYING and self.on_floor:
            self.FLYING = False

        # Physics Step
        dx = current_vx * dt
        self.physics(dt, dx, dy, holding_jump=input_jump)

        # Animations
        if self.is_dashing:
            self.set_state("walk")
        elif input_jump and dx > 0:
            self.set_state("jump_right")
        elif input_jump and dx < 0:
            self.set_state("jump_left")
        elif dx > 0:
            self.set_state("walk_right")
        elif dx < 0:
            self.set_state("walk_left")
        else:
            self.set_state("idle")

        self.update_animation(dt)

        ms = self.level.map_system
        mx, my = pygame.mouse.get_pos()

        distance = ms.get_grid_distance2(self.pos(), (mx, my))

        if distance < 4 or self.mode == 1:
            ms.placer_light(mx, my, color=(255, 255, 255))
            item_in_hand = self.slotbar.get_selected()

            clicked = ms.get_tile(mx, my)

            if self.click_cooldown_timer <= 0:
                action_performed = False

                if input_place:
                    if clicked is None and isinstance(item_in_hand, Block):
                        block = self.slotbar.get_and_use_selected(consume=(1 if self.mode == 0 else 0))
                        if block.id != '_None':
                            ms.place_tile(mx, my, block=block)
                            self.level.refresh_grid()
                            action_performed = True
                    elif isinstance(clicked, Block):
                        clicked.on_click(self)
                        action_performed = True
                    elif not isinstance(item_in_hand, Block):
                        if hasattr(item_in_hand, 'on_click'):
                            item_in_hand.on_click(self)
                        action_performed = True

                if input_destroy and clicked is not None and item_in_hand.id != '_None':
                    if not isinstance(item_in_hand, Block) and clicked.type in item_in_hand.breaking:
                        ms.remove_tile(mx, my)
                        item_in_hand.on_use(self)
                    self.level.refresh_grid()
                    action_performed = True

                if action_performed:
                    self.click_cooldown_timer = self.CLICK_COOLDOWN_DURATION

        else:
            if input_destroy or input_place:
                self.no_reachable_text.alpha = 128

            ms.placer_light(mx, my, color=(180, 20, 20))

        if self.no_reachable_text.alpha > 0:
            self.no_reachable_text.alpha -= self.no_reachable_text.alpha / 30

        if self.life <= 0:
            self.die('Player Dead')
        elif self.y > 1300:
            self.die('Player Exceeded world minimum, felt into the void')

        if self.life <= 0:
            self.life = 0

        ms.update(dt, mouse_pos=(mx, my))
        self.no_reachable_text.render_text()

    def save(self, path=''):
        f = {
            'x': self.x,
            'y': self.y,
            'slotbar': [[a.id, i] for a, i in self.slotbar.slots]
        }
        with open(path + '/..' + '/-player.json', 'w') as file:
            file.write(json.dumps(f))

    def load(self, level, path=''):
        try:
            with open(path + '/..' + '/-player.json') as file:
                datas = file.read()
        except FileNotFoundError:
            flag(f"No BlockMap found at {path}")
            return
        datas = json.loads(datas)
        x = datas['x']
        y = datas['y']
        self.setpos(x, y)
        self.slotbar.slots = [
            [level.registered_blocks.get(a, '_None'), b] for a, b in datas['slotbar']
        ]
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

        self.slots = []  # [[a, 1] for a in list(level.registered_blocks.values())[:slot_count]]
        # self.slots[0] = [level.registered_blocks['Fire'], 64]

        while len(self.slots) < slot_count:
            self.slots.append([self.level.registered_blocks['_None'], 1])

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
        self.spawnpoint = (400, 300)
        self.update_call = lambda: 0
        self.music = ""
        self.ambience = ""
        self.sound_engine = None

    def set_map(self, map_system):
        if self.map_system is not None:
            flag("map system already present, overwritten", level=2)
        self.map_system = map_system
        self.updatables.append(self.map_system)
        self.savable_objects.append(self.map_system)
        self.add_renderable(self.map_system, -2)

    def play_music(self):
        if self.sound_engine is None:
            raise Exception("You must initialize the Sound Engine before playing")
        if self.music:
            self.sound_engine.play_music(self.music)
        else:
            flag("No Music is present", 2)

    def play_ambience(self):
        if self.sound_engine is None:
            raise Exception("You must initialize the Sound Engine before playing")
        if self.ambience:
            self.sound_engine.play_ambience(self.ambience)
        else:
            flag("No Ambience is present", 2)

    def on_run(self):
        if self.sound_engine is not None:
            self.play_ambience()
            self.play_music()

    def on_stop(self):
        self.sound_engine.pause_all()

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

        for point in self.map_system.data['middle']:
            if 'spawnpoint.png' == self.map_system.data['middle'][point].id:
                self.spawnpoint = point[0] * self.map_system.tile_size, point[1] * self.map_system.tile_size
                flag(f"Set Spawnpoint at {point}")

    def register_blocks(self, blocks: dict):
        self.registered_blocks = blocks
        for block in self.registered_blocks:
            if isinstance(self.registered_blocks[block], AnimatedBlock):
                self.updatables.append(self.registered_blocks[block])

    def callback(self, game):
        return
        add_grain(self.screen, 6, dynamic=True)
        add_vignette(self.screen, 0.3)

    def update(self):
        self.update_call()

        super().update(execute=self.callback)


import os
import json


def get_save_names(path='saves/'):
    return [s for s in os.listdir(path) if '.' not in s]


class SaveMenu(Scene):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.root = game_root

        # 1. Setup Background Layer
        self.background = Layer('bg', 0)
        self.add_layer(self.background)  # <--- CRITICAL FIX: Add layer to scene

        # 2. Setup Background GIF
        self._init_random_bg_gif()

        # 3. Setup UI Layer
        self.ui = UILayer("SaveSelection", parallax=0)
        self.add_layer(self.ui)
        self.refresh_saves()

    def _init_random_bg_gif(self):
        bg_dir = "assets/backgrounds"

        # Ensure directory exists and get only GIFs
        if not os.path.exists(bg_dir):
            print(f"Directory missing: {bg_dir}")
            return

        gifs = [f for f in os.listdir(bg_dir) if f.lower().endswith('.gif')]

        if gifs:
            selected_gif = random.choice(gifs)
            full_path = os.path.join(bg_dir, selected_gif)

            # Create Sprite covering the whole screen
            w, h = self.root.w, self.root.h
            self.bg = AnimatedSprite(0, 0, w, h)

            frames = load_gif_as_surfaces(full_path, target_size=(1200, 900))

            if frames:
                self.bg.add_animation("idle", frames)
                self.bg.set_state('idle')

                self.background.sprites.append(self.bg)
                self.updatables.append(self.bg)
            else:
                print(f"Failed to load frames from {selected_gif}")

    def refresh_saves(self):
        self.ui.elements.clear()

        saves = get_save_names()

        self.ui.add_element(
            UIText(x=self.root.w // 2 - 120, y=50, text="Select Level", size=40, font_name="ArcadeClassic", shadow=True))

        start_y = 120
        for i, save_name in enumerate(saves):
            cx = self.root.w // 2

            btn = UIButton(
                x=cx - 150,
                y=start_y + (i * 55),
                width=300,
                height=45,
                text=save_name
            )
            btn2 = UIButton(
                x=cx + 160,
                y=start_y + (i * 55),
                width=50,
                height=45,
                text='Del'
            )

            # Lambda binding fix
            btn.on_click = lambda s=save_name: self.load_and_play(s)
            btn2.on_click = lambda s=save_name: self.remove(s)

            self.ui.add_element(btn)
            self.ui.add_element(btn2)

        back_btn = UIButton(x=self.root.w // 2 - 75, y=500, width=150, height=40, text="Back to Title")
        back_btn.on_click = lambda: self.root.set_scene('0')
        self.ui.add_element(back_btn)

    def load_and_play(self, save_name):
        self.root.set_scene('1')
        if '1' in self.root.loaded_scenes:
            world_level = self.root.loaded_scenes['1']
            world_level.load(f"saves/{save_name}")

    def remove(self, save_name):
        try:
            # Safer removal logic could be implemented here
            # For now keeping basic logic but wrapping paths
            import shutil
            save_path = f"saves/{save_name}"
            # If your saves are folders:
            if os.path.isdir(save_path):
                shutil.rmtree(save_path)
                self.refresh_saves()
            # If your saves are files prefix based (legacy way):
            else:
                for file in os.listdir("saves"):
                    if file.startswith(f"{save_name}"):
                        os.remove(f"saves/{file}")
                self.refresh_saves()

        except Exception as e:
            flag(f"Error deleting {save_name}: {e}", 3)


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
    walk_loader = AnimationLoader("assets/animations/Aury/AuryRunning.png", 64, 64, row=0, scale=(1, 1))
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
