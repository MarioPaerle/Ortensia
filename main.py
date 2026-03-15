from basic import *
from mapgen import *
from LevelSystem import LevelDataSystem
from DayNight import DayNightCycle, ShadedChunkedLayer
from Graphic._audio import SoundEngine

if __name__ == '__main__':
    game = Game(1200, 600, title='Ortensia Demo', flag=pygame.SCALED | pygame.RESIZABLE | pygame.HWSURFACE, icon="assets/Ortensia1.png")
    sound_engine = SoundEngine()
    from blocks import *

    SOUNDENGINE = sound_engine
    level = WorldLevel(game)
    level.register_blocks(BLOCKS)
    for block in level.registered_blocks:
        level.registered_blocks[block].sound_engine = sound_engine
    game.addscene(level, '1')
    level.stereo_separation = 40
    #######################################################
    bg1 = level.add_create_layer("bg1", 0, layertype=BakedLayer)
    bg2 = level.add_create_layer("bg2", 0.005, layertype=BakedLayer)
    bg3 = level.add_create_layer("bg3", 0.01, layertype=BakedLayer)
    bg4 = level.add_create_layer("bg4", 0.08, layertype=BakedLayer)
    bg5 = level.add_create_layer("bg5", 0.25, layertype=BakedLayer)
    uilayer = UILayer(parallax=0)
    deco_back = LitLayer("DecoBack", parallax=1.0, ambient_color=(80, 80, 130), realized_parallax=0.4)
    fg = ChunkedLayer("Foreground", 1, realized_parallax=1, chunk_size=50)
    deco_front = ChunkedLayer("DecoFront", parallax=1.0, realized_parallax=1.2)

    level.add_layer(deco_back)
    level.add_layer(fg)
    level.add_layer(deco_front)
    fg2 = level.add_create_layer("fg2", 1.3, layertype=ParticleLayer)

    level.sound_engine = sound_engine
    level.music = "assets/music/Ortensia_Playful.mp3"
    level.ambience = "assets/sounds/ambient_crickets1.mp3"

    ####################################################################################################################
    vg = level.add_create_layer("vignette", 0, layertype=Layer)

    vignette = Sprite(0, 0, 1200, 600, texture='assets/textures/vignette.png', alpha=True)
    vg.sprites.append(vignette)
    level.add_layer(uilayer)

    ####################################################################################################################
    if True:
        ff = FireflyEmitter(count=100, size=1)
        fg2.emitters.append(ff)
        # fg2.add_effect(PostProcessing.lumen, 2, 2)
        level.updatables.append(ff)
        for _ in range(100):
            ff.emit(random.randint(0, 3500), random.randint(100, 600), 1)

    ####################################################################################################################
    map = BlockMap(level, layers={
        'back': deco_back,
        'middle': fg,
        'front': deco_front,
    }, tile_size=40)
    map.tick_rate = 2

    level.set_map(map)
    player = Player(50, 200, 24, 64, level=level, cw=16, coffset_x=23, coffset_y=-6)
    player.mode = 1

    player.sound_engine = sound_engine
    player.add_animation('walk_right',
                         load_horizontal_spritesheet("assets/animations/Aury/AuryRunning.png", 64, 64, row=0,
                                                     scale=(1, 1)))
    player.add_animation('idle_right',
                         load_horizontal_spritesheet("assets/animations/Aury/idle_right.png", 64, 64, row=0,
                                                     scale=(1, 1)), speed=2)
    player.add_animation('idle_left',
                         load_horizontal_spritesheet("assets/animations/Aury/idle_right.png", 64, 64, row=0,
                                                     scale=(1, 1), flipx=True), speed=2)
    player.add_animation('walk_left',
                         load_horizontal_spritesheet("assets/animations/Aury/AuryRunning.png", 64, 64, row=0,
                                                     scale=(1, 1), flipx=True))
    player.add_animation('jump_right',
                         load_horizontal_spritesheet("assets/animations/Aury/aury_jumping.png", 64, 64, row=0,
                                                     scale=(1.2, 1.2)), stops=True, speed=8)
    player.add_animation('jump_left',
                         load_horizontal_spritesheet("assets/animations/Aury/aury_jumping.png", 64, 64, row=0,
                                                     scale=(1.2, 1.2), flipx=True), stops=True, speed=8)
    fg.sprites.append(player)
    level.add_player(player)

    loader = LevelDataSystem(level)
    loader.load_decorations("saves/level_decor_data.json")
    for layer in level.layers:
        if isinstance(layer, BakedLayer):
            layer.bake()
    player.die()
    menu_button = UIButton(10, 20, text="Menù", width=100, height=40)
    menu_button.on_click = lambda: game.set_scene('0')
    menu_button2 = UIButton(10, 60, text="Save", width=100, height=40)
    menu_button2.on_click = lambda: level.save('saves/')
    save_as_button = UIButton(10, 100, text="Save As...", width=100, height=40)
    save_as_button.on_click = lambda: game.set_scene('save_as')
    gamemode_button = UIButton(10, 140, text="Switch Gamemode", width=100, height=40)
    gamemode_button.on_click = lambda: player.switch_mode()

    uilayer.add_element(menu_button)
    # uilayer.add_element(menu_button2)
    # uilayer.add_element(save_as_button)
    # uilayer.add_element(gamemode_button)
    ########################################################################################################################

    save_menu = SaveMenu(game)
    game.addscene(save_menu, 'save_selector')
    save_as_scene = SaveAsMenu(game, level)
    game.addscene(save_as_scene, 'save_as')

    while game.state == 0:
        if player:
            cx = player.x + player.width / 2
            cy = player.y + player.height / 2
            sound_engine.set_listener_pos(cx, cy)

        game.update()
