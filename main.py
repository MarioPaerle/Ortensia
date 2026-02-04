from basic import *
from mapgen import *
from LevelSystem import LevelDataSystem
from DayNight import DayNightCycle, ShadedChunkedLayer

if __name__ == '__main__':
    game = Game(1200, 600, title='Ortensia Creative Mode', flag=pygame.SCALED | pygame.RESIZABLE)
    from blocks import *

    level = WorldLevel(game)
    level.register_blocks(BLOCKS)
    game.addscene(level, '1')
    level.stereo_separation = 50
    #######################################################
    """bg1 = level.add_create_layer("bg1", parallax=0.001)
    bg2 = level.add_create_layer("bg2", parallax=0.005)
    bg3 = level.add_create_layer("bg3", parallax=0.01)
    bg4 = level.add_create_layer("bg4", parallax=0.05)
    bg5 = level.add_create_layer("bg5", parallax=0.1)
    bg6 = level.add_create_layer("bg6", parallax=1.2)"""

    bg1 = level.add_create_layer("bg1", 0, layertype=Layer)
    bg2 = level.add_create_layer("bg2", 0.005, layertype=Layer)
    bg3 = level.add_create_layer("bg3", 0.01, layertype=Layer)
    bg4 = level.add_create_layer("bg4", 0.08, layertype=Layer)
    bg5 = level.add_create_layer("bg5", 0.25, layertype=Layer)

    # bg6 = level.add_create_layer("bg6", parallax=0.8)
    uilayer = UILayer(parallax=0)
    deco_back = ChunkedLayer("DecoBack", parallax=1.0)
    fg = ShadedChunkedLayer("Foreground", 1)
    deco_front = ChunkedLayer("DecoFront", parallax=1.0)

    # level.add_layer(deco_back)
    level.add_layer(fg)
    # level.add_layer(deco_front)
    level.add_layer(uilayer)
    #######################################################
    map = BlockMap(level, layers={
        'back': deco_back,
        'middle': fg,
        'front': deco_front,
    })
    level.set_map(map)

    # add_plane(map, 0, 10, 300, 1, block=level.registered_blocks['_Death'])

    player = Player(50, 200, 24, 64, level=level, cw=16, coffset_x=23, coffset_y=-6)

    player.add_animation('walk',
                         load_horizontal_spritesheet("Graphic/examples/AuryRunning.png", 64, 64, row=0, scale=(1, 1)))
    fg.sprites.append(player)
    level.add_player(player)

    loader = LevelDataSystem(level)
    loader.load_decorations("saves/level_decor_data.json")

    """
    emitter = FireflyEmitter()
    fg2.emitters.append(emitter)
    level.updatables.append(emitter)
    """
    # level.particle_emitters.append(emitter)
    # level.update_call = lambda : emitter.emit(300, 500, amount=100)

    menu_button = UIButton(10, 20, text="Menù", width=100, height=40)
    menu_button.on_click = lambda: game.set_scene('0')
    menu_button2 = UIButton(10, 60, text="Save", width=100, height=40)
    menu_button2.on_click = lambda: level.save('saves/')
    save_as_button = UIButton(10, 100, text="Save As...", width=100, height=40)
    save_as_button.on_click = lambda: game.set_scene('save_as')
    gamemode_button = UIButton(10, 140, text="Switch Gamemode", width=100, height=40)
    gamemode_button.on_click = lambda: player.switch_mode()

    uilayer.add_element(menu_button)
    uilayer.add_element(menu_button2)
    uilayer.add_element(save_as_button)
    uilayer.add_element(gamemode_button)

    ########################################################################################################################

    save_menu = SaveMenu(game)
    game.addscene(save_menu, 'save_selector')
    save_as_scene = SaveAsMenu(game, level)
    game.addscene(save_as_scene, 'save_as')

    while game.state == 0:
        game.update()
