from basic import *
from mapgen import *

game = Game(1000, 600, title='Ortensia Creative Mode', flag=pygame.SCALED | pygame.RESIZABLE)
from blocks import *

level = WorldLevel(game)
level.register_blocks(BLOCKS)
game.addscene(level, '1')

#######################################################
bg1 = level.add_create_layer("bg1", parallax=0.001)
bg2 = level.add_create_layer("bg2", parallax=0.005)
bg3 = level.add_create_layer("bg2", parallax=0.01)
bg4 = level.add_create_layer("bg2", parallax=0.06)
uilayer = UILayer()
fg = LitLayer("Foreground", 1, ambient_color=(210, 200, 200))
level.add_layer(fg)
level.add_layer(uilayer)
#######################################################
map = BlockMap(level, fg)
level.set_map(map)

add_plane(map, 0, 10, 300, 1, block=level.registered_blocks['_Death'])

bg1.add_static(
    Sprite(-0, -80, 2304 // 2, 1396 // 2, (40, 40, 80), texture='assets/textures/backgrounds/Clouds 1/1.png', alpha=True))
bg2.add_dynamic(
    WigglingSprite(
        x=-60, y=50, w=2304 // 2, h=1396 // 2,
        texture='assets/textures/backgrounds/Clouds 1/2.png',
        alpha=True,
        speed=0.00007,
        distance=60,
        vertical=False
    ))
bg3.add_dynamic(
    WigglingSprite(
        x=0, y=50, w=2304 // 2, h=1396 // 2,
        texture='assets/textures/backgrounds/Clouds 1/3.png',
        alpha=True,
        speed=0.0001,
        distance=35,
        vertical=False
    ))

bg4.add_dynamic(WigglingSprite(
    x=0, y=50, w=2304 // 2, h=1396 // 2,
    texture='assets/textures/backgrounds/Clouds 1/4.png',
    alpha=True,
    speed=0.0001,
    distance=15,
    vertical=False
))


"""
plane = SolidSprite(100, 600, 5000, 20, (40, 40, 40))
fg.add_static(
    plane
)
level.solids.append(plane)
"""

player = Player(50, 200, 64, 64, level=level, cw=16, coffset_x=23, coffset_y=-6)

player.add_animation('walk', load_horizontal_spritesheet("Graphic/examples/AuryRunning.png", 64, 64, row=0, scale=(1, 1)))
fg.sprites.append(player)
level.add_player(player)

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
