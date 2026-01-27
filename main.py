from basic import *
from mapgen import *

game = Game(1000, 600, title='Ortensia Creative Mode', flag=pygame.SCALED | pygame.RESIZABLE)
from blocks import *

level = WorldLevel(game)
level.registered_blocks = RAPIDBLOCKS
game.addscene(level, '1')

#######################################################
bg1 = level.add_create_layer("bg1", parallax=0.36)
bg2 = level.add_create_layer("bg2", parallax=0.67)
uilayer = UILayer()
fg = LitLayer("Foreground", 1, ambient_color=(210, 200, 200))
level.add_layer(fg)
level.add_layer(uilayer)
#######################################################
map = BlockMap(level, fg)
level.set_map(map)

add_plane(map, 0, 10, 300, 1, block=level.registered_blocks['deepslate.png'])

bg1.add_static(
    Sprite(-100, -80, 2304 // 2, 1396 // 2, (40, 40, 80), texture='assets/textures/backgrounds/4.png', alpha=True))
bg2.add_static(
    Sprite(-100, -20, 2304 // 2, 1396 // 2, (40, 40, 80), texture='assets/textures/backgrounds/5.png', alpha=True))

player = Player(400, 300, 64, 64, level=level, cw=16, coffset_x=23, coffset_y=-6)

walk_loader = AnimationLoader("Graphic/examples/AuryRunning.png", 64, 64, row=0, scale=(1, 1))
player.add_animation('walk', walk_loader)
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
