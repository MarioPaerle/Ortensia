from Graphic.base import *
from Graphic.gui import *
from basic import *

game = Game(1000, 600, title='Ortensia alpha 1', flag=pygame.SCALED | pygame.RESIZABLE)

level = WorldLevel(game)
game.addscene(level, '1')
#######################################################
bg1 = level.add_create_layer("bg1", parallax=0.36)
bg2 = level.add_create_layer("bg2", parallax=0.67)
uilayer = UILayer()
fg = LitLayer("Foreground", 1, ambient_color=(210, 200, 200))
level.add_layer(fg)
level.add_layer(uilayer)
#######################################################
menu_button = UIButton(10, 20, text="Menù", width=100, height=40)
menu_button.on_click = lambda: game.set_scene('0')
uilayer.add_element(menu_button)
map = BlockMap(level, fg)
level.set_map(map)

bg1.add_static(
        Sprite(-100, -80, 2304 // 2, 1396 // 2, (40, 40, 80), texture='assets/textures/backgrounds/4.png', alpha=True))
bg2.add_static(
    Sprite(-100, -20, 2304 // 2, 1396 // 2, (40, 40, 80), texture='assets/textures/backgrounds/5.png', alpha=True))
base = SolidSprite(-100, 700, 3000, 10, (40, 40, 40))
fg.add_static(base)
level.solids.append(base)
#######################################################
player = Player(400, 300, 64, 64, game=level, cw=16, coffset_x=23, coffset_y=-6)
walk_loader = AnimationLoader("Graphic/examples/AuryRunning.png", 64, 64, row=0, scale=(1, 1))
player.add_animation('walk', walk_loader)
player.show_hitboxes = True
fg.sprites.append(player)
level.solids.append(player)
level.main_camera.target = player
level.player = player
level.mechaniques.append(player)




while game.state == 0:
    game.update()
