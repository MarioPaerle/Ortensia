from Graphic.base import *
from Graphic.gui import *
from basic import *
from OMusic import AudioSystem

game = Game(1000, 600, title='Ortensia alpha 1', flag=pygame.SCALED | pygame.RESIZABLE)

level = Scene(game)
game.addscene(level, '1')
#######################################################
bg1 = level.add_create_layer("bg1")
uilayer = UILayer()
fg = LitLayer("Foreground", 1, ambient_color=(210, 200, 200))
level.add_layer(fg)
level.add_layer(uilayer)
#######################################################
menu_button = UIButton(10, 20, text="Menù", width=100, height=40)
menu_button.on_click = lambda: game.set_scene('0')
uilayer.add_element(menu_button)
# map = BlockMap(level, fg)
#######################################################

while game.state == 0:
    game.update()
