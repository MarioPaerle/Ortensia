from Graphic.base import *
from Graphic.gui import *
from basic import *

game = Engine(name='Ortensia alpha 1', base_size=(1000, 600), flag=pygame.SCALED | pygame.RESIZABLE)

gui = UILayer()
game.add_layer(gui)


def close_game():
    game.running = False
    exit()

playbutton = UIButton(*game.c_justified_pos(150, 50, dy=100))
playbutton.on_click = close_game
gui.add_element(playbutton)

while game.running:
    game.update()
