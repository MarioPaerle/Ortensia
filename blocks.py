from Graphic._sprites import Block, AnimatedBlock
import os
from world_blocks import Deathblock, Chest
from Graphic.functions import *

names = [o for o in os.listdir('assets/textures/blocks') if o.endswith('.png')]
print(names)
RAPIDBLOCKS = {
    n: Block(40, 40, n, texture=f'assets/textures/blocks/{n}',
             physics=False, #True if i%2 == 0 else False,
             # light_emission_color=(rd.randint(200, 255), 30, rd.randint(50, 200)),
             # light_emission_intensity=rd.randint(0, 10)
             )
    for i, n in enumerate(names)
}
BLOCKS = RAPIDBLOCKS
BLOCKS['_None'] = Block(40, 40, '_None')
BLOCKS['deepslate2.png'] = Block(40, 40, 'deepslate2.png', texture="assets/textures/blocks/deepslate2.png", name="Ancient Rock")
BLOCKS['BlueOrtensia.png'] = Block(40, 40, 'BlueOrtensia.png', texture="assets/textures/blocks/BlueOrtensia.png", name="Blue Ortensia", light_emission_color=(230, 235, 255))
BLOCKS['_Death'] = Deathblock(40, 40, '_Death', texture='assets/textures/blocks/deepslate.png')
BLOCKS['_Light1'] = Block(40, 40, '_Light1', texture='assets/textures/blocks/blank.png', light_emission_color=(180, 180, 255), light_emission_intensity=1)
BLOCKS['HighStainedGlass'] = Block(40, 80, 'HighStainedGlass', texture='assets/textures/blocks/high_stained_glass.png', light_emission_color=(255, 255, 120), light_emission_intensity=1)
BLOCKS['BigWindow'] = Block(40, 80, 'BigWindow', texture='assets/textures/blocks/BigWindow.png', light_emission_color=(255, 255, 100), light_emission_intensity=3)
BLOCKS['Chest'] = Chest(40, 40, 'Chest', name='Chest', textures=['assets/textures/blocks/chest_opened.png', 'assets/textures/blocks/chest_closed.png'])

#######################################################################################################################


print(f'Registered {len(RAPIDBLOCKS)} blocks')
