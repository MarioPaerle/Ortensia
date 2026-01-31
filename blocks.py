from Graphic._sprites import Block, AnimatedBlock
import os
from world_blocks import Deathblock
from Graphic.functions import *

names = os.listdir('assets/textures/blocks')
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
BLOCKS['_Death'] = Deathblock(40, 40, '_Death', texture=f'assets/textures/blocks/deepslate.png')
#######################################################################################################################
Fire = AnimatedBlock(40, 40, 'Fire')
Fire.add_animation('idle', load_vertical_spritesheet('assets/textures/blocks/campfire_fire.png', 32, 32, col='all', scale=(1, 1)))
Fire.set_state('idle')
BLOCKS['Fire'] = Fire
print(f'Registered {len(RAPIDBLOCKS)} blocks')
