from Graphic._sprites import Block, AnimatedBlock
import os
from world_blocks import Deathblock, Chest, Tool, Book, Ortensia
from Graphic.functions import *

names = [o for o in os.listdir('assets/textures/blocks') if o.endswith('.png')]
print(names)
RAPIDBLOCKS = {
    n: Block(40, 40, n, texture=f'assets/textures/blocks/{n}',
             physics=False, # True if i%2 == 0 else False,
             # light_emission_color=(rd.randint(200, 255), 30, rd.randint(50, 200)),
             # light_emission_intensity=rd.randint(0, 10)
             )
    for i, n in enumerate(names)
}

rocksounds = {'touch': [f"assets/sounds/Footsteps_Rock_Run/{k}" for k in os.listdir('assets/sounds/Footsteps_Rock_Run')],
                'break': ["assets/sounds/rock_break.mp3"]}
BLOCKS = RAPIDBLOCKS
BLOCKS['_None'] = Block(40, 40, '_None')
BLOCKS['deepslate2.png'] = Block(40, 40, 'deepslate2.png',
                                 texture="assets/textures/blocks/deepslate2.png",
                                 name="Ancient Rock",
                                 sounds=rocksounds)
for let in "bcdefghi":
    BLOCKS[f'deepslate2_{let}.png'] = Block(40, 40, f'deepslate2_{let}.png',
                                     texture=f"assets/textures/blocks/deepslate2_{let}.png",
                                     name="Ancient Rock",
                                     sounds=rocksounds)

BLOCKS['deepslate_down.png'].sounds = rocksounds
BLOCKS['BlueOrtensia.png'] = Ortensia(40, 40, 'BlueOrtensia.png',
                                   texture="assets/textures/blocks/BlueOrtensia.png",
                                   name="Blue Ortensia", light_emission_color=(230, 235, 255))
BLOCKS['_Death'] = Deathblock(40, 40, '_Death', texture='assets/textures/blocks/deepslate.png')
BLOCKS['_Light1'] = Block(40, 40, '_Light1',
                          texture='assets/textures/blocks/blank.png',
                          light_emission_color=(180, 180, 255), light_emission_intensity=1)
BLOCKS['_LightBlue'] = Block(40, 40, '_LightBlue',
                          texture='assets/textures/blocks/blank.png',
                          light_emission_color=(100, 100, 255), light_emission_intensity=1.2)
BLOCKS['HighStainedGlass'] = Block(40, 80, 'HighStainedGlass',
                                   texture='assets/textures/blocks/high_stained_glass.png',
                                   light_emission_color=(255, 255, 120), light_emission_intensity=1)
BLOCKS['BigWindow'] = Block(40, 80, 'BigWindow',
                            texture='assets/textures/blocks/BigWindow.png',
                            light_emission_color=(255, 255, 100), light_emission_intensity=3)
BLOCKS['Chest'] = Chest(40, 40, 'Chest', name='Chest',
                        textures=['assets/textures/blocks/chest_opened.png', 'assets/textures/blocks/chest_closed.png'],
                        registeres_blocks=BLOCKS)

BLOCKS['Pickaxe1'] = Tool(40, 40, 'Pickaxe1',
                          'Iron Hammer', "assets/textures/items/iron_hammer.png",
                          description="Can be used to remove Rocks Blocks")
BLOCKS['Peluche'] = Tool(40, 40, 'Peluche', 'Auguri <3  :)', "assets/textures/items/peluche.png")

BLOCKS['Book'] = Book(40, 40, 'Book',
                        texture='assets/textures/items/book.png', text_content=open("assets/books/book1.txt").read())
#######################################################################################################################


print(f'Registered {len(RAPIDBLOCKS)} blocks')
