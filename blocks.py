from Graphic._sprites import Block
import os

terra = Block(32, 32, 1, texture='assets/textures/blocks/bleh.png')
names = os.listdir('assets/textures/blocks')
RAPIDBLOCKS = {
    n: Block(32, 32, 1, texture=f'assets/textures/blocks/{n}') for n in names
}