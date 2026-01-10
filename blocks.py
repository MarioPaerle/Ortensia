from Graphic._sprites import Block
import os

names = os.listdir('assets/textures/blocks')
RAPIDBLOCKS = {
    n: Block(32, 32, i, texture=f'assets/textures/blocks/{n}', physics=True if i%2 == 0 else False) for i,n in enumerate(names)
}

print(f'Registered {len(RAPIDBLOCKS)} blocks')