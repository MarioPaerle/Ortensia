from Graphic._sprites import Block
import os
import random as rd

names = os.listdir('assets/textures/blocks')
RAPIDBLOCKS = {
    n: Block(40, 40, n, texture=f'assets/textures/blocks/{n}',
             physics=False, #True if i%2 == 0 else False,
             # light_emission_color=(rd.randint(200, 255), 30, rd.randint(50, 200)),
             # light_emission_intensity=rd.randint(0, 10)
             )
    for i,n in enumerate(names)
}

print(f'Registered {len(RAPIDBLOCKS)} blocks')
print(RAPIDBLOCKS)