"""
Example of how to use physics-enabled blocks in your game.
This demonstrates creating falling blocks for bridge-building mechanics.
"""

import pygame
from Graphic.base import *
from Graphic.functions import *
from basic import *

game = Engine(name='Physics Block Demo', base_size=(1000, 600))

s = game.scaler

# Create layers
fg = game.add_create_layer("Foreground", 1.0)
terrain_layer = game.add_create_layer("Terrain", 1.0)

# Create block types
# Normal solid block (no physics)
solid_block = Block(32, 32, 1, texture='assets/textures/blocks/clock.png')
solid_block.physics_block = False  # Static, won't fall

# Physics-enabled block (will fall with gravity)
falling_block = Block(32, 32, 2, texture='assets/textures/blocks/clock.png')
falling_block.physics_block = True  # Enable gravity
falling_block.gravity = 980  # Adjust fall speed (pixels/sec²)
falling_block.max_fall_speed = 800  # Terminal velocity

# Light block (falls slower - good for bridge building)
light_block = Block(32, 32, 3, texture='assets/textures/blocks/clock.png')
light_block.physics_block = True
light_block.gravity = 500  # Falls slower
light_block.max_fall_speed = 400

# Create BlockMap with your chosen block
map_system = BlockMap(game, terrain_layer, tile_size=s(32))

# Create player
player = Player(s(400), s(300), s(64), s(64), game=game, cw=48)
player.add_animation('walk', load_spritesheet("Graphic/examples/AuryRunning.png", 64, 64, row=0))
fg.sprites.append(player)
game.solids.append(player)
game.main_camera.target = player
game.player = player

# Add a base platform (static, no physics)
base = SolidSprite(0, 700, 3000, 32, (100, 100, 100))
fg.add_static(base)
game.solids.append(base)

# Create a gap to build a bridge over
left_platform = SolidSprite(100, 600, 200, 32, (150, 150, 150))
right_platform = SolidSprite(600, 600, 200, 32, (150, 150, 150))
fg.add_static(left_platform)
fg.add_static(right_platform)
game.solids.append(left_platform)
game.solids.append(right_platform)

# Current block selection
current_block_type = falling_block  # Start with physics block
block_types = [solid_block, falling_block, light_block]
current_block_index = 1


def update_logic(dt):
    """Main game loop"""
    global current_block_type, current_block_index

    keys = pygame.key.get_pressed()
    mouse_buttons = pygame.mouse.get_pressed()
    mx, my = pygame.mouse.get_pos()

    # Player controls
    player.mechaniches(keys, dt)

    # Switch block types with number keys
    if keys[pygame.K_1]:
        current_block_index = 0
        current_block_type = block_types[0]
    elif keys[pygame.K_2]:
        current_block_index = 1
        current_block_type = block_types[1]
        print('switched!')
    elif keys[pygame.K_3]:
        current_block_index = 2
        current_block_type = block_types[2]

    # Place blocks with left click
    if mouse_buttons[0]:
        map_system.place_tile(mx, my, block=current_block_type)
        game.refresh_grid()

    # Remove blocks with right click
    if mouse_buttons[2]:
        map_system.remove_tile(mx, my)
        game.refresh_grid()

    map_system.update(dt)

    # Display current block type
    block_names = ["Solid (No Physics)", "Heavy Falling", "Light Falling"]
    """print(f"\rCurrent Block: {block_names[current_block_index]} | "
          f"Physics Blocks Active: {len(map_system.physics_blocks)}", end="")"""


# Run the game
game.refresh_grid()
while game.running:
    dt = game.dt()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game.running = False

    update_logic(dt)

    game.main_camera.update()
    game.screen.fill((30, 30, 40))

    # Render all layers
    for layer in game.layers:
        layer.render(game.screen, game.main_camera)

    fps = game.clock.get_fps()
    pygame.display.set_caption(f"Physics Blocks | FPS: {int(fps)} | "
                               f"1/2/3: Switch Block Type")
    pygame.display.flip()

pygame.quit()