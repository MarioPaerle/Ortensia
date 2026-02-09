import json
from PIL import Image  # Requires: pip install pillow
import random

# ================= CONFIGURATION =================

# Input image path
INPUT_IMAGE = "my_level_des2.png"


OUTPUT_FILE = "saves/Opening/-blockmap_back.json"


COLOR_MAP = {
    (0, 0, 0): "_None",
    #(255, 255, 255): "spawnpoint.png",
    (255, 255, 255): "_Light1",
    (255, 255, 0): "HighStainedGlass",
    (100, 100, 100): ["deepslate2.png", "deepslate2.png",   "deepslate2_b.png", "deepslate2_b.png", "deepslate2_c.png", "deepslate2_c.png", "deepslate2_d.png",
                      "deepslate2_e.png", "deepslate2_f.png", "deepslate2_g.png", "deepslate2_i.png", "cobbled_deepslate.png", "cobbled_deepslate_b.png"],
    (50, 50, 50): "deepslate_down.png",
    (25, 25, 25): "black.png",
    (100, 100, 170): "BlueOrtensia.png",
    #(255, 0, 0): "_Death",
    (0, 255, 0): ["pebble1.png", "pebble2.png", "pebble3.png", "pebble4.png"],
    (50, 100, 100): ["deepslate_broken1.png", "deepslate_broken2.png", "deepslate_broken3.png"],
    (255, 0, 220): 'Chest',
    (255, 0, 0): '_LightBlue'
}


def convert_map():
    img = Image.open(INPUT_IMAGE)
    img = img.convert("RGB")  # Ensure we are working with RGB tuples
    width, height = img.size

    block_data = {}

    print(f"Processing {width}x{height} map...")

    for y in range(height):
        for x in range(width):
            pixel = img.getpixel((x, y))

            if pixel in COLOR_MAP:
                if isinstance(COLOR_MAP[pixel], list):
                    block_id = random.choice(COLOR_MAP[pixel])
                else:
                    block_id = COLOR_MAP[pixel]

                if block_id == "_None":
                    continue

                key = str((x, y))
                block_data[key] = block_id
            else:
                print(f"Warning: Unknown color {pixel} at {x},{y}")
                pass

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(block_data, f)

    print(f"Success! Map saved to {OUTPUT_FILE}")
    print(f"Total blocks generated: {len(block_data)}")


if __name__ == "__main__":
    convert_map()