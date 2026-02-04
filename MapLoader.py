import json
from PIL import Image  # Requires: pip install pillow

# ================= CONFIGURATION =================

# Input image path
INPUT_IMAGE = "my_level_des2.png"


OUTPUT_FILE = "saves/Test/-blockmap_front.json"


COLOR_MAP = {
    (0, 0, 0): "_None",
    (255, 255, 255): "spawnpoint.png",
    (100, 100, 100): "deepslate2.png",
    (100, 100, 170): "BlueOrtensia.png",
    (255, 0, 0): "_Death",
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