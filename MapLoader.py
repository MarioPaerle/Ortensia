import json
from PIL import Image  # Requires: pip install pillow

# ================= CONFIGURATION =================

# Input image path
INPUT_IMAGE = "my_level_design.png"

# Output folder (must match where the game looks for saves)
# The game loads from: saves/[SaveName]/-blockmap.json
OUTPUT_FILE = "saves/Test/-blockmap.json"

# MAP COLORS TO BLOCK IDs
# Format: (R, G, B): "Block_ID"
# NOTE: Block IDs in your code seem to be filenames (e.g., 'dirt.png')
# or special names (e.g., '_Death', 'Fire') depending on blocks.py.
COLOR_MAP = {
    (0, 0, 0): "_None",  # Black = Empty/Air (or just skip it)
    (100, 100, 100): "glass.png",  # Gray = Stone
    (255, 0, 0): "_Death",  # Red = Death Block
}


# =================================================

def convert_map():
    img = Image.open(INPUT_IMAGE)
    img = img.convert("RGB")  # Ensure we are working with RGB tuples
    width, height = img.size

    block_data = {}

    print(f"Processing {width}x{height} map...")

    for y in range(height):
        for x in range(width):
            pixel = img.getpixel((x, y))

            # Check if this color corresponds to a block
            if pixel in COLOR_MAP:
                block_id = COLOR_MAP[pixel]

                # Skip empty blocks if you don't want to save air
                if block_id == "_None":
                    continue

                # The game expects keys as string representations of tuples: "(x, y)"
                # and values as the block ID string.
                key = str((x, y))
                block_data[key] = block_id
            else:
                # Optional: Print warning for unknown colors
                print(f"Warning: Unknown color {pixel} at {x},{y}")
                pass

    # Save to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(block_data, f)

    print(f"Success! Map saved to {OUTPUT_FILE}")
    print(f"Total blocks generated: {len(block_data)}")


if __name__ == "__main__":
    convert_map()