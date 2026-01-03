import pygame
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time


# ============================================================================
# TILE SYSTEM
# ============================================================================

@dataclass
class Tile:
    """Represents a single tile in the game world"""
    tile_id: int
    x: int
    y: int
    properties: Dict = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class TileRegistry:
    """Central registry for all tile types"""

    def __init__(self):
        self.tiles: Dict[int, Dict] = {}
        self._register_default_tiles()

    def _register_default_tiles(self):
        """Register default tile types"""
        self.register(0, "air", (0, 0, 0, 0), solid=False)
        self.register(1, "ground", (139, 69, 19), solid=True)
        self.register(2, "platform", (160, 82, 45), solid=True, platform=True)
        self.register(3, "brick", (178, 34, 34), solid=True)
        self.register(4, "spike", (255, 0, 0), solid=False, hazard=True)
        self.register(5, "goal", (255, 215, 0), solid=False, goal=True)

    def register(self, tile_id: int, name: str, color: Tuple, **properties):
        """Register a new tile type"""
        self.tiles[tile_id] = {
            "name": name,
            "color": color,
            "properties": properties
        }

    def get(self, tile_id: int) -> Dict:
        """Get tile definition"""
        return self.tiles.get(tile_id, self.tiles[0])


# ============================================================================
# CHUNK SYSTEM (for efficient large maps)
# ============================================================================

class Chunk:
    """16x16 chunk of tiles for efficient loading/unloading"""
    CHUNK_SIZE = 16

    def __init__(self, cx: int, cy: int):
        self.cx = cx
        self.cy = cy
        self.tiles: Dict[Tuple[int, int], Tile] = {}
        self.dirty = False  # Needs saving

    def set_tile(self, x: int, y: int, tile_id: int):
        """Set tile in chunk-local coordinates"""
        if tile_id == 0:
            self.tiles.pop((x, y), None)
        else:
            self.tiles[(x, y)] = Tile(tile_id, x, y)
        self.dirty = True

    def get_tile(self, x: int, y: int) -> Optional[Tile]:
        """Get tile in chunk-local coordinates"""
        return self.tiles.get((x, y))

    def to_dict(self) -> Dict:
        """Serialize chunk"""
        return {
            "cx": self.cx,
            "cy": self.cy,
            "tiles": [(t.x, t.y, t.tile_id, t.properties)
                      for t in self.tiles.values()]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Chunk':
        """Deserialize chunk"""
        chunk = cls(data["cx"], data["cy"])
        for x, y, tid, props in data["tiles"]:
            chunk.tiles[(x, y)] = Tile(tid, x, y, props)
        chunk.dirty = False
        return chunk


# ============================================================================
# MAP SYSTEM
# ============================================================================

class GameMap:
    """Efficient tilemap with chunked storage"""

    def __init__(self, name: str = "untitled"):
        self.name = name
        self.chunks: Dict[Tuple[int, int], Chunk] = {}
        self.metadata = {
            "spawn_x": 0,
            "spawn_y": 0,
            "time_limit": 0,
            "par_time": 0
        }
        self.modified = False

    def _get_chunk_pos(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """Convert world coords to chunk coords"""
        cx = x // Chunk.CHUNK_SIZE
        cy = y // Chunk.CHUNK_SIZE
        lx = x % Chunk.CHUNK_SIZE
        ly = y % Chunk.CHUNK_SIZE
        return cx, cy, lx, ly

    def set_tile(self, x: int, y: int, tile_id: int):
        """Set tile at world coordinates"""
        cx, cy, lx, ly = self._get_chunk_pos(x, y)

        if (cx, cy) not in self.chunks:
            if tile_id == 0:
                return  # Don't create chunk for air
            self.chunks[(cx, cy)] = Chunk(cx, cy)

        self.chunks[(cx, cy)].set_tile(lx, ly, tile_id)
        self.modified = True

    def get_tile(self, x: int, y: int) -> int:
        """Get tile ID at world coordinates"""
        cx, cy, lx, ly = self._get_chunk_pos(x, y)
        chunk = self.chunks.get((cx, cy))
        if chunk:
            tile = chunk.get_tile(lx, ly)
            return tile.tile_id if tile else 0
        return 0

    def get_tiles_in_rect(self, x: int, y: int, w: int, h: int) -> List[Tile]:
        """Get all tiles in a rectangle (for rendering)"""
        tiles = []
        x1, y1 = x // Chunk.CHUNK_SIZE, y // Chunk.CHUNK_SIZE
        x2, y2 = (x + w) // Chunk.CHUNK_SIZE + 1, (y + h) // Chunk.CHUNK_SIZE + 1

        for cy in range(y1, y2):
            for cx in range(x1, x2):
                chunk = self.chunks.get((cx, cy))
                if chunk:
                    for tile in chunk.tiles.values():
                        wx = cx * Chunk.CHUNK_SIZE + tile.x
                        wy = cy * Chunk.CHUNK_SIZE + tile.y
                        if x <= wx < x + w and y <= wy < y + h:
                            tiles.append(Tile(tile.tile_id, wx, wy, tile.properties))
        return tiles

    def clear_empty_chunks(self):
        """Remove empty chunks to save memory"""
        empty = [k for k, v in self.chunks.items() if not v.tiles]
        for k in empty:
            del self.chunks[k]


# ============================================================================
# SAVE/LOAD SYSTEM
# ============================================================================

class SaveManager:
    """Handles saving and loading game states"""

    def __init__(self, save_dir: str = "saves"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.autosave_interval = 30.0  # seconds
        self.last_autosave = time.time()

    def save_map(self, game_map: GameMap, filename: Optional[str] = None):
        """Save map to JSON file"""
        if filename is None:
            filename = f"{game_map.name}.json"

        filepath = os.path.join(self.save_dir, filename)

        data = {
            "name": game_map.name,
            "metadata": game_map.metadata,
            "chunks": [chunk.to_dict() for chunk in game_map.chunks.values()]
        }

        # Atomic write with temp file
        temp_path = filepath + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        os.replace(temp_path, filepath)

        game_map.modified = False
        for chunk in game_map.chunks.values():
            chunk.dirty = False

    def load_map(self, filename: str) -> GameMap:
        """Load map from JSON file"""
        filepath = os.path.join(self.save_dir, filename)

        with open(filepath, 'r') as f:
            data = json.load(f)

        game_map = GameMap(data["name"])
        game_map.metadata = data["metadata"]

        for chunk_data in data["chunks"]:
            chunk = Chunk.from_dict(chunk_data)
            game_map.chunks[(chunk.cx, chunk.cy)] = chunk

        game_map.modified = False
        return game_map

    def autosave(self, game_map: GameMap):
        """Autosave if enough time has passed and map is modified"""
        now = time.time()
        if game_map.modified and (now - self.last_autosave) >= self.autosave_interval:
            self.save_map(game_map, f"{game_map.name}_autosave.json")
            self.last_autosave = now
            return True
        return False

    def list_saves(self) -> List[str]:
        """List all save files"""
        return [f for f in os.listdir(self.save_dir) if f.endswith('.json')]


# ============================================================================
# CAMERA SYSTEM
# ============================================================================

class Camera:
    """Smooth scrolling camera"""

    def __init__(self, width: int, height: int):
        self.x = 0.0
        self.y = 0.0
        self.width = width
        self.height = height
        self.lerp_speed = 0.1

    def follow(self, target_x: float, target_y: float):
        """Smoothly follow target position"""
        target_x -= self.width // 2
        target_y -= self.height // 2

        self.x += (target_x - self.x) * self.lerp_speed
        self.y += (target_y - self.y) * self.lerp_speed

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        return int(x - self.x), int(y - self.y)

    def screen_to_world(self, x: int, y: int) -> Tuple[int, int]:
        """Convert screen coordinates to world coordinates"""
        return int(x + self.x), int(y + self.y)


# ============================================================================
# MAIN GAME CLASS
# ============================================================================

class BuildingGame:
    """Main game class integrating all systems"""

    def __init__(self, width: int = 1280, height: int = 720):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("2D Builder Game")
        self.clock = pygame.time.Clock()

        # Core systems
        self.tile_registry = TileRegistry()
        self.game_map = GameMap("test_level")
        self.save_manager = SaveManager()
        self.camera = Camera(width, height)

        # Game state
        self.running = True
        self.tile_size = 32
        self.selected_tile = 1
        self.player_x = 0.0
        self.player_y = 0.0

        # UI
        self.font = pygame.font.Font(None, 24)

    def handle_events(self):
        """Handle input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                wx, wy = self.camera.screen_to_world(mx, my)
                tx, ty = wx // self.tile_size, wy // self.tile_size

                if event.button == 1:  # Left click - place
                    self.game_map.set_tile(tx, ty, self.selected_tile)
                elif event.button == 3:  # Right click - erase
                    self.game_map.set_tile(tx, ty, 0)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    self.selected_tile = 1
                elif event.key == pygame.K_2:
                    self.selected_tile = 2
                elif event.key == pygame.K_3:
                    self.selected_tile = 3
                elif event.key == pygame.K_4:
                    self.selected_tile = 4
                elif event.key == pygame.K_5:
                    self.selected_tile = 5
                elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.save_manager.save_map(self.game_map)
                    print("Map saved!")
                elif event.key == pygame.K_l and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    try:
                        self.game_map = self.save_manager.load_map("test_level.json")
                        print("Map loaded!")
                    except FileNotFoundError:
                        print("No save file found!")

    def update(self):
        """Update game logic"""
        keys = pygame.key.get_pressed()
        speed = 5.0

        # Camera movement with WASD
        if keys[pygame.K_w]:
            self.player_y -= speed
        if keys[pygame.K_s]:
            self.player_y += speed
        if keys[pygame.K_a]:
            self.player_x -= speed
        if keys[pygame.K_d]:
            self.player_x += speed

        self.camera.follow(self.player_x, self.player_y)

        # Autosave
        self.save_manager.autosave(self.game_map)

    def render(self):
        """Render game"""
        self.screen.fill((50, 50, 50))

        # Calculate visible area
        vis_x = int(self.camera.x // self.tile_size) - 1
        vis_y = int(self.camera.y // self.tile_size) - 1
        vis_w = (self.camera.width // self.tile_size) + 3
        vis_h = (self.camera.height // self.tile_size) + 3

        # Render visible tiles
        tiles = self.game_map.get_tiles_in_rect(vis_x, vis_y, vis_w, vis_h)
        for tile in tiles:
            tile_def = self.tile_registry.get(tile.tile_id)
            sx, sy = self.camera.world_to_screen(
                tile.x * self.tile_size,
                tile.y * self.tile_size
            )
            pygame.draw.rect(
                self.screen,
                tile_def["color"],
                (sx, sy, self.tile_size, self.tile_size)
            )

        # Render grid
        for x in range(vis_x, vis_x + vis_w + 1):
            sx, _ = self.camera.world_to_screen(x * self.tile_size, 0)
            pygame.draw.line(self.screen, (70, 70, 70),
                             (sx, 0), (sx, self.camera.height))
        for y in range(vis_y, vis_y + vis_h + 1):
            _, sy = self.camera.world_to_screen(0, y * self.tile_size)
            pygame.draw.line(self.screen, (70, 70, 70),
                             (0, sy), (self.camera.width, sy))

        # Render player position marker
        px, py = self.camera.world_to_screen(self.player_x, self.player_y)
        pygame.draw.circle(self.screen, (0, 255, 0), (int(px), int(py)), 8)

        # UI
        selected_def = self.tile_registry.get(self.selected_tile)
        ui_text = [
            f"Selected: {selected_def['name']} (Press 1-5)",
            f"Chunks: {len(self.game_map.chunks)}",
            f"Modified: {self.game_map.modified}",
            "WASD: Move | Mouse: Place/Erase | Ctrl+S: Save | Ctrl+L: Load"
        ]
        for i, text in enumerate(ui_text):
            surf = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surf, (10, 10 + i * 25))

        pygame.display.flip()

    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)

        # Save before exit
        if self.game_map.modified:
            self.save_manager.save_map(self.game_map)

        pygame.quit()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    game = BuildingGame()
    game.run()