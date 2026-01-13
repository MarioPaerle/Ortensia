import pickle


class WorldState:
    def __init__(self, game_engine):
        self.player_data = {
            'x': game_engine.player.x,
            'y': game_engine.player.y,
            'vx': game_engine.player.vx,
            'vy': game_engine.player.vy,
            'inventory': game_engine.player.inventory,
        }

        self.map_data = game_engine.map_system

        self.camera_pos = (game_engine.main_camera.x, game_engine.main_camera.y)

    def save(self, filename="quicksave.ort"):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename, game_engine):
        with open(filename, 'rb') as f:
            world = pickle.load(f)

        # --- RESTORE PLAYER ---
        # Apply the saved data to the EXISTING player (who has animations loaded)
        p = game_engine.player
        data = world.player_data

        p.x, p.y = data['x'], data['y']
        p.vx, p.vy = data['vx'], data['vy']
        p.inventory = data['inventory']

        # Update hitbox/rect to match new position
        if hasattr(p, 'frect'):
            p.frect.x = p.x + p.coffset_x
            p.frect.y = p.y + p.coffset_y

        # --- RESTORE MAP ---
        game_engine.map_system = world.map_data
        game_engine.map_system.game = game_engine  # Reattach engine

        # Re-link Map to the correct Layer (assuming 'Foreground')
        for layer in game_engine.layers:
            if layer.name == "Foreground":  # Or whatever name you use
                game_engine.map_system.layer = layer
                break

        # Refresh grid
        game_engine.solids = []
        game_engine.solids.append(game_engine.player)

        # Re-add blocks to collision list
        for block in game_engine.map_system.data.values():
            game_engine.solids.append(block)
            # Also ensure they are visually on the layer
            game_engine.map_system.layer.add_static(block)

        game_engine.refresh_grid()
