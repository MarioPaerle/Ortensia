import json
import pygame
from Graphic._sprites import Sprite, WigglingSprite
from Graphic._layers import Layer, ChunkedLayer, LitLayer


class LevelDataSystem:
    def __init__(self, level_instance):
        self.level = level_instance

    def save_decorations(self, filename):
        """Salva la configurazione dei layer e degli sprite decorativi."""
        data = []

        # Itera su tutti i layer registrati nel livello
        for layer in self.level.layers:
            # Saltiamo i layer UI
            if isinstance(layer, (Layer, ChunkedLayer, LitLayer)) and layer.name != "UI":
                layer_data = {
                    "name": layer.name,
                    "parallax": layer.parallax,
                    "type": layer.__class__.__name__,
                    "sprites": []
                }

                # Raccogli gli sprite (supporta sia liste normali che chunked/large)
                sprites_to_save = []
                if hasattr(layer, 'sprites'):
                    sprites_to_save.extend(layer.sprites)
                if hasattr(layer, 'large_sprites'):
                    sprites_to_save.extend(layer.large_sprites)
                # Nota: Se usi i chunk attivi, dovresti iterare anche su layer.chunks.values()

                for sprite in sprites_to_save:
                    # Salviamo solo se è marcato come decorazione o ha un path texture
                    if (hasattr(sprite, 'is_decoration') and sprite.is_decoration) or \
                            (hasattr(sprite, 'texture_path') and sprite.texture_path):

                        sprite_data = {
                            "x": sprite.x,
                            "y": sprite.y,
                            "w": sprite.width,
                            "h": sprite.height,
                            "texture": getattr(sprite, 'texture_path', None),
                            "class": sprite.__class__.__name__
                        }
                        # Salva solo se ha una texture valida
                        if sprite_data["texture"]:
                            layer_data["sprites"].append(sprite_data)

                data.append(layer_data)

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Livello decorativo salvato in {filename}")

    def load_decorations(self, filename):
        """Carica i layer e gli sprite da JSON applicando la SCALA corretta."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"File {filename} non trovato.")
            return

        for layer_conf in data:
            # 1. Trova o Crea il Layer
            existing_layer = next((l for l in self.level.layers if l.name == layer_conf['name']), None)

            if existing_layer:
                current_layer = existing_layer
                #current_layer.parallax = layer_conf['parallax']
            else:
                if layer_conf['type'] == "LitLayer":
                    current_layer = LitLayer(layer_conf['name'], layer_conf['parallax'])
                else:
                    # Default a ChunkedLayer o Layer standard
                    current_layer = ChunkedLayer(layer_conf['name'], layer_conf['parallax'])
                self.level.add_layer(current_layer)

            # 2. Popola gli sprite
            for s_conf in layer_conf['sprites']:
                # Supporto per vecchi salvataggi: cerca 'w' oppure 'width'
                w = s_conf.get('w', s_conf.get('width', 64))
                h = s_conf.get('h', s_conf.get('height', 64))
                texture = s_conf.get('texture')
                cls = s_conf.get('class', 'Sprite')

                # Creazione Sprite
                if cls == 'WigglingSprite':
                    sprite = WigglingSprite(
                        x=s_conf['x'] + 1000 * current_layer.parallax, y=s_conf['y'] + 120 * current_layer.parallax,
                        w=w, h=h,
                        texture=texture,
                        alpha=True
                    )
                else:
                    sprite = Sprite(
                        s_conf['x'] + 1000 * current_layer.parallax, s_conf['y'] + 120 * current_layer.parallax,
                        w, h,
                        (255, 255, 255),
                        texture=texture,
                        alpha=True
                    )

                # Impostazioni post-creazione
                sprite.is_decoration = True
                sprite.texture_path = texture  # Importante per ri-salvare

                # Aggiunta al layer
                if hasattr(current_layer, 'add_dynamic') and cls == 'WigglingSprite':
                    current_layer.add_dynamic(sprite)
                elif hasattr(current_layer, 'add_static'):
                    current_layer.add_static(sprite)
                else:
                    current_layer.sprites.append(sprite)