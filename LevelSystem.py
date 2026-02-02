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
            # Saltiamo i layer UI o quelli non pertinenti
            if isinstance(layer, (Layer, ChunkedLayer, LitLayer)) and layer.name != "UI":
                layer_data = {
                    "name": layer.name,
                    "parallax": layer.parallax,
                    "type": layer.__class__.__name__,
                    "sprites": []
                }

                # Raccogli gli sprite statici (o dinamici se serve)
                # Nota: qui assumiamo che tu voglia salvare solo gli oggetti decorativi
                # e non il Player o i Blocchi (che sono gestiti dal BlockMap)
                sprites_to_save = layer.sprites
                if hasattr(layer, 'large_sprites'):  # Supporto per ChunkedLayer
                    sprites_to_save = sprites_to_save + layer.large_sprites

                # Se usi i chunk, dovresti iterare anche su self.chunks,
                # ma per un editor semplice conviene tenere una lista "editabile" a parte
                # o iterare tutto. Per ora semplifichiamo sugli sprite diretti.

                for sprite in sprites_to_save:
                    # Ignoriamo il Player o entità generate dal codice
                    if hasattr(sprite, 'is_decoration') and sprite.is_decoration:
                        sprite_data = {
                            "x": sprite.x,
                            "y": sprite.y,
                            "texture": sprite.texture_path,  # Devi assicurarti che lo sprite ricordi il path
                            "width": sprite.width,
                            "height": sprite.height,
                            "class": sprite.__class__.__name__
                        }
                        layer_data["sprites"].append(sprite_data)

                data.append(layer_data)

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Livello decorativo salvato in {filename}")

    def load_decorations(self, filename):
        """Carica i layer e gli sprite da JSON."""
        with open(filename, 'r') as f:
            data = json.load(f)

        for layer_conf in data:
            # 1. Crea o Trova il Layer esistente
            # Se il layer esiste già nel main (es. 'fg'), usiamo quello, altrimenti lo creiamo
            existing_layer = next((l for l in self.level.layers if l.name == layer_conf['name']), None)

            if existing_layer:
                current_layer = existing_layer
                current_layer.parallax = layer_conf['parallax']
            else:
                # Creazione dinamica se non esiste
                if layer_conf['type'] == "LitLayer":
                    current_layer = LitLayer(layer_conf['name'], layer_conf['parallax'])
                else:
                    current_layer = ChunkedLayer(layer_conf['name'], layer_conf['parallax'])
                self.level.add_layer(current_layer)

            # 2. Popola gli sprite
            for s_conf in layer_conf['sprites']:
                if s_conf['class'] == 'WigglingSprite':
                    sprite = WigglingSprite(
                        x=s_conf['x'], y=s_conf['y'],
                        w=s_conf['w'], h=s_conf['h'],
                        texture=s_conf['texture'],
                        alpha=True  # O leggi dal json
                        # Aggiungi parametri specifici se salvati
                    )
                else:
                    sprite = Sprite(
                        s_conf['x'], s_conf['y'],
                        s_conf['w'], s_conf['h'],
                        (255, 255, 255),  # Colore dummy
                        texture=s_conf['texture'],
                        alpha=True
                    )

                # Flag essenziale per distinguerli dagli oggetti di gioco
                sprite.is_decoration = True
                sprite.texture_path = s_conf['texture']  # Salviamo il path per il prossimo salvataggio

                if hasattr(current_layer, 'add_dynamic') and s_conf['class'] == 'WigglingSprite':
                    current_layer.add_dynamic(sprite)
                elif hasattr(current_layer, 'add_static'):
                    current_layer.add_static(sprite)
                else:
                    current_layer.sprites.append(sprite)