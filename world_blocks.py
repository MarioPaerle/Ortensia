from basic import Block
import pygame
from Graphic._layers import UILayer, UIText, UIButton
from Graphic._sprites import Sprite, AnimatedSprite

class Plant(Block):
    pass


class GuiBlock(Block):
    pass


class Deathblock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def on_touch(self, other, dt=1):
        if hasattr(other, 'life'):
            other.take_damage(float('inf'))


import pygame
import random
from Graphic._sprites import Block
from Graphic.functions import flag


import pygame
import random
from Graphic._sprites import Block
from Graphic.functions import flag


class ChestUI:
    """
    The Inventory UI for the Chest.
    Handles interaction between Chest storage and Player inventory.
    """

    def __init__(self, chest, x=None, y=None, rows=3, cols=9, slot_size=40, padding=6):
        self.chest = chest
        self.rows = rows
        self.cols = cols
        self.slot_size = slot_size
        self.padding = padding

        self.player = None  # Reference to player for inventory access
        self.held_item = None  # [Block, count] currently attached to cursor

        # --- Layout Calculation ---
        self.chest_h = rows * (slot_size + padding) + padding
        self.mid_gap = 20
        self.player_h = 1 * (slot_size + padding) + padding

        self.width = cols * (slot_size + padding) + padding
        self.height = 30 + self.chest_h + self.mid_gap + self.player_h + 10

        if x is None:
            self.x = (800 - self.width) // 2
        else:
            self.x = x
        if y is None:
            self.y = (600 - self.height) // 2
        else:
            self.y = y

        # Styling
        self.bg_color = (40, 40, 50, 230)
        self.border_color = (180, 180, 200)
        self.slot_bg_color = (20, 20, 25, 180)
        self.slot_border_color = (80, 80, 90)
        self.text_color = (220, 220, 220)

        self.font = pygame.font.SysFont("Arial", 16, bold=True)
        self.tooltip_font = pygame.font.SysFont("Arial", 14)  # Smaller font for tooltip
        self.title_surf = self.font.render("Chest", True, self.text_color)
        self.inv_surf = self.font.render("Inventory", True, self.text_color)

        self.visible = False

    def set_player(self, player):
        self.player = player

    def close(self):
        """Safely closes the UI and returns held item to player or drops it."""
        self.visible = False

        # If we are holding an item, try to return it to player, otherwise drop (destroy for now)
        if self.held_item and self.player:
            self.held_item = None

        # Remove from player UI layer
        if self.player and self in self.player.uilayer.elements:
            self.player.uilayer.elements.remove(self)

        # Reset chest state visual
        self.chest.state = False
        self.chest.surface = self.chest.textures_loaded[0]
        self.chest.image = self.chest.textures_loaded[0]
        flag("Chest Closed")

    def draw(self, screen):
        if not self.visible: return

        # --- WAILA SUPPRESSION ---
        # Force hide the BlockMap WAILA while this UI is open
        if self.player and hasattr(self.player.level, 'map_system'):
            ms = self.player.level.map_system
            if hasattr(ms, '_waila_alpha'):
                ms._waila_alpha = 0.0
                ms._waila_target_alpha = 0.0

        # 1. Main Background
        pygame.draw.rect(screen, self.bg_color, (self.x, self.y, self.width, self.height), border_radius=6)
        pygame.draw.rect(screen, self.border_color, (self.x, self.y, self.width, self.height), 2, border_radius=6)

        # 2. Chest Section
        screen.blit(self.title_surf, (self.x + 12, self.y + 8))
        chest_start_y = self.y + 35
        self._draw_grid(screen, self.chest.inventory, self.rows, self.cols, self.x + self.padding, chest_start_y)

        # 3. Player Inventory Section
        if self.player and hasattr(self.player, 'slotbar'):
            inv_y = chest_start_y + self.chest_h + self.mid_gap - 15
            screen.blit(self.inv_surf, (self.x + 12, inv_y))
            hotbar_start_y = inv_y + 20
            self._draw_grid(screen, self.player.slotbar.slots, 1, 9, self.x + self.padding, hotbar_start_y)

        # 4. Held Item (Cursor)
        mx, my = pygame.mouse.get_pos()
        if self.held_item:
            item, count = self.held_item
            if item.id != '_None':
                self._draw_item(screen, item, count, mx - self.slot_size // 2, my - self.slot_size // 2)

        # 5. Chest WAILA (Item Tooltip)
        # Check hover for tooltip
        hovered_item = self._get_hovered_item(mx, my, self.chest.inventory, self.rows, self.cols, self.x + self.padding,
                                              chest_start_y)
        if not hovered_item and self.player and hasattr(self.player, 'slotbar'):
            inv_y = chest_start_y + self.chest_h + self.mid_gap + 5  # Adjusted to match click area logic
            hovered_item = self._get_hovered_item(mx, my, self.player.slotbar.slots, 1, 9, self.x + self.padding, inv_y)

        if hovered_item:
            self._draw_tooltip(screen, hovered_item, mx, my)

    def _draw_tooltip(self, screen, item_data, mx, my):
        """Renders a tooltip for the item."""
        if isinstance(item_data, list):
            item, count = item_data
        else:
            item = item_data

        if not item or item.id == '_None':
            return

        text = getattr(item, 'name', item.id)
        text_surf = self.tooltip_font.render(text, True, (255, 255, 255))

        pad = 6
        bg_rect = pygame.Rect(mx + 10, my - 25, text_surf.get_width() + pad * 2, text_surf.get_height() + pad * 2)

        # Ensure tooltip stays on screen
        if bg_rect.right > screen.get_width():
            bg_rect.x -= bg_rect.width + 20

        pygame.draw.rect(screen, (10, 10, 20, 230), bg_rect, border_radius=4)
        pygame.draw.rect(screen, (100, 100, 150), bg_rect, 1, border_radius=4)
        screen.blit(text_surf, (bg_rect.x + pad, bg_rect.y + pad))

    def _get_hovered_item(self, mx, my, collection, rows, cols, start_x, start_y):
        """Similar to click check, but returns the item at the position."""
        rel_x = mx - start_x
        rel_y = my - start_y

        if rel_x < 0 or rel_y < 0: return None

        full_slot_w = self.slot_size + self.padding
        col = int(rel_x // full_slot_w)
        row = int(rel_y // full_slot_w)

        if col < cols and row < rows:
            slot_inner_x = rel_x % full_slot_w
            slot_inner_y = rel_y % full_slot_w

            if slot_inner_x <= self.slot_size and slot_inner_y <= self.slot_size:
                index = row * cols + col
                if index < len(collection):
                    return collection[index]
        return None

    def _draw_grid(self, screen, collection, rows, cols, start_x, start_y):
        for r in range(rows):
            for c in range(cols):
                index = r * cols + c
                if index >= len(collection): break

                sx = start_x + c * (self.slot_size + self.padding)
                sy = start_y + r * (self.slot_size + self.padding)

                # Slot Box
                pygame.draw.rect(screen, self.slot_bg_color, (sx, sy, self.slot_size, self.slot_size), border_radius=4)
                pygame.draw.rect(screen, self.slot_border_color, (sx, sy, self.slot_size, self.slot_size), 1,
                                 border_radius=4)

                # Item
                slot_data = collection[index]
                if slot_data:
                    # Normalize data
                    if isinstance(slot_data, list):
                        item, count = slot_data
                    else:
                        item, count = slot_data, 1

                    # VISUAL FIX: Don't render if it's the "None" block
                    if item and item.id != '_None':
                        self._draw_item(screen, item, count, sx, sy)

    def _draw_item(self, screen, item, count, x, y):
        if hasattr(item, 'surface') and item.surface:
            icon_size = self.slot_size - 6
            icon = pygame.transform.scale(item.surface, (icon_size, icon_size))
            screen.blit(icon, (x + 3, y + 3))

            if count > 1:
                count_surf = self.font.render(str(count), True, (255, 255, 255))
                screen.blit(count_surf, (
                    x + self.slot_size - count_surf.get_width() - 2, y + self.slot_size - count_surf.get_height() - 2))

    def handle_event(self, event):
        if not self.visible: return False

        # --- MOUSE INPUT ---
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            self._process_click(mx, my)
            return True

        # --- CONTROLLER INPUT ---
        # Simulate click with 'A' button (Button 0)
        if event.type == pygame.JOYBUTTONDOWN:
            if event.button == 0:  # A Button -> Click
                mx, my = pygame.mouse.get_pos()
                self._process_click(mx, my)
                return True
            elif event.button == 1: # B Button -> Close
                self.close()
                return True

        # --- KEYBOARD INPUT ---
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.close()
            return True

        return False

    def _process_click(self, mx, my):
        """Processes a click action at screen coordinates (mx, my)."""
        # CLOSE ON OUTSIDE CLICK
        if not (self.x <= mx <= self.x + self.width and self.y <= my <= self.y + self.height):
            self.close()
            return

        # 1. Check Chest Grid
        chest_start_y = self.y + 35
        if self._check_grid_click(mx, my, self.chest.inventory, self.rows, self.cols, self.x + self.padding,
                                  chest_start_y, is_player_inv=False):
            return

        # 2. Check Player Grid
        if self.player:
            inv_y = chest_start_y + self.chest_h + self.mid_gap + 5
            if self._check_grid_click(mx, my, self.player.slotbar.slots, 1, 9, self.x + self.padding, inv_y,
                                      is_player_inv=True):
                return

    def _check_grid_click(self, mx, my, collection, rows, cols, start_x, start_y, is_player_inv):
        rel_x = mx - start_x
        rel_y = my - start_y

        if rel_x < 0 or rel_y < 0: return False

        full_slot_w = self.slot_size + self.padding
        col = int(rel_x // full_slot_w)
        row = int(rel_y // full_slot_w)

        if col < cols and row < rows:
            slot_inner_x = rel_x % full_slot_w
            slot_inner_y = rel_y % full_slot_w

            if slot_inner_x <= self.slot_size and slot_inner_y <= self.slot_size:
                index = row * cols + col
                if index < len(collection):
                    self._interact_with_slot(collection, index, is_player_inv)
                    return True
        return False

    def _get_none_block(self):
        """Safe retrieval of the 'Empty' block placeholder."""
        if self.player and self.player.level:
            return self.player.level.registered_blocks.get('_None', None)
        return None

    def _interact_with_slot(self, collection, index, is_player_inv):
        slot_data = collection[index]

        # --- NORMALIZE INPUT ---
        slot_item, slot_count = None, 0

        if slot_data:
            if isinstance(slot_data, list):
                i, c = slot_data
                if i and i.id != '_None':
                    slot_item, slot_count = i, c
            else:
                if slot_data.id != '_None':
                    slot_item, slot_count = slot_data, 1

        # --- HELPER: SET SLOT ---
        def set_slot(idx, item, count):
            if item is None:
                if is_player_inv:
                    none_blk = self._get_none_block()
                    if none_blk:
                        collection[idx] = [none_blk, 0]
                    else:
                        collection[idx] = None
                else:
                    collection[idx] = None
            else:
                collection[idx] = [item, count]

        # --- LOGIC ---
        if not self.held_item:
            # PICK UP
            if slot_item:
                self.held_item = [slot_item, slot_count]
                set_slot(index, None, 0)  # Clear slot
                flag(f"Picked up {slot_item.id}")
        else:
            # PLACE / SWAP
            held_item, held_count = self.held_item

            if not slot_item:
                # Place into empty
                set_slot(index, held_item, held_count)
                self.held_item = None
                flag("Placed item")
            else:
                # Slot Occupied
                if slot_item.id == held_item.id:
                    # Stack
                    space = 64 - slot_count
                    if space > 0:
                        to_add = min(space, held_count)
                        set_slot(index, slot_item, slot_count + to_add)
                        held_count -= to_add
                        if held_count > 0:
                            self.held_item = [held_item, held_count]
                        else:
                            self.held_item = None
                        flag("Stacked items")
                else:
                    # Swap
                    self.held_item = [slot_item, slot_count]
                    set_slot(index, held_item, held_count)
                    flag("Swapped items")

    def update(self):
        pass


class Chest(Block):
    def __init__(self, w, h, id, textures, slots=27, registeres_blocks=None, *args, **kwargs):
        self.registered_blocks = {} if registeres_blocks is None else registeres_blocks
        self.textures_loaded = [
            pygame.transform.scale(pygame.image.load(t).convert_alpha(), (w, h))
            for t in textures
        ]

        self.state = False

        super().__init__(w, h, id, texture=textures[1], *args, **kwargs)

        self.surface = self.textures_loaded[1]
        self.inventory = [None] * slots
        self.ui = ChestUI(self, rows=3, cols=9)

    def place(self, x, y):

        # Create a basic copy via super (which likely uses copy.copy)
        new_chest = super().place(x, y)

        # Deep initialization for the new instance
        new_chest.inventory = [None] * len(self.inventory)
        new_chest.ui = ChestUI(new_chest, rows=self.ui.rows, cols=self.ui.cols)
        new_chest.state = False

        # Reset visuals for the new instance
        new_chest.surface = self.textures_loaded[1]
        new_chest.image = self.textures_loaded[1]

        return new_chest

    def get_metadata(self):
        return {'inventory': [[i[0].id, i[1]] if i is not None else None for i in self.inventory]}

    def set_metadata(self, metadata):
        self.inventory = [[self.registered_blocks.get(m[0], None), m[1]] if isinstance(m, list) else None for m in metadata.get('inventory', [None] * len(self.inventory))]

    def on_click(self, player):
        if not self.state:
            self.state = True
            self.surface = self.textures_loaded[0]
            self.image = self.textures_loaded[0]
            # self._populate_randomly()

            self.ui.set_player(player)
            if self.ui not in player.uilayer.elements:
                player.uilayer.add_element(self.ui)
            self.ui.visible = True
            flag("Chest Opened")

        else:
            # CLOSING
            self.ui.close()

    def _populate_randomly(self):
        available_blocks = list(self.registered_blocks.values())

        items_to_add = random.randint(3, 6)
        for _ in range(items_to_add):
            block_type = random.choice(available_blocks)
            if block_type.id == '_None': continue

            for i in range(len(self.inventory)):
                if self.inventory[i] is None:
                    count = random.randint(1, 16)
                    self.inventory[i] = [block_type, count]
                    break

    def render(self, layer):
        pass

    def on_place(self, layer, other=None):
        self.layer = layer


class Tool(Sprite):
    def __init__(self, w, h, id, name, texture, breaktypes: set=None):
        super().__init__(x=0, y=0, w=w, h=h, texture=texture, alpha=True)
        self.id = id
        self.name = name
        self.breaking = breaktypes if breaktypes is not None else {"Generic",}

    def on_click(self, other):
        pass

    def on_use(self, othera):
        pass