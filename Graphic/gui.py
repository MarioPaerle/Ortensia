import pygame
from typing import Optional, Callable, Tuple
from dataclasses import dataclass
from base import *


class Sprite:
    """Base sprite class compatible with Layer system"""

    def __init__(self, x: float, y: float, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)


class Button(Sprite):
    """Interactive button with hover animation and texture support"""

    def __init__(self, x, y, width: int, height: int,
                 text: str = "",
                 callback: Optional[Callable] = None,
                 texture: Optional[pygame.Surface] = None,
                 bg_color: Tuple[int, int, int] = (70, 70, 70),
                 hover_color: Tuple[int, int, int] = (100, 100, 100),
                 text_color: Tuple[int, int, int] = (255, 255, 255),
                 font_size: int = 20,
                 border_radius: int = 5,
                 screen_width: Optional[int] = None,
                 screen_height: Optional[int] = None):

        self.x_input = x
        self.y_input = y
        self.screen_width = screen_width
        self.screen_height = screen_height

        actual_x = self._calculate_position(x, width, screen_width, is_x=True)
        actual_y = self._calculate_position(y, height, screen_height, is_x=False)

        super().__init__(actual_x, actual_y, width, height)

        self.text = text
        self.callback = callback
        self.texture = texture
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.border_radius = border_radius

        # State
        self.hovered = False
        self.pressed = False
        self.hover_scale = 1.0
        self.target_scale = 1.0

        # Font
        self.font = pygame.font.Font(None, font_size)

        self._render()

    def _calculate_position(self, pos, size, screen_size, is_x=True):
        """Calculate position, handling 'center' keyword"""
        if isinstance(pos, str) and pos.lower() == 'center':
            if screen_size is None:
                # Try to get screen size from pygame display
                display_surf = pygame.display.get_surface()
                if display_surf:
                    screen_size = display_surf.get_width() if is_x else display_surf.get_height()
                else:
                    return 0  # Fallback if no display
            return (screen_size - size) / 2
        return float(pos)

    def _render(self):
        """Render button surface"""
        # Create fresh surface and clear it completely
        self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.surface.fill((0, 0, 0, 0))

        # Apply scale animation
        scale_offset = int((1.0 - self.hover_scale) * self.width * 0.05)
        draw_rect = pygame.Rect(
            scale_offset, scale_offset,
            self.width - scale_offset * 2,
            self.height - scale_offset * 2
        )

        # Background
        color = self.hover_color if self.hovered else self.bg_color

        if self.texture:
            scaled_tex = pygame.transform.scale(self.texture, (draw_rect.width, draw_rect.height))
            self.surface.blit(scaled_tex, draw_rect)
        else:
            pygame.draw.rect(self.surface, color, draw_rect, border_radius=self.border_radius)

        # Text
        if self.text:
            text_surf = self.font.render(self.text, True, self.text_color)
            text_rect = text_surf.get_rect(center=(self.width // 2, self.height // 2))
            self.surface.blit(text_surf, text_rect)

    def update(self, mouse_pos: Tuple[int, int], mouse_pressed: bool, camera_offset: Tuple[float, float] = (0, 0)):
        """Update button state based on mouse input"""
        cx, cy = camera_offset
        mx, my = mouse_pos

        # Adjust for camera and check collision
        btn_rect = pygame.Rect(self.x - cx, self.y - cy, self.width, self.height)
        was_hovered = self.hovered
        self.hovered = btn_rect.collidepoint(mx, my)

        # Animate scale
        self.target_scale = 0.95 if self.hovered else 1.0
        self.hover_scale += (self.target_scale - self.hover_scale) * 0.3

        # Handle click
        if self.hovered and mouse_pressed and not self.pressed:
            self.pressed = True
            if self.callback:
                self.callback()
        elif not mouse_pressed:
            self.pressed = False

        # Re-render if state changed
        if was_hovered != self.hovered or abs(self.hover_scale - self.target_scale) > 0.01:
            self._render()


class Text(Sprite):
    """Enhanced text rendering with custom fonts and styling"""

    def __init__(self, x, y, text: str = "",
                 font_path: Optional[str] = None,
                 font_size: int = 24,
                 color: Tuple[int, int, int] = (255, 255, 255),
                 bg_color: Optional[Tuple[int, int, int, int]] = None,
                 align: str = "left",
                 bold: bool = False,
                 italic: bool = False,
                 antialias: bool = True,
                 shadow: bool = False,
                 shadow_offset: Tuple[int, int] = (3, 3),
                 shadow_color: Tuple[int, int, int] = (0, 0, 0),
                 screen_width: Optional[int] = None,
                 screen_height: Optional[int] = None):

        self.text = text
        self.color = color
        self.bg_color = bg_color
        self.align = align
        self.antialias = antialias
        self.shadow = shadow
        self.shadow_offset = shadow_offset
        self.shadow_color = shadow_color

        # Store original position values and screen dimensions
        self.x_input = x
        self.y_input = y
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Load font
        if font_path:
            self.font = pygame.font.Font(font_path, font_size)
        else:
            self.font = pygame.font.SysFont(None, font_size, bold=bold, italic=italic)

        # Initial render to get dimensions
        temp_surf = self.font.render(text if text else " ", self.antialias, color)

        # Calculate actual position based on 'center' keyword
        actual_x = self._calculate_position(x, temp_surf.get_width(), screen_width, is_x=True)
        actual_y = self._calculate_position(y, temp_surf.get_height(), screen_height, is_x=False)

        super().__init__(actual_x, actual_y, temp_surf.get_width(), temp_surf.get_height())

        self._render()

    def _calculate_position(self, pos, size, screen_size, is_x=True):
        """Calculate position, handling 'center' keyword"""
        if isinstance(pos, str) and pos.lower() == 'center':
            if screen_size is None:
                # Try to get screen size from pygame display
                display_surf = pygame.display.get_surface()
                if display_surf:
                    screen_size = display_surf.get_width() if is_x else display_surf.get_height()
                else:
                    return 0  # Fallback if no display
            return (screen_size - size) / 2
        return float(pos)

    def _render(self):
        """Render text surface"""
        if not self.text:
            self.surface = pygame.Surface((10, self.font.get_height()), pygame.SRCALPHA)
            return

        # Render text
        text_surf = self.font.render(self.text, self.antialias, self.color)

        # Update dimensions
        self.width = text_surf.get_width() + (self.shadow_offset[0] if self.shadow else 0)
        self.height = text_surf.get_height() + (self.shadow_offset[1] if self.shadow else 0)

        # Recalculate position if using 'center'
        self.x = self._calculate_position(self.x_input, self.width, self.screen_width, is_x=True)
        self.y = self._calculate_position(self.y_input, self.height, self.screen_height, is_x=False)

        # Create surface with background
        self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        if self.bg_color:
            self.surface.fill(self.bg_color)

        # Draw shadow
        if self.shadow:
            shadow_surf = self.font.render(self.text, self.antialias, self.shadow_color)
            self.surface.blit(shadow_surf, self.shadow_offset)

        # Draw text
        self.surface.blit(text_surf, (0, 0))

    def set_text(self, text: str):
        """Update text content"""
        if self.text != text:
            self.text = text
            self._render()

    def set_color(self, color: Tuple[int, int, int]):
        """Update text color"""
        if self.color != color:
            self.color = color
            self._render()


class GUIManager:
    """Manages GUI elements and integrates with Layer system"""

    def __init__(self, layer):
        self.layer = layer
        self.buttons = []
        self.texts = []

    def add_button(self, button: Button):
        """Add button to GUI"""
        self.buttons.append(button)
        self.layer.sprites.append(button)
        return button

    def add_text(self, text: Text):
        """Add text to GUI"""
        self.texts.append(text)
        self.layer.sprites.append(text)
        return text

    def update(self, mouse_pos: Tuple[int, int], mouse_pressed: bool, camera_offset: Tuple[float, float] = (0, 0)):
        """Update all GUI elements"""
        for btn in self.buttons:
            btn.update(mouse_pos, mouse_pressed, camera_offset)

    def remove_button(self, button: Button):
        """Remove button from GUI"""
        if button in self.buttons:
            self.buttons.remove(button)
        if button in self.layer.sprites:
            self.layer.sprites.remove(button)

    def remove_text(self, text: Text):
        """Remove text from GUI"""
        if text in self.texts:
            self.texts.remove(text)
        if text in self.layer.sprites:
            self.layer.sprites.remove(text)


# === Example Usage ===
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()

    # Mock Camera class for example

    # Setup
    camera = Camera()
    gui_layer = Layer("GUI", parallax=0.0)
    gui = GUIManager(gui_layer)


    # Create GUI elements
    def on_click():
        print("Button clicked!")


    btn = gui.add_button(Button(
        'center', 250, 200, 50,
        text="Click Me!",
        callback=on_click,
        bg_color=(50, 120, 200),
        hover_color=(70, 150, 230)
    ))

    btn2 = gui.add_button(Button(
        'center', 'center', 180, 45,
        text="Centered Button",
        callback=lambda: print("Centered clicked!"),
        bg_color=(200, 50, 120),
        hover_color=(230, 70, 150)
    ))

    title = gui.add_text(Text(
        'center', 150, "Pygame GUI Library",
        font_size=36,
        color=(255, 200, 50),
        shadow=True
    ))

    subtitle = gui.add_text(Text(
        'center', 200, "Centered Text Demo",
        font_size=24,
        color=(150, 200, 255)
    ))

    running = True
    while running:
        mouse_pressed = pygame.mouse.get_pressed()[0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update GUI
        gui.update(pygame.mouse.get_pos(), mouse_pressed)

        # Render
        screen.fill((30, 30, 40))
        gui_layer.render(screen, camera)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
