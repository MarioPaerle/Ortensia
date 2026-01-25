import pygame

from Graphic.base import *

import pygame


class FontManager:
    _cache = {}

    @staticmethod
    def get(font_name=None, size=20, bold=False):
        key = (font_name, size, bold)
        if key not in FontManager._cache:
            # Load system font or custom font file
            try:
                if font_name and font_name.endswith(('.ttf', '.otf')):
                    font = pygame.font.Font(font_name, size)
                else:
                    font = pygame.font.SysFont(font_name, size, bold=bold)
                FontManager._cache[key] = font
            except Exception as e:
                print(f"Font Load Error: {e}")
                FontManager._cache[key] = pygame.font.SysFont("Arial", size, bold=bold)

        return FontManager._cache[key]


class UIElement:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.visible = True
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.rect = pygame.Rect(x, y, width, height)

    def handle_event(self, event):
        return False

    def update(self):
        pass

    def draw(self, screen):
        if self.visible:
            screen.blit(self.surface, (self.x, self.y))


class UIText(UIElement):
    def __init__(self, x, y, text="Label", size=24, color=(255, 255, 255),
                 font_name=None, align="left", shadow=False):
        super().__init__(x, y, 1, 1)
        self.text = text
        self.size = size
        self.color = color
        self.font_name = font_name
        self.align = align
        self.shadow = shadow
        self._rendered_text = text
        self.render_text()

    def set_text(self, new_text):
        if new_text != self._rendered_text:
            self.text = new_text
            self.render_text()

    def render_text(self):
        font = FontManager.get(self.font_name, self.size)

        # Render main text
        text_surf = font.render(str(self.text), True, self.color)

        final_w, final_h = text_surf.get_size()

        # Create container surface (account for shadow offset)
        if self.shadow:
            self.surface = pygame.Surface((final_w + 2, final_h + 2), pygame.SRCALPHA)
            shadow_surf = font.render(str(self.text), True, (0, 0, 0, 128))
            self.surface.blit(shadow_surf, (2, 2))
            self.surface.blit(text_surf, (0, 0))
        else:
            self.surface = text_surf

        self.width, self.height = self.surface.get_size()
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self._rendered_text = self.text

        # Alignment adjustments (relative to the x, y point provided)
        if self.align == "center":
            self.x -= self.width // 2
        elif self.align == "right":
            self.x -= self.width


class UIButton(UIElement):
    def __init__(self, x, y, width=150, height=50, text="Button",
                 bg_color=(60, 60, 60), hover_color=(100, 100, 100),
                 text_color=(255, 255, 255), on_click=None, border_radius=5):
        super().__init__(x, y, width, height)
        self.text_str = text
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.on_click = on_click
        self.border_radius = border_radius

        self.is_hovered = False
        self.is_pressed = False

        self.font = FontManager.get(None, size=int(height * 0.5))
        self.render_button()

    def render_button(self):
        # Clear surface
        self.surface.fill((0, 0, 0, 0))

        color = self.hover_color if self.is_hovered else self.bg_color
        if self.is_pressed:
            # Darken slightly when pressed
            color = tuple(max(0, c - 30) for c in color)

        # Draw Background
        pygame.draw.rect(self.surface, color, (0, 0, self.width, self.height),
                         border_radius=self.border_radius)

        # Draw Border
        pygame.draw.rect(self.surface, (200, 200, 200), (0, 0, self.width, self.height),
                         width=2, border_radius=self.border_radius)

        # Draw Text
        text_surf = self.font.render(self.text_str, True, self.text_color)
        text_rect = text_surf.get_rect(center=(self.width // 2, self.height // 2))
        self.surface.blit(text_surf, text_rect)

    def update(self):
        # Check mouse position
        mouse_pos = pygame.mouse.get_pos()
        # We need absolute coordinates for collision
        rect = pygame.Rect(self.x, self.y, self.width, self.height)

        now_hovered = rect.collidepoint(mouse_pos)

        if now_hovered != self.is_hovered:
            self.is_hovered = now_hovered
            self.render_button()  # Re-render only on state change

    def handle_event(self, event):
        if not self.visible: return False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.is_hovered:
                self.is_pressed = True
                self.render_button()
                return True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if self.is_pressed and self.is_hovered:
                    if self.on_click:
                        self.on_click()
                self.is_pressed = False
                self.render_button()

        return False


class UITextInput(UIElement):
    def __init__(self, x, y, width=200, height=40, font_name='minecraftia20', initial_text="new_world"):
        super().__init__(x, y, width, height)
        self.text = initial_text
        self.active = False
        self.color_active = (255, 255, 255)
        self.color_inactive = (150, 150, 150)
        self.font = pygame.font.Font(None, 32)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            return self.active

        if self.active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                self.active = False
            else:
                self.text += event.unicode
            return True
        return False

    def draw(self, screen):
        color = self.color_active if self.active else self.color_inactive
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        screen.blit(text_surf, (self.x + 5, self.y + 10))
        pygame.draw.rect(screen, color, self.rect, 2)
