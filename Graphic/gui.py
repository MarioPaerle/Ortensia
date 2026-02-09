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

    def draw(self, screen, offset=(0, 0)):
        if self.visible:
            screen.blit(self.surface, (self.x + offset[0], self.y + offset[1]))


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
        self.alpha = 128
        self.render_text()

    def set_text(self, new_text):
        if new_text != self._rendered_text:
            self.text = new_text
            self.render_text()

    def render_text(self):
        self.surface.fill((0, 0, 0, 0))
        color = (*self.color, self.alpha)
        font = FontManager.get(self.font_name, self.size)
        text_surf = font.render(str(self.text), True, color)
        final_w, final_h = text_surf.get_size()

        if self.shadow:
            self.surface = pygame.Surface((final_w + 2, final_h + 2), pygame.SRCALPHA)
            shadow_surf = font.render(str(self.text), True, (0, 0, 0, 128))
            self.surface.blit(shadow_surf, (2, 2))
            self.surface.blit(text_surf, (0, 0))
        else:
            self.surface = text_surf
        if self.alpha < 128:
            self.surface.set_alpha(self.alpha)
        self.width, self.height = self.surface.get_size()
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self._rendered_text = self.text

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


class TypewriterText:
    """Displays text character by character."""

    def __init__(self, x, y, text, size=40, speed=0.08, font_path="Graphic/Minecraftia-Regular.ttf"):
        self.x = x
        self.y = y
        self.text = text
        self.display = ""

        # Font Loading
        try:
            self.font = pygame.font.Font(font_path, size)
        except:
            self.font = pygame.font.SysFont("Arial", size, bold=True)

        self.speed = speed
        self.timer = 0
        self.idx = 0
        self.finished = False
        self.linger_timer = 0
        self.visible = True

    def update(self, dt=0.016):
        if not self.finished:
            self.timer += dt
            if self.timer >= self.speed:
                self.timer = 0
                self.display += self.text[self.idx]
                self.idx += 1
                if self.idx >= len(self.text):
                    self.finished = True
        else:
            self.linger_timer += dt

    def draw(self, screen):
        if not self.visible: return
        surf = self.font.render(self.display, True, (255, 255, 255))
        rect = surf.get_rect(center=(self.x, self.y))

        shadow = self.font.render(self.display, True, (0, 0, 0))
        shadow_rect = shadow.get_rect(center=(self.x + 2, self.y + 2))
        screen.blit(shadow, shadow_rect)
        screen.blit(surf, rect)

    def handle_event(self, event):
        pass


class CreditsRoll:
    """Sliding text for end credits."""

    def __init__(self, w, h, lines, speed=40, font_path="PixeloidSans.ttf"):
        self.w = w
        self.h = h
        self.lines = lines
        self.y_offset = h + 50  # Start below screen
        self.speed = speed
        self.font_path = font_path
        self.font_cache = {}
        self.visible = True

    def update(self, dt=0.016):
        self.y_offset -= self.speed * dt

    def draw(self, screen):
        if not self.visible: return
        current_y = self.y_offset

        for text, size, color in self.lines:
            # Cache fonts to avoid reloading every frame
            if size not in self.font_cache:
                try:
                    self.font_cache[size] = pygame.font.Font(self.font_path, size)
                except:
                    self.font_cache[size] = pygame.font.SysFont("Arial", size, bold=True)

            font = self.font_cache[size]
            surf = font.render(text, True, color)
            rect = surf.get_rect(center=(self.w // 2, current_y))

            # Simple Shadow
            shadow = font.render(text, True, (0, 0, 0))
            s_rect = shadow.get_rect(center=(self.w // 2 + 2, current_y + 2))

            # Culling: Only draw if on screen
            if rect.bottom > 0 and rect.top < self.h:
                screen.blit(shadow, s_rect)
                screen.blit(surf, rect)

            current_y += size + 20  # Line spacing

    def handle_event(self, event):
        pass


class CinematicOverlay:
    """Manages the sequence of animations for the ending."""

    def __init__(self, uilayer, screen_w=1200, screen_h=600):
        self.uilayer = uilayer
        self.w = screen_w
        self.h = screen_h
        self.timer = 0.0
        self.stage = 0

        # Visuals
        self.overlay_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.overlay_alpha = 0
        self.active_elements = []

        self.intro_phrases = [
            "Will you believe in yourself?",
            "You are the only one who can guide you",
            "...",
            "Everytime you take a step into the unknown",
            "Everytime you feel the edge",
            "Everytime you're scared",
            "Everytime you're alone",
            "Will you believe?      "
        ]
        self.phrase_idx = 0

        self.credits_data = [
            ("O R T E N S I A", 100, (150, 255, 150)),
            ("", 50, (0, 0, 0)),
            ("Demo 1", 100, (150, 255, 150)),
            ("", 25, (200, 200, 200)),
            ("demo of a Game By", 25, (200, 200, 200)),
            ("Paerle", 30, (255, 255, 255)),
            ("", 50, (0, 0, 0)),
            ("Music", 30, (200, 200, 200)),
            ("Paerle", 30, (255, 255, 255)),
            ("", 150, (0, 0, 0)),
            ("", 150, (0, 0, 0)),
            ("Thanks for Playing", 60, (255, 255, 200)),
            ("", 50, (255, 255, 200)),
            ("", 50, (255, 255, 200)),
            ("", 50, (255, 255, 200)),
            ("", 50, (255, 255, 200)),
            ("", 50, (255, 255, 200)),
        ]

    def update(self, dt=0.016):
        if dt is None: dt = 0.016 / 3

        self.timer += dt

        if self.stage == 0:
            if self.overlay_alpha < 220:
                self.overlay_alpha += 10 * dt
            else:
                self.stage = 1
                self.timer = 0

        elif self.stage == 1:
            if not self.active_elements:
                if self.phrase_idx < len(self.intro_phrases):
                    tw = TypewriterText(self.w // 2, self.h // 2, self.intro_phrases[self.phrase_idx])
                    self.active_elements.append(tw)
                else:
                    self.stage = 2
            else:
                current = self.active_elements[0]
                current.update(dt)
                if current.finished and current.linger_timer > 2.0:
                    self.active_elements.pop()
                    self.phrase_idx += 1

        elif self.stage == 2:
            if not self.active_elements:
                cr = CreditsRoll(self.w, self.h, self.credits_data)
                self.active_elements.append(cr)
            else:
                self.active_elements[0].update(dt)

    def draw(self, screen):
        if self.overlay_alpha > 0:
            self.overlay_surf.fill((0, 0, 0, min(int(self.overlay_alpha) + 7, 128)))
            screen.blit(self.overlay_surf, (0, 0))

        for el in self.active_elements:
            el.draw(screen)

    def handle_event(self, event):
        pass

