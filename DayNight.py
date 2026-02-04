import math
import pygame
from Graphic._layers import ChunkedLayer


class DayNightCycle:
    def __init__(self, cycle_duration=120.0):
        self.time = 0.0
        self.cycle_duration = cycle_duration
        self.time_scale = 1.0

        self.dawn_color = (255, 200, 150)
        self.day_color = (255, 255, 255)
        self.dusk_color = (255, 150, 100)
        self.night_color = (80, 100, 140)

        self.ambient_brightness = 1.0
        self.current_tint = (255, 255, 255)

    def update(self, dt):
        self.time += dt * self.time_scale
        if self.time >= self.cycle_duration:
            self.time -= self.cycle_duration

        phase = self.time / self.cycle_duration

        if phase < 0.25:
            t = phase / 0.25
            self.current_tint = self._lerp_color(self.night_color, self.dawn_color, t)
            self.ambient_brightness = 0.4 + t * 0.4
        elif phase < 0.5:
            t = (phase - 0.25) / 0.25
            self.current_tint = self._lerp_color(self.dawn_color, self.day_color, t)
            self.ambient_brightness = 0.8 + t * 0.2
        elif phase < 0.75:
            t = (phase - 0.5) / 0.25
            self.current_tint = self._lerp_color(self.day_color, self.dusk_color, t)
            self.ambient_brightness = 1.0 - t * 0.2
        else:
            t = (phase - 0.75) / 0.25
            self.current_tint = self._lerp_color(self.dusk_color, self.night_color, t)
            self.ambient_brightness = 0.8 - t * 0.4

    def _lerp_color(self, c1, c2, t):
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t)
        )

    def get_sun_position(self):
        angle = (self.time / self.cycle_duration) * 2 * math.pi - math.pi / 2
        return angle

    def get_ambient_color(self):
        r = int(self.current_tint[0] * self.ambient_brightness)
        g = int(self.current_tint[1] * self.ambient_brightness)
        b = int(self.current_tint[2] * self.ambient_brightness)
        return (r, g, b)

    def apply_to_layer(self, layer_surface):
        tint_surf = pygame.Surface(layer_surface.get_size())
        tint_surf.fill(self.get_ambient_color())
        layer_surface.blit(tint_surf, (0, 0), special_flags=pygame.BLEND_MULT)

    def get_block_shade(self, block_x, block_y):
        sun_angle = self.get_sun_position()
        sun_x = math.cos(sun_angle)
        sun_y = math.sin(sun_angle)

        if sun_y < 0:
            shade_factor = 0.4 + (1 + sun_y) * 0.4
        else:
            shade_factor = 0.8 + sun_y * 0.2

        horizontal_offset = int(block_x * 0.001)
        shade_variation = math.sin(horizontal_offset + sun_angle) * 0.1

        final_shade = max(0.3, min(1.0, shade_factor + shade_variation))

        tint = self.current_tint
        return (
            int(tint[0] * final_shade),
            int(tint[1] * final_shade),
            int(tint[2] * final_shade)
        )


class ShadedChunkedLayer(ChunkedLayer):
    def __init__(self, name, parallax=1.0, chunk_size=500, day_night_cycle=None):
        super().__init__(name, parallax, chunk_size)
        self.day_night = day_night_cycle
        self.use_per_block_shading = False

    def render(self, screen, camera, emitters=None):
        if not self.visible: return

        screen_w, screen_h = screen.get_size()
        zoom = camera.zoom
        if abs(zoom - camera.target_zoom) < 0.001:
            zoom = camera.target_zoom

        view_w = int(screen_w / zoom)
        view_h = int(screen_h / zoom)
        layer_surf = self._get_view_surface(view_w, view_h)

        cx, cy = camera.x * self.parallax, camera.y * self.parallax

        start_chunk_x = int(cx // self.chunk_size)
        end_chunk_x = int((cx + view_w) // self.chunk_size) + 1
        start_chunk_y = int(cy // self.chunk_size)
        end_chunk_y = int((cy + view_h) // self.chunk_size) + 1

        for x in range(start_chunk_x - 1, end_chunk_x + 1):
            for y in range(start_chunk_y - 1, end_chunk_y + 1):
                if (x, y) in self.chunks:
                    for s in self.chunks[(x, y)]:
                        sx = s.x - cx
                        sy = s.y - cy
                        if -s.width < sx < view_w and -s.height < sy < view_h:
                            if self.day_night and self.use_per_block_shading:
                                shaded = s.surface.copy()
                                shade_color = self.day_night.get_block_shade(s.x, s.y)
                                shade_surf = pygame.Surface(shaded.get_size())
                                shade_surf.fill(shade_color)
                                shaded.blit(shade_surf, (0, 0), special_flags=pygame.BLEND_MULT)
                                layer_surf.blit(shaded, (int(sx), int(sy)))
                            else:
                                layer_surf.blit(s.surface, (int(sx), int(sy)))

        for s in self.large_sprites:
            sx = s.x - cx
            sy = s.y - cy
            if -s.width < sx < view_w and -s.height < sy < view_h:
                layer_surf.blit(s.surface, (int(sx), int(sy)))

        for s in self.sprites:
            sx = s.x - cx
            sy = s.y - cy
            if -s.width < sx < view_w and -s.height < sy < view_h:
                if hasattr(s, 'update'):
                    s.update()
                layer_surf.blit(s.surface, (int(sx), int(sy)))

        if self.day_night and not self.use_per_block_shading:
            self.day_night.apply_to_layer(layer_surf)

        if emitters:
            for emitter in emitters:
                emitter.draw(layer_surf, camera, self.parallax)

        for effect_fn, args in self.effects:
            effect_fn(layer_surf, *args)

        if zoom != 1.0:
            scaled_output = pygame.transform.smoothscale(layer_surf, (screen_w, screen_h))
            screen.blit(scaled_output, (0, 0))
        else:
            screen.blit(layer_surf, (0, 0))