import pygame


def load_spritesheet(filename, frame_w, frame_h, row='all', scale=(1, 1)):
    if row == 'all':
        sheet = pygame.image.load(filename).convert_alpha()
        frames = []
        for y in range(0, sheet.get_height(), frame_h):
            for x in range(0, sheet.get_width(), frame_w):
                frame = sheet.subsurface(pygame.Rect(x, y, frame_w, frame_h))
                frames.append(frame)
    else:
        sheet = pygame.image.load(filename).convert_alpha()
        frames = []
        for x in range(0, sheet.get_width(), frame_w):
            frame = sheet.subsurface(pygame.Rect(x, row*frame_h, frame_w, frame_h))
            frames.append(frame)

    return [pygame.transform.scale_by(f, factor=scale) for f in frames]

def scale_color(color, factor):
    return [min(255, max(0, int(color[i]*factor))) for i in range(len(color))]

