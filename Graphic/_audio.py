import pygame
import os
from typing import Dict, Optional, List

DEFAULT_FREQ = 44100
DEFAULT_SIZE = -16
DEFAULT_CHANNELS = 2
DEFAULT_BUFFER = 512


class SoundObject:
    def __init__(self, name, path, sound_obj):
        self.name = name
        self.path = path
        self.sound = sound_obj
        self.base_volume = 1.0