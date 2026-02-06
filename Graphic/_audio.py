import pygame
import os
import math
import random
import os
import pygame
import time

pygame.mixer.init()

import pygame
import os
import math
import random


class SoundEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SoundEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, frequency=48000, buffer=512):
        if hasattr(self, 'initialized'): return

        # 1. Init Mixer
        if not pygame.mixer.get_init():
            try:
                # Standard init, usually works best without arguments if problems persist
                pygame.mixer.pre_init(frequency=frequency, size=-16, channels=2, buffer=buffer)
                pygame.mixer.init()
            except Exception as e:
                print(f"[SoundEngine] Init Error: {e}")

        # 2. Allocation
        # Channels 0-1: Ambience (Crossfade A/B)
        # Channels 2-3: Music (Crossfade A/B) - UPGRADED from streaming
        # Channels 4-31: SFX
        self.num_channels = 32
        pygame.mixer.set_num_channels(self.num_channels)

        # Reserve first 4 channels so SFX don't interrupt them
        pygame.mixer.set_reserved(4)

        self.ambience_channels = [pygame.mixer.Channel(0), pygame.mixer.Channel(1)]
        self.music_channels = [pygame.mixer.Channel(2), pygame.mixer.Channel(3)]

        # 3. State
        self.sound_cache = {}

        # Ambience State
        self.current_ambience_path = None
        self.ambience_idx = 0

        # Music State
        self.current_music_path = None
        self.music_idx = 0

        self.listener_pos = (0, 0)

        # Settings
        self.master_volume = 1.0
        self.music_volume = 0.5
        self.sfx_volume = 0.8
        self.ambience_volume = 0.1

        self.initialized = True
        print("[SoundEngine] Initialized with Music Crossfading")

    # --- HELPERS ---

    def _crossfade_track(self, path, channels, current_idx, volume_category, fade_ms, loop):
        """
        Generic logic to crossfade between two channels.
        Returns: (new_index, current_path)
        """
        # 1. Identify active and next channels
        active_channel = channels[current_idx]
        next_idx = 1 - current_idx
        next_channel = channels[next_idx]

        # 2. If path is same, do nothing (keep playing)
        #    If you WANT to restart the track, remove this check.
        #    Usually for ambience/music, we want continuity.
        #    We check against a stored path in the caller, but here we just process.

        # 3. Fade out current active
        active_channel.fadeout(fade_ms)

        # 4. Fade in next (if path is valid)
        if path:
            sound = self._get_sound(path)
            if sound:
                target_vol = volume_category * self.master_volume
                next_channel.set_volume(target_vol)
                next_channel.play(sound, loops=loop, fade_ms=fade_ms)
                return next_idx, path
            else:
                return next_idx, None  # Failed to load
        else:
            return next_idx, None  # Stop requested

    def _get_sound(self, path):
        if path not in self.sound_cache:
            if not os.path.exists(path):
                return None
            try:
                sound = pygame.mixer.Sound(path)
                self.sound_cache[path] = sound
            except Exception as e:
                print(f"[SoundEngine] Error loading {path}: {e}")
                return None
        return self.sound_cache[path]

    # --- PUBLIC API ---

    def set_listener_pos(self, x, y):
        self.listener_pos = (x, y)

    def play_music(self, path, fade_ms=2000):
        """
        Crossfades to a new music track using Channels 2 & 3.
        Supports overlapping fades unlike standard mixer.music.
        """
        if path == self.current_music_path: return

        print(f"[SoundEngine] Crossfading Music -> {path}")

        new_idx, new_path = self._crossfade_track(
            path,
            self.music_channels,
            self.music_idx,
            self.music_volume,
            fade_ms,
            loop=-1
        )
        self.music_idx = new_idx
        self.current_music_path = new_path

    def stop_music(self, fade_ms=1000):
        self.play_music(None, fade_ms)  # Fades out current to silence

    def play_ambience(self, path, fade_ms=100):
        """
        Crossfades to a new ambience track using Channels 0 & 1.
        """
        if path == self.current_ambience_path: return

        print(f"[SoundEngine] Crossfading Ambience -> {path}")

        new_idx, new_path = self._crossfade_track(
            path,
            self.ambience_channels,
            self.ambience_idx,
            self.ambience_volume,
            fade_ms,
            loop=-1
        )
        self.ambience_idx = new_idx
        self.current_ambience_path = new_path

    def play_sfx(self, path, pos=None, loop=0, volume=1.0, pitch_variance=0.1, max_dist=800):
        sound = self._get_sound(path)
        if not sound: return
        final_vol = self.sfx_volume * self.master_volume * volume
        left_vol = final_vol
        right_vol = final_vol

        if pos:
            sx, sy = pos
            lx, ly = self.listener_pos
            dist = math.sqrt((sx - lx) ** 2 + (sy - ly) ** 2)
            if dist > max_dist: return

            attenuation = max(0.0, min(1.0, 1.0 - (dist / max_dist)))
            pan = (sx - lx) / 500
            pan = max(-1.0, min(1.0, pan))

            if pan < 0:
                right_vol = (1.0 + pan) * attenuation * final_vol
                left_vol = attenuation * final_vol
            else:
                left_vol = (1.0 - pan) * attenuation * final_vol
                right_vol = attenuation * final_vol

        if pitch_variance > 0:
            v_mod = random.uniform(1.0 - pitch_variance, 1.0)
            left_vol *= v_mod
            right_vol *= v_mod

        channel = pygame.mixer.find_channel()
        if channel:
            channel.set_volume(left_vol, right_vol)
            channel.play(sound, loops=loop)

    def pause_all(self):
        pygame.mixer.pause()

    def unpause_all(self):
        pygame.mixer.unpause()
