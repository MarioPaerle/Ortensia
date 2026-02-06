import os
import random
import time
# FORCE older DirectSound backend before any imports
os.environ['SDL_AUDIODRIVER'] = 'dsound'
del os.environ['SDL_AUDIODRIVER']
import pygame
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)

# Test sound...
# The device name you confirmed works
TARGET_DEVICE = 'Speaker (Realtek(R) Audio)'




def generate_noise(freq):
    # Create 1 second of static
    return bytearray([random.randint(0, 255) for _ in range(freq * 1)])


def test_config(freq, size, channels, buff, description):
    print(f"\n--- TESTING: {description} ---")
    print(f"Params: {freq}Hz, {size}-bit, {channels} chan, buffer {buff}")

    try:
        pygame.mixer.quit()
        pygame.mixer.init(frequency=freq, size=size, channels=channels, buffer=buff, devicename=TARGET_DEVICE)

        # Generate sound tailored to this frequency
        sound_data = generate_noise(freq)
        sound = pygame.mixer.Sound(buffer=sound_data)

        # Play on loop for 4 seconds to give driver time to wake up
        channel = sound.play(loops=-1)
        time.sleep(4)
        channel.stop()
        print("Test finished. Did you hear static?")

    except Exception as e:
        print(f"CRASHED: {e}")


pygame.init()

# TEST 1: The "Wake Up" (Large Buffer + 48k)
# Fixes issues where the driver sleeps or expects DVD quality
test_config(48000, -16, 2, 4096, "High Latency 48k Stereo")

# TEST 2: The "Mono Force" (1 Channel)
# Fixes the 'Surround Sound' bug. If this works, your channels are mapped wrong.
test_config(44100, -16, 1, 1024, "Mono Mode (Force Center)")

# TEST 3: The "Float" (32-bit Audio)
# Some new Realtek drivers refuse 16-bit integer audio
# Note: size=32 is usually integer, but let's try standard 32-bit int first
test_config(44100, 32, 2, 1024, "32-bit High Def")

pygame.quit()