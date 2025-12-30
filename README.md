# Ortensia

![Ortensia1.png](examples%2FOrtensia1.png)
> [!WARNING]
> OrtensiaLIB is just born, so its full of bug, and misses a documentation

Ortensia is a Game I wanna realize, from soundtracks, to graphics, and of course gameplay, based on OrtensiaLIB.
OrtensiaLIB will feature small lightweight libraries that will help me, and maybe others develop games.
- Graphic (CPU Bounded simple yet fast visual effects)
- Animations
- Automatic Music Scoring
- Lightweigh AI NPCs
It aims to make simple game development easy wrapping pygame-ce.
It will support simple graphic effects, with quality adjustment, all CPU computed leveraging 
numpy vectorized operations.
Collision grids are being developed.\
Simple GUIs creation as well.

# Examples of OrtensiaLIB
**Ortensia** can support high frame rate (400+ on intel 5 12th), here are two examples working on mid and low settings,
using the Bloom effect and a Particle Emitter, Animations and Water Simulation all on single thread CPU

GIF quality is of course very bad, but you can imagine come on:
 
## Animation, Particle Emission, Lumen Effect, Water Simulation and parallax
![OrtensiaWaterUpdate.gif](examples%2FOrtensiaWaterUpdate.gif)

## Subwater Effect added on top, and we're still at around 95 FPS with max quality
![ortensiasubwatery.gif](examples%2Fortensiasubwatery.gif)

## Better Quality _Lumen_ effect, on Animated sprite, and camera shake (@ 200FPS) mid quality
![OrtensiaHQlumen.gif](examples%2FOrtensiaHQlumen.gif)

### Mid Quality _Bloom_:
![ortensiagoodgraphicsgif.gif](examples%2Fortensiagoodgraphicsgif.gif)

### Low Quality _Bloom_:
![ortensiabadquality.gif](examples%2Fortensiabadquality.gif)
