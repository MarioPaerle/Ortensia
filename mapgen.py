def add_plane(map_system, x, y, w, h, block):
    for hh in range(h):
        for ww in range(w):
            map_system.set_tile(x + ww, y + hh, block)
