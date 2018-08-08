import numpy as np

def get_nonintersecting_positions(grid_size, num_positions, existing_positions=set()):
    positions = set()
    while len(positions) < num_positions:
        candidate = tuple(np.random.randint(0, grid_size, size=2))
        if (candidate not in existing_positions) and (candidate not in positions):
            positions.add(candidate)
    return list(positions)