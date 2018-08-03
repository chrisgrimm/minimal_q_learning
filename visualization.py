import numpy as np
import cv2


def visualize_values(value_set):
    all_grids = []
    for i, values in enumerate(value_set):
        grid_size = np.sqrt(len(values))
        assert int(grid_size) == grid_size
        grid_size = int(grid_size)
        value_grid = np.zeros([grid_size, grid_size])
        for y in range(grid_size):
            for x in range(grid_size):
                val = values[y*grid_size + x]
                value_grid[y, x] = val
        print(f'grid {i}:')
        print(value_grid)
        all_grids.append(value_grid)

def visualize_actions(action_set):
    action_mapping = {
        2:'►', 3:'◄', 4:'■', 1:'▲', 0:'▼'
    }
    for i, actions in enumerate(action_set):
        grid_size = np.sqrt(len(actions))
        assert int(grid_size) == grid_size
        grid_size = int(grid_size)
        grid_string = ''
        for y in range(grid_size):
            for x in range(grid_size):
                grid_string += action_mapping[actions[y*grid_size + x]]
            grid_string += '\n'
        print(f'grid {i}:')
        print(grid_string)





