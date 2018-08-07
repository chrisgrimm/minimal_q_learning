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



def get_state_max_rewards(states, network):
    state_rewards1, state_rewards2 = [], []
    for state in states:
        rewards = network.get_state_rewards(state)
        max_reward1 = np.max(rewards[:, 0], axis=0)
        max_reward2 = np.max(rewards[:, 1], axis=0)
        state_rewards1.append(max_reward1)
        state_rewards2.append(max_reward2)
    state_rewards1 = cv2.applyColorMap(cv2.resize(255*np.clip(np.array(state_rewards1).reshape([5,5]), 0, 1), (400, 400), interpolation=cv2.INTER_NEAREST).astype(np.uint8), cv2.COLORMAP_JET)
    state_rewards2 = cv2.applyColorMap(cv2.resize(255*np.clip(np.array(state_rewards2).reshape([5,5]), 0, 1), (400, 400), interpolation=cv2.INTER_NEAREST).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite('rewards1.png', state_rewards1)
    cv2.imwrite('rewards2.png', state_rewards2)



