import numpy as np
import cv2
import os
import pickle
from utils import horz_stack_images

BASEDIR = os.path.split(os.path.realpath(__file__))[0]


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
        2:'►', 3:'◄', 4:'■', 1:'▲', 0:'▼', None: '+'
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


def produce_two_goal_visualization(network, env, name):
    state_pairs = env.get_all_agent_positions()
    current_object_positions = env.produce_object_positions_from_blocks()
    state_image = env.produce_image(current_object_positions, env.render_mode_image_size)
    state_image = cv2.resize(state_image, (400, 400), interpolation=cv2.INTER_NEAREST)
    data = [[] for _ in range(network.num_partitions)]
    for (x,y), state in state_pairs:
        #print('state', state)
        #print(cv2.imwrite(f'./temp/{x}_{y}.png', state))
        state_reward = network.get_reward(state)
        for i in range(network.num_partitions):
            data[i].append(((x,y), state_reward[i]))
    images = []
    images.append(state_image)
    for partition_state_pairs in data:
        image = produce_reward_image(partition_state_pairs)
        images.append(image)
    stacked = horz_stack_images(*images)

    cv2.imwrite(name, stacked)

# run each policy like 10 times. track the ships that it targets
def produce_assault_ship_histogram_visualization(network, env, name):
    num_episodes_per_policy = 30
    all_hist_arrays = []
    for i in range(network.num_partitions):
        ship1 = ship2 = ship3 = miss = 0
        for j in range(num_episodes_per_policy):
            s = env.reset()
            current_episode_length = 0
            while True:
                a = network.get_state_actions([s])[i][0]
                s, r, t, info = env.step(a)
                if info['internal_terminal']:
                    # if there are two dead ships, something is wrong.
                    ships_alive = info['ship_status']
                    assert sum(1 if not ship_alive else 0 for ship_alive in ships_alive) <= 1
                    if all(ships_alive):
                        miss += 1
                    elif not ships_alive[0]:
                        ship1 += 1
                    elif not ships_alive[1]:
                        ship2 += 1
                    elif not ships_alive[2]:
                        ship3 += 1
                    break
                current_episode_length += 1
        # make a histogram for each policy
        hist_array = np.array([[ship1, ship2, ship3, miss]], dtype=np.float32) / (ship1 + ship2 + ship3 + miss)
        all_hist_arrays.append(hist_array)
    all_hist_arrays = np.concatenate(all_hist_arrays, axis=0)
    all_hist_arrays = (255*all_hist_arrays).astype(np.uint8)
    print(all_hist_arrays.shape, all_hist_arrays.dtype)
    color_map = cv2.applyColorMap(all_hist_arrays, cv2.COLORMAP_JET)
    color_map = cv2.resize(color_map, (200*4, 200*network.num_partitions), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(name, color_map)

def produce_assault_reward_visualization(network, env, name):
    with open(os.path.join('envs/atari/stored_obs_64.pickle'), 'rb') as f:
        obs_samples = pickle.load(f)
    all_partitioned_rewards = []
    for ship_i_examples in obs_samples:
        average_reward = np.mean(network.get_partitioned_reward(ship_i_examples[:10]), axis=0)
        all_partitioned_rewards.append(average_reward)
    all_partitioned_rewards = np.concatenate(all_partitioned_rewards, axis=0)
    all_partitioned_rewards = (255*all_partitioned_rewards).astype(np.uint8)
    all_partitioned_rewards = cv2.applyColorMap(all_partitioned_rewards, cv2.COLORMAP_JET)
    all_partitioned_rewards = cv2.resize(all_partitioned_rewards, (200*len(obs_samples), 200*network.num_partitions), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(name, all_partitioned_rewards)



def produce_reward_image(partition_state_pairs):
    canvas = np.zeros(shape=(5,5), dtype=np.float32)
    for (x,y), reward in partition_state_pairs:
        canvas[y, x] = reward
    color_map = cv2.applyColorMap((255*np.clip(np.array(canvas), 0, 1)).astype(np.uint8), cv2.COLORMAP_JET)
    color_map = cv2.resize(color_map, (400, 400), interpolation=cv2.INTER_NEAREST)
    return color_map








def apply_color_scheme(state_rewards, blank_state_indices, blank_color=(0,0,0)):
    color_map = cv2.applyColorMap((255*np.clip(np.array(state_rewards), 0, 1)).astype(np.uint8), cv2.COLORMAP_JET)
    for i in blank_state_indices:
        color_map[i, :] = blank_color
    color_map = cv2.resize(np.reshape(color_map, (5, 5, 3)), (400, 400), interpolation=cv2.INTER_NEAREST)
    return color_map

def get_state_max_rewards(states, network):
    state_rewards1, state_rewards2 = [], []
    blank_state_indices = []
    for i, state in enumerate(states):
        if state is None:
            blank_state_indices.append(i)
            max_reward1 = max_reward2 = 0
        else:
            rewards = network.get_state_rewards(state)
            max_reward1 = np.max(rewards[:, 0], axis=0)
            max_reward2 = np.max(rewards[:, 1], axis=0)
        state_rewards1.append(max_reward1)
        state_rewards2.append(max_reward2)
    state_rewards1 = apply_color_scheme(state_rewards1, blank_state_indices)
    state_rewards2 = apply_color_scheme(state_rewards2, blank_state_indices)
    #state_rewards1 = cv2.applyColorMap(cv2.resize(255*np.clip(np.array(state_rewards1).reshape([5,5]), 0, 1), (400, 400), interpolation=cv2.INTER_NEAREST).astype(np.uint8), cv2.COLORMAP_JET)
    #state_rewards2 = cv2.applyColorMap(cv2.resize(255*np.clip(np.array(state_rewards2).reshape([5,5]), 0, 1), (400, 400), interpolation=cv2.INTER_NEAREST).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite('rewards1.png', state_rewards1)
    cv2.imwrite('rewards2.png', state_rewards2)



