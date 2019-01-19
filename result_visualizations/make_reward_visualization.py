from envs.atari.atari_wrapper import PacmanWrapper, SeaquestWrapper, AtariWrapper, AssaultWrapper
import cv2
import numpy as np
from reward_network import RewardPartitionNetwork
from result_visualizations.generate_occupancy_maps import merge_histogram_and_env_image
import re
import os
import tqdm
import sys

def build_reward_net(env, run_dir, run_name):
    num_rewards = int(re.match(r'^.+?(\d+)reward.+?$', run_name).groups()[0])
    weights_path = os.path.join(run_dir, run_name, 'best_weights')
    reward_net = RewardPartitionNetwork(env, None, None, num_rewards, env.observation_space.shape[0],
                                        env.action_space.n, 'reward_net', traj_len=10, gpu_num=-1,
                                        use_gpu=False, num_visual_channels=9, visual=True)
    reward_net.restore(weights_path, 'reward_net.ckpt')
    return reward_net

class Agent(object):
    def get_action(self, s):
        raise NotImplemented

class HumanAgent(Agent):

    def __init__(self, env : AtariWrapper, mapping):
        self.env = env
        self.mapping = mapping

    def get_action(self, s):
        try:
            a = self.mapping[input('Action: ')]
            return a
        except KeyError:
            return self.get_action(s)

class RandomAgent(Agent):

    def __init__(self, env : AtariWrapper):
        self.env = env

    def get_action(self, s):
        return np.random.randint(0, self.env.action_space.n)


game_coords = [(88,98), (18, 158), (158, 2)]
screen_coords = [(79, 103), (9, 164), (150, 9)]

def get_screen_coords_pacman(game_coords):
    return game_coords[0]-9, game_coords[1]+5

def get_screen_coords_seaquest(game_coords):
    return game_coords[0]+7, game_coords[1]+40

def get_screen_coords_assault(game_coords):
    return game_coords[0]-31, game_coords[1]+219

def make_reward_bins(game_size, bin_radius):
    (height, width, _) = game_size
    assert height % bin_radius == 0
    assert width & bin_radius == 0
    bins = [x for x in range(0, height+1, bin_radius)]
    height_bins = [(x1, x2) for x1, x2 in zip(bins, bins[1:])]
    bins = [x for x in range(0, width+1, bin_radius)]
    width_bins = [(x1, x2) for x1, x2 in zip(bins, bins[1:])]
    #print(width_bins)
    #input('...')
    return {(height_range, width_range): 0.0 for height_range in height_bins
            for width_range in width_bins}

def insert_to_bin(x_pos, y_pos, game_size, bins, bin_radius, reward, get_screen_coords):
    (height, width, _) = game_size
    screen_x, screen_y = get_screen_coords((x_pos, y_pos))
    width_range = ((screen_x // bin_radius)*bin_radius, ((screen_x // bin_radius)+1)*bin_radius)
    height_range = ((screen_y // bin_radius)*bin_radius, ((screen_y // bin_radius)+1)*bin_radius)
    if (height_range, width_range) not in bins:
        print('!!!', x_pos, y_pos, screen_x, screen_y)
        return
    #assert (height_range, width_range) in bins
    bins[(height_range, width_range)] += reward

def visualize_bin(env_image, name, reward_num, reward_bins, game_size):
    (height, width, _) = game_size
    canvas = np.zeros(shape=(height, width))
    for ((h1, h2), (w1, w2)) in reward_bins[0].keys():
        total = np.sum([bin[((h1,h2),(w1,w2))] for bin in reward_bins])
        if total == 0.0:
            perc = 0.0
        else:
            perc = reward_bins[reward_num][((h1,h2),(w1,w2))] / total
        canvas[h1:h2,w1:w2] = perc
    canvas = 255*np.tile(np.reshape(canvas, [height, width, 1]), [1,1,3])
    merged = merge_histogram_and_env_image(canvas, env_image, blur_size=5, heatmap_color=(104,255,3))
    cv2.imwrite(name, merged)


def add_decomposition_to_bins(x_pos, y_pos, decomposed_reward, reward_bins, game_size, bin_radius, get_screen_coords):
    for i, reward in enumerate(decomposed_reward):
        insert_to_bin(x_pos, y_pos, game_size, reward_bins[i], bin_radius, reward, get_screen_coords)


def visualize_all_bins(name, env_image, reward_bins, game_size):
    for i, bin in enumerate(reward_bins):
        visualize_bin(env_image, f'{name}_{i}.png', i, reward_bins, game_size)




def collect_points(name, get_screen_coords, reward_net : RewardPartitionNetwork, agent : Agent, env, num_steps=1000,
                   bin_radius=10):
    s = env.reset()
    env_image = env.get_unprocessed_obs()
    #cv2.imshow('game', env.get_unprocessed_obs())
    #cv2.waitKey(1)
    game_size = env.get_unprocessed_obs().shape

    bins = [make_reward_bins(game_size, bin_radius) for _ in range(reward_net.num_partitions)]

    for i in tqdm.tqdm(range(num_steps)):
        #print(env.get_unprocessed_obs().shape)
        a = agent.get_action(s)
        s, r, t, info = env.step(a)
        #cv2.imshow('game', env.get_unprocessed_obs())
        #cv2.waitKey(1)
        x_pos, y_pos = env.get_agent_position()
        #cv2.imwrite('game_frame.png', env.get_unprocessed_obs())
        #print(x_pos, y_pos)
        if r == 1:
            partitioned_reward = reward_net.get_reward(s, r)
            add_decomposition_to_bins(x_pos, y_pos, partitioned_reward, bins, game_size, bin_radius, get_screen_coords)
            #env.reset()
        #agent_pos = env.get_agent_position()
        #cv2.imshow('game', env.get_unprocessed_obs())
        #cv2.imwrite('game_frame.png', env.get_unprocessed_obs())
        #print(agent_pos)
        #cv2.waitKey(1)
    visualize_all_bins(name, env_image, bins, game_size)

pacman_mapping = {
        '': 0,
        'w': 1,
        'd': 2,
        'a': 3,
        's': 4
    }
seaquest_mapping = {
    '': 0,
    ' ': 1,
    'w': 2,
    'd': 3,
    'a': 4,
    's': 5
}
assault_mapping = {
    '': 0,
    ' ': 1,
    'w': 2,
    'd': 3,
    'a': 4
}

game_context = {
    'assault': (AssaultWrapper, assault_mapping, get_screen_coords_assault),
    'pacman': (PacmanWrapper, pacman_mapping, get_screen_coords_pacman),
    'seaquest': (SeaquestWrapper, seaquest_mapping, get_screen_coords_seaquest)
}

# def compute_run(dest_dir, run_dir, run_name, mode):
#     assert mode in ['random', 'policies']
#     name = re.match(r'^(.+?)\_\d.+?$', run_name).groups()[0]
#     env_cls, env_mapping, get_screen_coords = game_context[name]
#     env = env_cls()
#     reward_net = build_reward_net(env, run_dir, run_name)
#     if mode == 'random':
#         agent = RandomAgent(env)
#         dest_name = os.path.join(dest_dir, run_name)
#         collect_points(dest_name, get_screen_coords, reward_net, agent, env, num_steps=10000, bin_radius=5)

def do_run(run_name):
    name = re.match(r'^(.+?)\_\d.+?$', run_name).groups()[0]
    env_cls, env_mapping, get_screen_coords = game_context[name]
    env = env_cls()
    agent = RandomAgent(env)
    #agent = HumanAgent(env, assault_mapping)
    weights = '/Users/chris/projects/q_learning/new_dqn_results/new_weights'
    reward_net = build_reward_net(env, weights, run_name)
    dest_dir = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/reward_maps'
    collect_points(os.path.join(dest_dir, run_name), get_screen_coords, reward_net, agent, env, num_steps=100000, bin_radius=10)

def make_command():
    paths = ['/Users/chris/projects/q_learning', '/Users/chris/git_downloads/implementations/DL/ICF_simple',
             '/Users/chris/projects/baselines']
    pypath = f'PYTHONPATH={":".join(paths)}'

    runs = ['assault_2reward_10mult_1', 'assault_3reward_10mult_2', 'assault_5reward_10mult_3', 'assault_8reward_10mult_2',
            'seaquest_2reward_10mult_3', 'seaquest_3reward_10mult_4', 'seaquest_5reward_10mult_4', 'seaquest_8reward_10mult_3',
            'pacman_2reward_10mult_2', 'pacman_3reward_10mult_3', 'pacman_5reward_10mult_1', 'pacman_8reward_10mult_2']
    weights = '/Users/chris/projects/q_learning/new_dqn_results/new_weights'

    for run in runs:
        assert os.path.isdir(os.path.join(weights, run))

    commands = []
    for run in runs:
        commands.append(f'{pypath} python /Users/chris/projects/q_learning/result_visualizations/make_reward_visualization.py {run}')
    print('; '.join(commands))


if __name__ == '__main__':
    run_name = sys.argv[1]
    if run_name == 'make_command':
        make_command()
    else:
        do_run(run_name)