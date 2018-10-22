import cv2
import numpy as np
import os
import re
import tqdm
from utils import horz_stack_images
from envs.atari.pacman import PacmanWrapper, AssaultWrapper, QBertWrapper, SeaquestWrapper, BreakoutWrapper, AtariWrapper
from baselines.deepq.experiments.training_wrapper import QNetworkTrainingWrapper
from reward_network import RewardPartitionNetwork
import gc


def match_by_color(frame: np.ndarray, colors, threshold):
    last_frame = np.copy(frame[:, :, -3:])
    result = np.zeros(last_frame.shape, dtype=np.uint8)
    for color in colors:
        result[np.where((np.square(last_frame/255. - np.array(color)/255.) < threshold).all(axis=2))] = 255
    return result



pacman_colors = [
    (186, 149, 81),
    (210, 164, 74),
    (166, 136, 87),
    (199, 158, 94),
    (193, 155, 95),
    (165, 136, 96),
    (153, 129, 97),
    (186, 150, 95),
    (128, 112, 100),
]
pacman_threshold = 0.003

assault_colors = [
    #(65, 89, 69),
    (146, 122, 95),
    (112, 82, 49),
    (115, 101, 84),
    (141, 103, 60),
]
assault_threshold = 0.003

qbert_colors = [
    (175, 86, 44),
]
qbert_threshold = 0.005
def compute_policy_histogram(name: str, agent: QNetworkTrainingWrapper, env: AtariWrapper, target_colors, target_threshold):
    s = env.reset()
    histogram = np.zeros(shape=env.observation_space.shape[:2], dtype=np.float32)
    n = 0
    before_start = 0
    num_steps = 1000
    for i in tqdm.tqdm(range(num_steps + before_start)):
        if np.random.uniform(0,1) < 0.01:
            a = np.random.randint(0, env.action_space.n)
        else:
            a = agent.get_action([s])[0]
        s, r, t, info = env.step(a)
        if t:
            env.reset()

        if i < before_start:
            continue
        match = match_by_color(s, target_colors, target_threshold).astype(np.float32)[:,:,0] / 255.
        histogram += match
        n += 1
    z_score = (histogram - np.mean(histogram)) / np.std(histogram)
    #threshold = 5
    #histogram[histogram < threshold] = 0
    #histogram[histogram > threshold] = 1

    normed = z_score - np.min(z_score) / (np.max(z_score) - np.min(z_score))

    histogram = 255*np.tile(np.reshape(normed, list(histogram.shape[:2]) + [1]), [1,1,3])
    cv2.imwrite(name, histogram)

class RandomAgent():
    def __init__(self, num_actions):
        self.n = num_actions

    def get_action(self, s):
        return [np.random.randint(0, self.n)]

def test_color_getter(env, colors, threshold):
    #threshold, colors = (pacman_threshold, pacman_colors)
    #env = PacmanWrapper()
    s = env.reset()
    while True:
        s, r, t, info = env.step(np.random.randint(0, env.action_space.n))
        # cv2.imwrite('sample_image.png', s[:, :, -3:])
        match = match_by_color(s, colors, threshold)
        match = cv2.resize(match, (64, 64))
        s = cv2.resize(s[:, :, -3:], (64, 64))
        both = horz_stack_images(match, s)
        cv2.imwrite('match.png', both)
        input('...')
        #cv2.imshow('match.png', both)

        # stacked = horz_stack_images(match, s[:, :, 6:9], background_color=(255,0,0))
        # cv2.imshow('game', match)
        #cv2.waitKey(1)

game_binding = {'pacman': (pacman_colors, pacman_threshold, PacmanWrapper),
                'assault': (assault_colors, assault_threshold, AssaultWrapper),
                'qbert': (qbert_colors, qbert_threshold, QBertWrapper),
                'seaquest': None,
                'breakout': None}


def compute_occupancy_maps(run_name):
    global game_binding
    match = re.match(r'^(restore\_)?(.+?)\_(\d)reward\_10mult(\_\d)?$', run_name)
    if not match:
        raise Exception(f'Run name {run_name} did not match regex.')
    (restore, game, reward_num, run_num) = match.groups()
    # get the game_binding
    if game not in game_binding:
        # do nothing if the game binding is not present for the run
        return
    (colors, threshold, EnvClass) = game_binding[game]
    env = EnvClass()
    weights_path = os.path.join('/Users/chris/projects/q_learning/new_dqn_results/weights', run_name, 'best_weights')
    reward_net = RewardPartitionNetwork(env, None, None, int(reward_num), env.observation_space.shape[0],
                                        env.action_space.n, 'reward_net', traj_len=10,  gpu_num=-1,
                                        use_gpu=False, num_visual_channels=9, visual=True)

    reward_net.restore(weights_path, 'reward_net.ckpt')
    occupancy_maps_path = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/occupancy_maps'
    for i in range(int(reward_num)):
        agent = reward_net.Q_networks[i]
        compute_policy_histogram(os.path.join(occupancy_maps_path, f'{run_name}_policy{i}.png'), agent, env, colors, threshold)
    #reward_net.clean()


def compute_all_occupancy_maps():
    weights_path = '/Users/chris/projects/q_learning/new_dqn_results/weights'
    all_runs = [x for x in os.listdir(weights_path)
                if re.match(r'^(restore\_)?(.+?)\_(\d)reward\_10mult(\_\d)?$', x)]
    for run in tqdm.tqdm(all_runs):
        compute_occupancy_maps(run)




def generate_command():
    weights_path = '/Users/chris/projects/q_learning/new_dqn_results/weights'
    all_runs = [x for x in os.listdir(weights_path)
                if re.match(r'^(restore\_)?(.+?)\_(\d)reward\_10mult(\_\d)?$', x)]
    command = ''
    for run in all_runs:
        command += f'PYTHONPATH=.:~/projects/baselines python result_visualizations/generate_occupancy_maps.py {run}; '
    print(command)

generate_command()

if __name__ == '__main__':
    import sys
    name = sys.argv[1]
    compute_occupancy_maps(name)
    #compute_all_occupancy_maps()
    #env = QBertWrapper()
    #colors = qbert_colors
    #threshold = qbert_threshold
    #compute_policy_histogram(RandomAgent(env.action_space.n), env, colors, threshold)
    #test_color_getter(env, colors, threshold)
