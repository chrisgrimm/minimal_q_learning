import cv2
import numpy as np
import os
import re
import tqdm
from utils import horz_stack_images
from envs.atari.atari_wrapper import PacmanWrapper, AssaultWrapper, QBertWrapper, SeaquestWrapper, BreakoutWrapper, AtariWrapper
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
    #(214, 214, 214),
    #(207, 207, 207),
    #(68, 151, 68),
    #(163, 110, 44),
    (206, 206, 206),
]
assault_threshold = 0.005

qbert_colors = [
    (175, 86, 44),
]
qbert_threshold = 0.005

seaquest_colors = [
    (187, 187, 53)
]
seaquest_threshold = 0.005


def compute_policy_histogram(agent: QNetworkTrainingWrapper, env: AtariWrapper, target_colors, target_threshold):
    s = env.reset()
    s_unprocessed = env.get_unprocessed_obs()
    histogram = np.zeros(shape=s_unprocessed.shape[:2], dtype=np.float32)
    n = 0
    before_start = 0
    num_steps = 1000
    for i in tqdm.tqdm(range(num_steps + before_start)):
        if np.random.uniform(0,1) < 0.01:
            a = np.random.randint(0, env.action_space.n)
        else:
            a = agent.get_action([s])[0]
        s, r, t, info = env.step(a)
        s_unprocessed = env.get_unprocessed_obs()
        if t:
            env.reset()

        if i < before_start:
            continue
        match = match_by_color(s_unprocessed, target_colors, target_threshold).astype(np.float32)[:,:,0] / 255.
        histogram += match
        n += 1
    z_score = np.clip((histogram - np.mean(histogram)) / (0.01*np.std(histogram)), 0, 10)
    #threshold = 5
    #histogram[histogram < threshold] = 0
    #histogram[histogram > threshold] = 1

    normed = (z_score - np.min(z_score)) / (np.max(z_score) - np.min(z_score))

    histogram = (255*np.tile(np.reshape(normed, list(histogram.shape[:2]) + [1]), [1,1,3])).astype(np.uint8)
    cv2.imwrite('./histogram_test2.png', histogram)
    return histogram
    #cv2.imwrite(name, histogram)

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
                'seaquest': (seaquest_colors, seaquest_threshold, SeaquestWrapper),
                }
                #'breakout': None}


def compute_occupancy_maps(run_name):
    print(f'Computing occupancy map for {run_name}...')
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
    weights_path = os.path.join('/Users/chris/projects/q_learning/new_dqn_results/new_weights', run_name, 'best_weights')
    reward_net = RewardPartitionNetwork(env, None, None, int(reward_num), env.observation_space.shape[0],
                                        env.action_space.n, 'reward_net', traj_len=10,  gpu_num=-1,
                                        use_gpu=False, num_visual_channels=9, visual=True)

    reward_net.restore(weights_path, 'reward_net.ckpt')
    occupancy_maps_path = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/occupancy_maps'
    for i in range(int(reward_num)):
        agent = reward_net.Q_networks[i]
        name = os.path.join(occupancy_maps_path, f'{run_name}_policy{i}.png')
        histogram = compute_policy_histogram(agent, env, colors, threshold)
        image = get_environment_image(env)
        merged = merge_histogram_and_env_image(histogram, image)
        cv2.imwrite(name, merged)
    #reward_net.clean()

def get_environment_image(env):
    env.reset()
    num_steps = 100
    for _ in range(num_steps):
        s, r, t, _ = env.step(0)
    return env.get_unprocessed_obs()


def merge_histogram_and_env_image(histogram, env_image):
    histogram = histogram[:, :, [0]]
    heatmap_color = (0, 119, 255)
    heatmap = np.tile([[list(heatmap_color)]], list(env_image.shape[:2]) + [1])
    heatmap_alpha = cv2.applyColorMap(histogram, cv2.COLORMAP_BONE)[:, :, [0]] / 255.
    heatmap_alpha = np.reshape(cv2.blur(heatmap_alpha, (10,10)), list(env_image.shape[:2]) + [1])
    min_image = 0.1
    max_histogram = 1 - min_image
    histogram = np.minimum(histogram / 255., max_histogram)
    grayscale = np.tile(np.reshape(cv2.cvtColor(env_image, cv2.COLOR_BGR2GRAY), list(env_image.shape[:2]) + [1]), [1, 1, 3])
    merged = heatmap_alpha * heatmap + (1 - heatmap_alpha) * grayscale
    return merged


def compute_all_occupancy_maps():
    weights_path = '/Users/chris/projects/q_learning/new_dqn_results/new_weights'
    all_runs = [x for x in os.listdir(weights_path)
                if re.match(r'^(restore\_)?(.+?)\_(\d)reward\_10mult(\_\d)?$', x)]
    for run in tqdm.tqdm(all_runs):
        compute_occupancy_maps(run)




def generate_command(regex):
    weights_path = '/Users/chris/projects/q_learning/new_dqn_results/new_weights'
    all_runs = [x for x in os.listdir(weights_path)
                if re.match(regex, x)]
    command = ''
    for run in all_runs:
        command += f'PYTHONPATH=.:~/projects/baselines python result_visualizations/generate_occupancy_maps.py {run}; '
    print(command)


if __name__ == '__main__':
    import sys
    name = sys.argv[1]
    if name == 'make_command':
        regex = sys.argv[2]
        generate_command(regex)
    else:
        compute_occupancy_maps(name)
    #compute_occupancy_maps('assault_2reward_10mult_1')
    #compute_all_occupancy_maps()
    #env = QBertWrapper()
    #colors = qbert_colors
    #threshold = qbert_threshold
    #compute_policy_histogram(RandomAgent(env.action_space.n), env, colors, threshold)
    #test_color_getter(env, colors, threshold)
