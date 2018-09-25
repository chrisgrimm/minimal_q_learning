from random import choice
import numpy as np
from replay_buffer import ReplayBuffer
from envs.block_world.block_pushing_domain import BlockPushingDomain
from envs.atari.pacman import PacmanWrapper
from envs.atari.simple_assault import SimpleAssault
from reward_network import RewardPartitionNetwork
from visualization import produce_two_goal_visualization, produce_assault_ship_histogram_visualization, produce_assault_reward_visualization, produce_reward_statistics, visualize_all_representations_all_reward_images
import argparse
from utils import LOG, build_directory_structure
from reward_prob_tracker import RewardProbTracker
import argparse
from random import choice
import cv2
import os
from replay_buffer import StateReplayBuffer

from envs.atari.threaded_environment import ThreadedEnvironment
from envs.block_world.block_pushing_domain import BlockPushingDomain
from replay_buffer import ReplayBuffer
from reward_network import RewardPartitionNetwork
from utils import LOG, build_directory_structure

parser = argparse.ArgumentParser()
screen_name = None
if 'STY' in os.environ:
    screen_name = ''.join(os.environ['STY'].split('.')[1:])
    parser.add_argument('--name', type=str, default=screen_name)
else:
    parser.add_argument('--name', type=str, required=True)

parser.add_argument('--reuse-visual', action='store_true')
parser.add_argument('--traj-len', type=int, required=True)
parser.add_argument('--max-value-mult', type=float, required=True)
parser.add_argument('--dynamic-weighting-disentangle', action='store_true')
parser.add_argument('--mode', type=str, required=True, choices=['SOKOBAN', 'ASSAULT', 'PACMAN'])
parser.add_argument('--visual', action='store_true')
parser.add_argument('--learning-rate', type=float, required=True)
parser.add_argument('--gpu-num', type=int, required=True)
parser.add_argument('--separate-reward-repr', action='store_true')
parser.add_argument('--bayes-reward-filter', action='store_true')
parser.add_argument('--use-ideal-filter', action='store_true')
parser.add_argument('--num-pacman-partitions', type=int, default=2)
args = parser.parse_args()

use_gpu = args.gpu_num >= 0
mode = args.mode
visual = args.visual

observation_mode = 'image' if visual else 'vector'




if mode == 'ASSAULT':
    num_partitions = 3
    num_visual_channels = 9


    def run_assault_visualizations(network, env, name):
        [path, name] = os.path.split(name)
        [name, name_extension] = name.split('.')
        hist_full_name = os.path.join(path, name + '_hist') + '.' + name_extension
        reward_full_name = os.path.join(path, name + '_reward') + '.' + name_extension
        statistics_full_name = os.path.join(path, name+'_statistics.txt')
        behavior_full_name = os.path.join(path, name+'_behavior_file.pickle')
        #produce_assault_ship_histogram_visualization(network, env, hist_full_name)
        #produce_assault_reward_visualization(network, env, reward_full_name)
        produce_reward_statistics(network, env, statistics_full_name, behavior_full_name)
        #cv2.imwrite(os.path.join(path, f'{name}_thres.{name_extension}'), 255*np.tile(network.threshold, [1,1,3]))



    def on_reward_print_func(r, sp, info, network, reward_buffer):
        partitioned_r = network.get_partitioned_reward([sp], [r])[0]
        print(reward_buffer.length(), partitioned_r, info['ship_status'])

    visualization_func = run_assault_visualizations
    # visual mode must be on for Assault domain.
    assert visual
    env = SimpleAssault(initial_states_file=None)
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: SimpleAssault(initial_states_file=None),
                                            SimpleAssault)
    dummy_env = SimpleAssault(initial_states_file=None)
elif mode == 'SOKOBAN':
    num_partitions = 2
    num_visual_channels = 3
    visualization_func = produce_two_goal_visualization

    def on_reward_print_func(r, sp, info, network, reward_buffer):
        partitioned_r = network.get_partitioned_reward([sp], [r])[0]
        print(r, partitioned_r)

    env = BlockPushingDomain(observation_mode=observation_mode)
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: BlockPushingDomain(observation_mode=observation_mode),
                                            BlockPushingDomain)
    dummy_env = BlockPushingDomain(observation_mode=observation_mode)
elif mode == 'PACMAN':
    num_partitions = args.num_pacman_partitions
    num_visual_channels = 9
    def run_assault_visualizations(network, env, name):
        [path, name] = os.path.split(name)
        [name, name_extension] = name.split('.')
        hist_full_name = os.path.join(path, name + '_hist') + '.' + name_extension
        reward_full_name = os.path.join(path, name + '_reward') + '.' + name_extension
        statistics_full_name = os.path.join(path, name+'_statistics.txt')
        behavior_full_name = os.path.join(path, name+'_behavior_file.pickle')
        #produce_assault_ship_histogram_visualization(network, env, hist_full_name)
        #produce_assault_reward_visualization(network, env, reward_full_name)
        produce_reward_statistics(network, env, statistics_full_name, behavior_full_name)
        #cv2.imwrite(os.path.join(path, f'{name}_thres.{name_extension}'), 255*np.tile(network.threshold, [1,1,3]))
    visualization_func = run_assault_visualizations
    def on_reward_print_func(r, sp, info, network, reward_buffer):
        partitioned_r = network.get_partitioned_reward([sp], [r])[0]
        print(r, partitioned_r)
    env = PacmanWrapper()
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: PacmanWrapper(),
                                            PacmanWrapper)
    dummy_env = PacmanWrapper()

else:
    raise Exception(f'mode must be in {mode_options}.')

build_directory_structure('.', {'runs': {
                                    args.name: {
                                        'images': {},
                                        'weights': {}}}})
LOG.setup(f'./runs/{args.name}')
save_path = os.path.join('runs', args.name, 'weights')

#agent = QLearnerAgent(env.observation_space.shape[0], env.action_space.n)
buffer = ReplayBuffer(100000)

reward_buffer = ReplayBuffer(100000)
state_replay_buffer = StateReplayBuffer(100000)

reward_net = RewardPartitionNetwork(buffer, reward_buffer, state_replay_buffer, num_partitions, env.observation_space.shape[0],
                                    env.action_space.n, 'reward_net', traj_len=args.traj_len,  gpu_num=args.gpu_num,
                                    use_gpu=use_gpu, num_visual_channels=num_visual_channels, visual=visual,
                                    max_value_mult=args.max_value_mult, use_dynamic_weighting_disentangle_value=args.dynamic_weighting_disentangle,
                                    lr=args.learning_rate, reuse_visual_scoping=args.reuse_visual, separate_reward_repr=args.separate_reward_repr,
                                    use_ideal_threshold=args.use_ideal_filter)

(height, width, depth) = env.observation_space.shape
tracker = RewardProbTracker(height, width, depth)



batch_size = 32
epsilon = 0.1
episode_reward = 0
print(env.action_space)
epsilon = 1.0
min_epsilon = 0.1
num_epsilon_steps = 100000
min_reward_experiences = 500
num_reward_steps = 30000
save_freq = 1000
current_reward_training_step = 0 if args.separate_reward_repr else num_reward_steps
epsilon_delta = (epsilon - min_epsilon) / num_epsilon_steps
i = 0

# indices of current policy
policy_indices = list(range(num_partitions)) + [-1]
current_policy = choice(policy_indices)

def get_action(s):
    global pre_training
    global current_policy
    is_random = np.random.uniform(0, 1) < 0.1
    if current_policy == -1 or is_random or pre_training:
        action = np.random.randint(0, env.action_space.n)
    else:
        action = reward_net.get_state_actions([s])[current_policy][0]
    return action

pre_training = True
current_episode_length = 0
max_length_before_policy_switch = 30
update_threshold_frequency = 100
(h, w, d) = env.observation_space.shape

s = env.reset()

ideal_threshold = (cv2.imread('./ideal_threshold.png')[:, :, [0]] / 255).astype(np.uint8)

reward_tracker_zero_filter = 0
while True:
    # take random action

    #a = np.random.randint(0, env.action_space.n)
    a = get_action(s)
    sp, r, t, info = env.step(a)
    state_replay_buffer.append(env.get_current_state())

    if r > 0:
        #partitioned_r = reward_net.get_partitioned_reward([sp], [r])[0]
        #print(f'{reward_buffer.length()}/{1000}')
        reward_buffer.append(s, a, r, sp, t)
        on_reward_print_func(r, sp, info, reward_net, reward_buffer)
        if args.bayes_reward_filter:
            tracker.add(sp, r)
        #LOG.add_line('max_reward_on_positive', np.max(partitioned_r))
        #image = np.concatenate([sp[:,:,0:3], sp[:,:,3:6], sp[:,:,6:9]], axis=1)
        #cv2.imwrite(f'pos_reward_{i}.png', cv2.resize(image, (400*3, 400), interpolation=cv2.INTER_NEAREST))
        #print(r, partitioned_r)
    else:
        if args.bayes_reward_filter and reward_tracker_zero_filter % 100 == 0:
            tracker.add(sp, r)

    reward_tracker_zero_filter += 1


    episode_reward += r
    #env.render()
    buffer.append(s, a, r, sp, t)
    if current_episode_length >= max_length_before_policy_switch:
        current_episode_length = 0
        current_policy = choice(policy_indices)
    if t:
        s = env.reset()
        current_policy = choice(policy_indices)
        current_episode_length = 0
        print(f'Episode Reward: {episode_reward}')
        #print(f'Epsilon {epsilon}')
        episode_reward = 0
    else:
        s = sp

    #epsilon = max(min_epsilon, epsilon - epsilon_delta)

    if args.bayes_reward_filter and (reward_buffer.length() >= min_reward_experiences) and (i % update_threshold_frequency == 0):
        threshold = np.max(tracker.compute_threshold_image(0.09), axis=2, keepdims=True)
        reward_net.update_threshold_image(threshold)

    if (buffer.length() >= batch_size) and (reward_buffer.length() >= min_reward_experiences) and (current_reward_training_step >= num_reward_steps):
        pre_training = False
        #s_sample, a_sample, r_sample, sp_sample, t_sample = buffer.sample(batch_size)
        for j in range(1):
            q_losses = reward_net.train_Q_networks()
        if i % 1 == 0:
            for j in range(1):
                reward_loss, max_value_constraint, value_constraint, J_indep, J_nontrivial = reward_net.train_R_function(dummy_env_cluster)
                LOG.add_line('reward_loss', reward_loss)
                LOG.add_line('max_value_constraint', max_value_constraint)
                LOG.add_line('value_constraint', value_constraint)
                LOG.add_line('J_indep', J_indep)
                LOG.add_line('J_nontrivial', J_nontrivial)

        #if args.separate_reward_repr:
        #    pred_reward_loss = reward_net.train_predicted_reward()

        # tensorboard logging.
        for j in range(num_partitions):
            LOG.add_line(f'q_loss{j}', q_losses[j])


        log_string = f'({i}) ' + \
                     ''.join([f'Q_{j}_loss: {q_losses[j]}\t' for j in range(num_partitions)]) + \
                     f'Reward Loss: {reward_loss}' + \
                     f'(MaxValConst: {max_value_constraint}, ValConst: {value_constraint})'
        print(log_string)

        if i % 100 == 0:
            visualization_func(reward_net, dummy_env, f'./runs/{args.name}/images/policy_vis_{i}.png')

        if i % save_freq == 0:
            reward_net.save(save_path, 'reward_net.ckpt')

        i += 1




    # train the reward initially
    if (buffer.length() >= batch_size) and (reward_buffer.length() >= min_reward_experiences) and current_reward_training_step < num_reward_steps:
        reward_loss = reward_net.train_predicted_reward()
        print(f'{current_reward_training_step}/{num_reward_steps} Loss : {reward_loss}')
        current_reward_training_step += 1







    current_episode_length += 1



