from random import choice
import numpy as np
from replay_buffer import ReplayBuffer
from envs.block_world.block_pushing_domain import BlockPushingDomain
from envs.atari.simple_assault import SimpleAssault
from reward_network import RewardPartitionNetwork
from visualization import produce_two_goal_visualization, produce_assault_ship_histogram_visualization
import argparse
from utils import LOG, build_directory_structure
import argparse
from random import choice
import cv2

import numpy as np
from envs.atari.threaded_environment import ThreadedEnvironment
from envs.block_world.block_pushing_domain import BlockPushingDomain
from replay_buffer import ReplayBuffer
from reward_network import RewardPartitionNetwork
from utils import LOG, build_directory_structure

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--mode', type=str, required=True, choices=['SOKOBAN', 'ASSAULT'])
parser.add_argument('--visual', action='store_true')
parser.add_argument('--gpu-num', type=int, required=True)
args = parser.parse_args()

mode = args.mode
visual = args.visual

observation_mode = 'image' if visual else 'vector'


if mode == 'ASSAULT':
    num_partitions = 3
    num_visual_channels = 9
    visualization_func = produce_assault_ship_histogram_visualization
    # visual mode must be on for Assault domain.
    assert visual
    env = SimpleAssault(initial_states_file='stored_states_64.pickle')
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: SimpleAssault(initial_states_file='stored_states_64.pickle'),
                                            SimpleAssault)
    dummy_env = SimpleAssault(initial_states_file='stored_states_64.pickle')
elif mode == 'SOKOBAN':
    num_partitions = 3
    num_visual_channels = 3
    visualization_func = produce_two_goal_visualization
    env = BlockPushingDomain(observation_mode=observation_mode)
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: BlockPushingDomain(observation_mode=observation_mode),
                                            BlockPushingDomain)
    dummy_env = BlockPushingDomain(observation_mode=observation_mode)
else:
    raise Exception(f'mode must be in {mode_options}.')

build_directory_structure('.', {'runs': {
    args.name: {
        'images': {}}}})
LOG.setup(f'./runs/{args.name}')

#agent = QLearnerAgent(env.observation_space.shape[0], env.action_space.n)
buffer = ReplayBuffer(100000)

reward_buffer = ReplayBuffer(100000)
reward_net = RewardPartitionNetwork(buffer, reward_buffer, num_partitions, env.observation_space.shape[0], env.action_space.n, 'reward_net', gpu_num=args.gpu_num, num_visual_channels=num_visual_channels, visual=visual)

batch_size = 32
s = env.reset()
epsilon = 0.1
episode_reward = 0
print(env.action_space)
epsilon = 1.0
min_epsilon = 0.1
num_epsilon_steps = 100000
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
while True:
    # take random action

    #a = np.random.randint(0, env.action_space.n)
    a = get_action(s)
    sp, r, t, _ = env.step(a)
    if r > 0:
        partitioned_r = reward_net.get_partitioned_reward([sp])[0]
        print(f'{reward_buffer.length()}/{1000}')

        reward_buffer.append(s, a, r, sp, t)
        #LOG.add_line('max_reward_on_positive', np.max(partitioned_r))
        #image = np.concatenate([sp[:,:,0:3], sp[:,:,3:6], sp[:,:,6:9]], axis=1)
        #cv2.imwrite(f'pos_reward_{i}.png', cv2.resize(image, (400*3, 400), interpolation=cv2.INTER_NEAREST))
        print(r, partitioned_r)

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

    if buffer.length() >= batch_size and reward_buffer.length() >= 1000:
        pre_training = False
        #s_sample, a_sample, r_sample, sp_sample, t_sample = buffer.sample(batch_size)
        for j in range(100):
            q_losses = reward_net.train_Q_networks()
        for j in range(1):
            reward_loss = reward_net.train_R_function(dummy_env_cluster)
        # tensorboard logging.
        for j in range(num_partitions):
            LOG.add_line(f'q_loss{j}', q_losses[j])

        LOG.add_line('reward_loss', reward_loss)

        log_string = f'({i}) ' + \
                     ''.join([f'Q_{j}_loss: {q_losses[j]}\t' for j in range(num_partitions)]) + \
                     f'Reward Loss: {reward_loss}'
        print(log_string)

        if i % 100 == 0:
            visualization_func(reward_net, dummy_env, f'./runs/{args.name}/images/policy_vis_{i}.png')



    i += 1
    current_episode_length += 1



