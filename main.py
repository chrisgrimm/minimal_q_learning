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

import numpy as np

from envs.block_world.block_pushing_domain import BlockPushingDomain
from replay_buffer import ReplayBuffer
from reward_network import RewardPartitionNetwork
from utils import LOG, build_directory_structure

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
args = parser.parse_args()

mode_options = ['SOKOBAN', 'ASSAULT']
mode = 'ASSAULT'
visual = True

assert mode in mode_options



observation_mode = 'image' if visual else 'vector'


if mode == 'ASSAULT':
    num_partitions = 3
    num_visual_channels = 9
    visualization_func = produce_assault_ship_histogram_visualization
    # visual mode must be on for Assault domain.
    assert visual
    env = SimpleAssault()
    dummy_env = SimpleAssault()
elif mode == 'SOKOBAN':
    num_partitions = 2
    num_visual_channels = 3
    visualization_func = produce_two_goal_visualization
    env = BlockPushingDomain(observation_mode=observation_mode)
    dummy_env = BlockPushingDomain(observation_mode=observation_mode)
else:
    raise Exception(f'mode must be in {mode_options}.')

build_directory_structure('.', {'runs': {args.name: {}}})
LOG.setup(f'./runs/{args.name}')

#agent = QLearnerAgent(env.observation_space.shape[0], env.action_space.n)
buffer = ReplayBuffer(100000)

reward_buffer = ReplayBuffer(100000)
reward_net = RewardPartitionNetwork(buffer, reward_buffer, num_partitions, env.observation_space.shape[0], env.action_space.n, 'reward_net', num_visual_channels=num_visual_channels, visual=visual)

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

while True:
    # take random action

    #a = np.random.randint(0, env.action_space.n)
    a = get_action(s)
    sp, r, t, _ = env.step(a)
    if r > 0:
        partitioned_r = reward_net.get_partitioned_reward([s])[0]
        reward_buffer.append(s, a, r, sp, t)
        print(r, partitioned_r)
    episode_reward += r
    #env.render()
    buffer.append(s, a, r, sp, t)
    if t:
        s = env.reset()
        current_policy = choice(policy_indices)
        print(f'Episode Reward: {episode_reward}')
        print(f'Epsilon {epsilon}')
        episode_reward = 0
    else:
        s = sp

    #epsilon = max(min_epsilon, epsilon - epsilon_delta)

    if buffer.length() >= batch_size and reward_buffer.length() >= batch_size:
        pre_training = False
        #s_sample, a_sample, r_sample, sp_sample, t_sample = buffer.sample(batch_size)
        for j in range(5):
            q_losses = reward_net.train_Q_networks()
        for j in range(3):
            reward_loss = reward_net.train_R_function(dummy_env)
        # tensorboard logging.
        for j in range(num_partitions):
            LOG.add_line(f'q_loss{j}', q_losses[j])
        LOG.add_line('reward_loss', reward_loss)

        log_string = f'({i}) ' + \
                     ''.join([f'Q_{j}_loss: {q_losses[j]}\t' for j in range(num_partitions)]) + \
                     f'Reward Loss: {reward_loss}'
        print(log_string)

        if i % 100 == 0:
            visualization_func(reward_net, dummy_env)



    i += 1



