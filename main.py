import gym
from random import choice
import numpy as np
from q_learner_agent import QLearnerAgent
from replay_buffer import ReplayBuffer
from envs.block_pushing_domain import BlockPushingDomain
from reward_network import RewardPartitionNetwork
from visualization import visualize_actions, get_state_max_rewards, produce_two_goal_visualization
import argparse
from utils import LOG, build_directory_structure


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
args = parser.parse_args()
num_partitions = 2

build_directory_structure('.', {'runs': {args.name: {}}})
LOG.setup(f'./runs/{args.name}')

visual = True
observation_mode = 'image' if visual else 'vector'
env = BlockPushingDomain(observation_mode=observation_mode)
dummy_env = BlockPushingDomain(observation_mode=observation_mode)

#agent = QLearnerAgent(env.observation_space.shape[0], env.action_space.n)
buffer = ReplayBuffer(100000)

reward_buffer = ReplayBuffer(100000)
reward_net = RewardPartitionNetwork(buffer, reward_buffer, num_partitions, env.observation_space.shape[0], env.action_space.n, 'reward_net', visual=visual)

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
    global current_policy
    is_random = np.random.uniform(0, 1) < 0.1
    if current_policy == -1 or is_random:
        action = np.random.randint(0, env.action_space.n)
    else:
        action = reward_net.get_state_actions([s])[current_policy][0]
    return action


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

        produce_two_goal_visualization(reward_net, dummy_env)


    i += 1



