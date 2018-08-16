import gym
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


build_directory_structure('.', {'runs': {args.name: {}}})
LOG.setup(f'./runs/{args.name}')

visual = True
observation_mode = 'image' if visual else 'vector'
env = BlockPushingDomain(observation_mode=observation_mode)
dummy_env = BlockPushingDomain(observation_mode=observation_mode)

#agent = QLearnerAgent(env.observation_space.shape[0], env.action_space.n)
buffer = ReplayBuffer(100000)

reward_buffer = ReplayBuffer(100000)
reward_net = RewardPartitionNetwork(buffer, reward_buffer, 2, env.observation_space.shape[0], env.action_space.n, 'reward_net', visual=visual)

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



while True:
    # take random action
    a = np.random.randint(0, env.action_space.n)
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
        LOG.add_line('q_loss0', q_losses[0])
        LOG.add_line('q_loss1', q_losses[1])
        LOG.add_line('reward_loss', reward_loss)
        print(f'({i}) Q_1_loss: {q_losses[0]}\t Q_2_loss: {q_losses[1]}\t Reward Loss: {reward_loss}')

        #state_pairs = dummy_env.get_all_agent_positions()
        produce_two_goal_visualization(reward_net, dummy_env)
        #all_states = dummy_env.get_all_states()
        #get_state_max_rewards(all_states, reward_net)
        #values = reward_net.get_state_values(all_states)
        #actions = reward_net.get_state_actions(all_states)
        #visualize_actions(actions)
        #loss = agent.train_batch(s_sample, a_sample, r_sample, sp_sample, t_sample)

    i += 1



