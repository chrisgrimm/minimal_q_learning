import gym
import numpy as np
from q_learner_agent import QLearnerAgent
from replay_buffer import ReplayBuffer
from envs.block_pushing_domain import BlockPushingDomain
from reward_network import RewardPartitionNetwork
from visualization import visualize_actions
import argparse
from utils import LOG, build_directory_structure


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
args = parser.parse_args()


build_directory_structure('.', {'runs': {args.name: {}}})
LOG.setup(f'./runs/{args.name}')


env = BlockPushingDomain()
dummy_env = BlockPushingDomain()

#agent = QLearnerAgent(env.observation_space.shape[0], env.action_space.n)
buffer = ReplayBuffer(100000)
reward_net = RewardPartitionNetwork(buffer, 2, env.observation_space.shape[0], env.action_space.n, 'reward_net')

batch_size = 32
s = env.reset()
epsilon = 0.1
episode_reward = 0
print(env.action_space)
epsilon = 1.0
min_epsilon = 0.1
num_epsilon_steps = 100000
epsilon_delta = (epsilon - min_epsilon) / num_epsilon_steps
while True:
    # take random action
    a = np.random.randint(0, env.action_space.n)
    sp, r, t, _ = env.step(a)
    episode_reward += r
    env.render()
    buffer.append(s, a, r, sp, t)
    if t:
        s = env.reset()
        print(f'Episode Reward: {episode_reward}')
        print(f'Epsilon {epsilon}')
        episode_reward = 0
    else:
        s = sp

    #epsilon = max(min_epsilon, epsilon - epsilon_delta)

    if buffer.length() >= batch_size:
        #s_sample, a_sample, r_sample, sp_sample, t_sample = buffer.sample(batch_size)
        q_losses = reward_net.train_Q_networks()
        reward_loss = reward_net.train_R_function(dummy_env)
        LOG.add_line('q_loss0', q_losses[0])
        LOG.add_line('q_loss1', q_losses[1])
        LOG.add_line('reward_loss', reward_loss)
        #print(f'Q_1_loss: {q_losses[0]}\t Q_2_loss: {q_losses[1]}\t Reward Loss: {reward_loss}')

        all_states = dummy_env.get_all_states()
        #values = reward_net.get_state_values(all_states)
        actions = reward_net.get_state_actions(all_states)
        visualize_actions(actions)
        #loss = agent.train_batch(s_sample, a_sample, r_sample, sp_sample, t_sample)



