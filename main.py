import gym
import numpy as np
from q_learner_agent import QLearnerAgent
from replay_buffer import ReplayBuffer

env = gym.make('Pendulum-v0')
agent = QLearnerAgent(env.observation_space.shape[0], 2)
buffer = ReplayBuffer(100000)
batch_size = 32
s = env.reset()
epsilon = 0.1
episode_reward = 0
print(env.action_space)
while True:
    # take action
    if buffer.length() < batch_size or np.random.uniform(0, 1) < epsilon:
        a = np.random.randint(0, 2)#env.action_space.n)
    else:
        a = agent.get_action([s])[0]
    processed_action = [2.0] if a == 0 else [-2.0]
    sp, r, t, _ = env.step(processed_action)
    episode_reward += r
    env.render()
    buffer.append(s, a, r, sp, t)
    if t:
        s = env.reset()
        print(f'Episode Reward: {episode_reward}')
        episode_reward = 0
    else:
        s = sp

    if buffer.length() >= batch_size:
        s_sample, a_sample, r_sample, sp_sample, t_sample = buffer.sample(batch_size)
        loss = agent.train_batch(s_sample, a_sample, r_sample, sp_sample, t_sample)



