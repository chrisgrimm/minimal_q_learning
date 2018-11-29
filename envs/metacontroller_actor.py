from q_learner_agent import QLearnerAgent
from typing import List
from gym.spaces import Box, Discrete
import numpy as np

class MetaEnvironment(object):

    def __init__(self, env, q_learners: List[QLearnerAgent], repeat : int):
        self.env = env
        self.q_learners = q_learners
        self.action_space = Discrete(len(q_learners) + env.action_space.n)
        self.observation_space = env.observation_space
        self.current_obs = np.copy(self.env.get_obs())
        self.repeat = repeat

    def step(self, a):
        total_reward = 0
        terminal = False
        internal_terminal = False
        for i in range(self.repeat):
            if a < len(self.q_learners):
                actual_action = self.q_learners[a].get_action([self.current_obs])[0]
            else:
                actual_action = a - len(self.q_learners)
            sp, r, t, info = self.env.step(actual_action)
            self.current_obs = np.copy(sp)
            internal_terminal = info['internal_terminal'] or internal_terminal
            total_reward += r
            terminal = terminal or t
            if terminal:
                break
        assert 'a' not in info
        info['a'] = actual_action
        info['internal_terminal'] = internal_terminal
        return sp, total_reward, terminal, info

    def reset(self):
        obs = self.env.reset()
        self.current_obs = np.copy(obs)
        return obs

    def render(self):
        return self.env.render()

