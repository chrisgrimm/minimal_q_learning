from q_learner_agent import QLearnerAgent
from typing import List
from gym.spaces import Box, Discrete
import numpy as np

class MetaEnvironment(object):

    def __init__(self, env, q_learners: List[QLearnerAgent]):
        self.env = env
        self.q_learners = q_learners
        self.action_space = Discrete(len(q_learners))
        self.observation_space = env.observation_space
        self.current_obs = np.copy(self.env.get_obs())

    def step(self, a):
        actual_action = self.q_learners[a].get_action([self.current_obs])[0]
        sp, r, t, info = self.env.step(actual_action)
        self.current_obs = np.copy(sp)
        assert 'a' not in info
        info['a'] = actual_action
        return sp, r, t, info

    def reset(self):
        obs = self.env.reset()
        self.current_obs = np.copy(obs)
        return obs

    def render(self):
        return self.env.render()

