from q_learner_agent import QLearnerAgent
from typing import List
from gym.spaces import Box, Discrete

class MetaEnvironment(object):

    def __init__(self, env, q_learners: List[QLearnerAgent]):
        self.env = env
        self.q_learners = q_learners
        self.action_space = Discrete(len(q_learners))
        self.observation_space = env.observation_space

    def step(self, a):
        obs = self.env.get_obs()
        actual_action = self.q_learners[a].get_action([obs])[0]
        sp, r, t, info = self.env.step(actual_action)
        assert 'a' not in info
        info['a'] = actual_action
        return sp, r, t, info

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

