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
        actual_action = self.q_learners[a].get_action(obs)
        return self.env.step(actual_action)

    def reset(self):
        return self.env.reset()

