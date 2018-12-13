from q_learner_agent import QLearnerAgent
from reward_network import RewardPartitionNetwork
from typing import List
from gym.spaces import Box, Discrete
import numpy as np

class MetaEnvironment(object):

    def __init__(self,
                 env,
                 r_net : RewardPartitionNetwork,
                 q_learners: List[QLearnerAgent],
                 stop_at_reward : bool,
                 repeat : int,
                 allow_base_actions : bool = False):
        self.env = env
        self.allow_base_actions = allow_base_actions
        self.q_learners = q_learners
        self.action_space = Discrete(len(q_learners) + (env.action_space.n if allow_base_actions else 0))
        self.observation_space = env.observation_space
        self.current_obs = np.copy(self.env.get_obs())
        self.stop_at_reward = stop_at_reward
        self.repeat = repeat
        self.r_net = r_net

    def step(self, a):
        total_reward = 0
        terminal = False
        internal_terminal = False

        # when we allow base actions, the base actions have a repeat on them.
        if self.allow_base_actions and a < len(self.q_learners):
            for i in range(self.repeat):
                # if we allow base actions the first len(self.q_learners) actions are meta-actions, rest are base.
                actual_action = self.q_learners[a].get_action([self.current_obs])[0]
                sp, r, t, info = self.env.step(actual_action)
                self.current_obs = np.copy(sp)
                internal_terminal = info['internal_terminal'] or internal_terminal
                total_reward += r
                terminal = terminal or t
                if self.stop_at_reward and r == 1:
                    reward_part = self.r_net.get_reward(sp, 1)[a]
                    if reward_part > 0.99:
                        break
                if terminal:
                    break
        else:

            if self.allow_base_actions:
                actual_action = a - len(self.q_learners)
            else:
                actual_action = a
            sp, r, t, info = self.env.step(actual_action)
            self.current_obs = np.copy(sp)
            internal_terminal = info['internal_terminal'] or internal_terminal
            total_reward += r
            terminal = terminal or t
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

