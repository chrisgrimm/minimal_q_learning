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
                 allow_base_actions : bool = False,
                 icf_policies=None,
                 num_icf_policies=None):
        self.env = env
        self.allow_base_actions = allow_base_actions
        self.icf_policies = icf_policies
        if icf_policies is None:
            self.q_learners = q_learners
            self.action_space = Discrete(len(q_learners) + (env.action_space.n if allow_base_actions else 0))
            self.r_net = r_net
        else:
            self.num_icf_policies = num_icf_policies
            self.action_space = Discrete(num_icf_policies + (env.action_space.n if allow_base_actions else 0))
            self.q_learners = None
        self.observation_space = env.observation_space
        self.current_obs = np.copy(self.env.get_obs())
        self.stop_at_reward = stop_at_reward
        self.repeat = repeat
        # should protect us against accidentally passing in both modes.
        self.offset = self.num_icf_policies if self.icf_policies is not None else len(self.q_learners)



    # get the corresponding base action for the meta action: either using learned reward policies or ICF
    def get_base_action(self, meta_action):
        if self.icf_policies is not None:
            processed_state = np.array([self.current_obs.flatten() / 255.]).astype(np.float32)
            action_probs = self.icf_policies(processed_state)[0]  # [num_factors, num_actions]
            policy = action_probs[meta_action]
            return np.random.choice(list(range(self.env.action_space.n)), p=policy)
        else:
            return self.q_learners[meta_action].get_action([self.current_obs])[0]


    # here meta_actions are 0-len(Q_learners)
    def do_repeated_meta_action(self, meta_action):
        total_reward = 0
        terminal = False
        internal_terminal = False
        for i in range(self.repeat):
            a = self.get_base_action(meta_action)
            sp, r, t, info = self.env.step(a)
            self.current_obs = np.copy(sp)
            internal_terminal = info['internal_terminal'] or internal_terminal
            total_reward += r
            terminal = terminal or t
            if self.stop_at_reward and r == 1:
                break
            if terminal or internal_terminal:
                break
        info['internal_terminal'] = internal_terminal
        # TODO : is this going to be a problem
        return sp, total_reward, terminal, info


    def step(self, a):
        # if we are allowing base actions and we have selected a base-action, perform it
        if self.allow_base_actions and a >= self.offset:
            actual_action = a - self.offset
            sp, r, t, info = self.env.step(actual_action)
        else:
        # otherwise we have not selected a base action and need to execute the meta_action logic.
            sp, r, t, info = self.do_repeated_meta_action(a)
        return sp, r, t, info

    def reset(self):
        obs = self.env.reset()
        self.current_obs = np.copy(obs)
        return obs

    def render(self):
        return self.env.render()

