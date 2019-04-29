from reward_network2 import ReparameterizedRewardNetwork
from envs.multitask_corners_world import CornersTaskWorld
import numpy as np
from typing import List
from gym import Env
import os

class StateRepresentationWrapper(object):

    def __init__(self, env: Env, num_rewards: int, paths: List[str]):
        self.networks = []
        self.num_rewards = num_rewards
        self.env = env
        for path in paths:
            net = ReparameterizedRewardNetwork(env, num_rewards, 0.001, None, env.action_space.n, 'reward_net', visual=True,
                                         gpu_num=-1)
            net.restore(path, 'reward_net.ckpt')
            self.networks.append(net)

    def get_state_repr(self, s):
        all_qs = []
        for net in self.networks:
            qs = net.get_Qs(s) # [num_rewards, bs, num_actions]
            qs = np.transpose(qs, [1,2,0]) # [bs, num_rewards, num_actions]
            qs = np.reshape(qs, [-1, self.num_rewards * self.env.action_space.n]) # [bs, num_rewards * num_actions]
            all_qs.append(qs)
        return np.concatenate(all_qs, axis=1) # [bs, num_rewards * num_actions * num_tasks]



if __name__ == '__main__':
    env = CornersTaskWorld(visual=True, task=(1,1,1,1))

    path = '/Users/chris/projects/q_learning/reparam_runs/'
    names = ['top_5', 'bottom_5']#, 'left_5', 'right_5']
    paths = [os.path.join(path, name, 'weights') for name in names]
    repr_wrapper = StateRepresentationWrapper(env, 2, paths)
    s = env.reset()
    print(repr_wrapper.get_state_repr([s]))



