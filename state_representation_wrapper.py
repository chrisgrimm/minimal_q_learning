from reward_network2 import ReparameterizedRewardNetwork
from envs.multitask_corners_world import CornersTaskWorld
from baselines.deepq.experiments.training_wrapper import make_dqn

import numpy as np
from typing import List, Optional
from gym import Env
from gym.spaces import Box
import os
import tensorflow as tf


class StateRepresentationWrapper(Env):

    def __init__(self, env: Env, num_rewards: int, paths: List[str], mode='reward_net', gpu_num=-1,
                 idealized_task_list=None, create_env=None):
        assert mode in ['reward_net', 'dqn', 'idealized']
        if mode == 'dqn':
            assert num_rewards == 1
        if mode == 'idealized':
            assert idealized_task_list is not None
            assert create_env is not None


        self.mode = mode
        self.networks = []
        self.num_rewards = num_rewards
        self.env = env
        if mode == 'reward_net':
            for path in paths:
                net = ReparameterizedRewardNetwork(env, num_rewards, 0.001, None, env.action_space.n, 'reward_net', visual=True,
                                             gpu_num=gpu_num)
                net.restore(path, 'reward_net.ckpt')
                self.networks.append(net)
        elif mode == 'idealized':
            self.idealized_env_list = []
            for task in idealized_task_list:
                self.idealized_env_list.append(create_env(task))

        else:
            self.sessions = []
            self.graphs = []
            self.dqns = []
            for path in paths:
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                graph = tf.Graph()
                sess = tf.Session(config=config, graph=graph)
                with graph.as_default():
                    with sess.as_default():
                        dqn = make_dqn(env, 'dqn', gpu_num=gpu_num, visual=True)
                        dqn.restore(path, 'qnet.ckpt')
                self.sessions.append(sess)
                self.graphs.append(graph)
                self.dqns.append(dqn)


        self.action_space = env.action_space
        if mode == 'idealized':
            self.observation_space = Box(0, 100, shape=[self.action_space.n * len(idealized_task_list)])
        else:
            self.observation_space = Box(0, 100, shape=[num_rewards * self.action_space.n * len(paths)])




    def step(self, a):
        s, r, t, info = self.env.step(a)
        s = self.get_state_repr([s], [info])[0]
        return s, r, t, info

    def reset(self):
        return self.env.reset()




    def get_state_repr_reward_net(self, s, info):
        all_qs = []
        for net in self.networks:
            qs = net.get_Qs(s) # [num_rewards, bs, num_actions]
            qs = np.transpose(qs, [1,2,0]) # [bs, num_rewards, num_actions]
            qs = np.reshape(qs, [-1, self.num_rewards * self.env.action_space.n]) # [bs, num_rewards * num_actions]
            all_qs.append(qs)
        return np.concatenate(all_qs, axis=1) # [bs, num_rewards * num_actions * num_tasks]

    def get_state_repr_dqns(self, s, info):
        all_qs = []
        for sess, graph, dqn in zip(self.sessions, self.graphs, self.dqns):
            with graph.as_default():
                with sess.as_default():
                    q = dqn.get_Q(s, [0]*len(s)) # [bs, 1*num_actions]
                    all_qs.append(q)
        return np.concatenate(all_qs, axis=1) # [bs, num_tasks * 1 * num_actions]

    def get_state_repr_idealized(self, s, info_list):
        all_qs = [] # [bs, num_tasks, num_actions]
        for info_elem in info_list:
            all_env_qs = [] # [num_envs, num_actions]
            for env in self.idealized_env_list:
                qs = [env.get_true_q_values(info_elem['agent_position'], i, 0.99) for i in range(env.action_space.n)] # [num_actions]
                all_env_qs.append(qs)
            all_qs.append(all_env_qs)
        return np.reshape(all_qs, [len(info_list), -1])

    def get_state_repr(self, s, info):
        if self.mode == 'reward_net':
            return self.get_state_repr_reward_net(s, info)
        elif self.mode == 'idealized':
            return self.get_state_repr_idealized(s, info)
        else:
            return self.get_state_repr_dqns(s, info)



if __name__ == '__main__':
    env = CornersTaskWorld(visual=True, task=(1,1,1,1))

    path = '/Users/chris/projects/q_learning/reparam_runs/'
    #names = ['top_5', 'bottom_5']#, 'left_5', 'right_5']
    names = ['undecomp_bottom_5', 'undecomp_top_5']
    paths = [os.path.join(path, name, 'weights') for name in names]
    repr_wrapper = StateRepresentationWrapper(env, 1, paths, mode='dqn')
    s = env.reset()
    print(repr_wrapper.get_state_repr([s]))



