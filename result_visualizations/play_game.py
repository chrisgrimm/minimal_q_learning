import numpy as np
import cv2
from envs.atari.atari_wrapper import PacmanWrapper, AssaultWrapper, SeaquestWrapper, AtariWrapper
from baselines.deepq.experiments.training_wrapper import QNetworkTrainingWrapper
from theano_converter import ICF_Policy
from reward_network import RewardPartitionNetwork
import sys, re, os

class Agent(object):
    def get_action(self, s):
        raise NotImplemented

class RD_Agent(Agent):

    def __init__(self, q_net : QNetworkTrainingWrapper):
        self.q_net = q_net

    def get_action(self, s):
        return self.q_net.get_action([s])[0]

class ICF_Agent(Agent):

    def __init__(self, tf_icf_agent, policy_index, num_env_actions):
        self.tf_icf_agent = tf_icf_agent
        self.policy_index = policy_index
        self.num_env_actions = num_env_actions

    def get_action(self, s):
        action_probs = self.tf_icf_agent.get_probs([s])[0]  # [num_factors, num_actions]
        policy = action_probs[self.policy_index]
        return np.random.choice(list(range(self.num_env_actions)), p=policy)


def play_game_using_policy(agent, env : AtariWrapper):
    s = env.reset()
    while True:
        a = agent.get_action(s)
        s, r, t, _ = env.step(a)
        cv2.imshow('game', env.get_unprocessed_obs())
        cv2.waitKey(1)
        input('...')

def load_agent(mode, env, run_dir, name, policy_index):
    if mode == 'icf':
        num_rewards = 2 * int(re.match(r'^.+?(\d+)reward.+?$', name).groups()[0])
        tf_icf_policy = ICF_Policy(num_rewards, env.action_space.n, 'tf_icf')
        tf_icf_policy.restore(os.path.join(run_dir, name, 'converted_weights.ckpt'))
        agent = ICF_Agent(tf_icf_policy, policy_index, env.action_space.n)
        return agent
    elif mode == 'rd':
        num_rewards = int(re.match(r'^.+?(\d+)reward.+?$', name).groups()[0])

        weights_path = os.path.join(run_dir, name, 'best_weights')
        reward_net = RewardPartitionNetwork(env, None, None, num_rewards, env.observation_space.shape[0],
                                            env.action_space.n, 'reward_net', traj_len=10, gpu_num=-1,
                                            use_gpu=False, num_visual_channels=9, visual=True)
        reward_net.restore(weights_path, 'reward_net.ckpt')
        return RD_Agent(reward_net.Q_networks[policy_index])
    else:
        raise Exception(f'mode: {mode} must be "icf" or "rd"')

env_mapping = {
    'assault': AssaultWrapper,
    'pacman': PacmanWrapper,
    'seaquest': SeaquestWrapper
}

if __name__ == '__main__':
    mode = 'icf'
    env_name = 'assault'
    run_dir = '/Users/chris/git_downloads/implementations/DL/ICF_simple/converted_icf_data'
    name = 'assault_5reward_2'
    policy_index = 9

    env = env_mapping[env_name]()
    agent = load_agent(mode, env, run_dir, name, policy_index)
    play_game_using_policy(agent, env)

