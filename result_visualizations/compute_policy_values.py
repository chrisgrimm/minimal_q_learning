import os
import numpy as np
import re
import tqdm
from envs.atari.atari_wrapper import SeaquestWrapper, AssaultWrapper, PacmanWrapper
from baselines.deepq.experiments.training_wrapper import QNetworkTrainingWrapper
from reward_network import RewardPartitionNetwork
from theano_converter import ICF_Policy
import sys


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


def compute_value(env, agent, rollout=1000, repeats=10):
    gamma = 0.99
    all_V = []
    for _ in range(repeats):
        s = env.reset()
        V = 0
        for t in tqdm.tqdm(range(rollout)):
            a = agent.get_action(s)
            s, r, _, _ = env.step(a)
            V += gamma**t * r
        all_V.append(V)
    return np.mean(all_V)

mapping = {
    'seaquest': SeaquestWrapper,
    'assault': AssaultWrapper,
    'pacman': PacmanWrapper
}

def compute_all_values_rd(run_dir, name):
    game, num_rewards = re.match(r'^(.+?)\_(\d+)reward$', name).groups()
    num_rewards = int(num_rewards)
    env = mapping[game]()
    reward_net = RewardPartitionNetwork(env, None, None, num_rewards, env.observation_space.shape[0],
                                        env.action_space.n, 'reward_net', traj_len=10,  gpu_num=-1,
                                        use_gpu=False, num_visual_channels=9, visual=True)
    reward_net.restore(os.path.join(run_dir, name), 'reward_net.ckpt')
    for i in range(num_rewards):
        agent = RD_Agent(reward_net.Q_networks[i])
        value = compute_value(env, agent)
        with open('values.txt', 'a') as f:
            f.write(f'rd,{game},{num_rewards},{value}\n')


def compute_all_values_icf(run_dir, name):
    game, num_rewards = re.match(r'^(.+?)\_(\d+)reward$', name).groups()
    num_rewards = int(num_rewards)
    env = mapping[game]()
    icf = ICF_Policy(2*num_rewards, env.action_space.n, 'tf_icf')
    icf.restore(os.path.join(run_dir, name, 'converted_weights.ckpt'))
    for i in range(num_rewards):
        agent = ICF_Agent(icf, i, env.action_space.n)
        value = compute_value(env, agent)
        with open('values.txt', 'a') as f:
            f.write(f'icf,{game},{num_rewards},{value}\n')

def make_command(run_dir, mode):
    files = [x for x in os.listdir(run_dir) if x.contains('reward')]
    path_variables = [
        '~/minimal_q_learning',
        '~/baselines',
        '~/ICF_copy/DL/ICF_simple'
    ]
    preamble = f'PYTHONPATH={":".join(path_variables)} '
    all_commands = []
    for file in files:
        command = preamble+f'python result_visualizations/compute_policy_values.py {run_dir} {file} {mode}'
        all_commands.append(command)
    print('; '.join(all_commands))

if __name__ == '__main__':
    run_dir, name, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    if name == 'make_command':
        make_command(run_dir, mode)
    else:
        if mode == 'icf':
            compute_all_values_icf(run_dir, name)
        else:
            compute_all_values_rd(run_dir, name)


