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

    def get_action_no_stoch(self, s):
        raise NotImplemented

class RD_Agent(Agent):

    def __init__(self, q_net : QNetworkTrainingWrapper):
        self.q_net = q_net

    def get_action(self, s):
        action = self.q_net.get_action([s])[0]
        action_prob = np.zeros(shape=[self.q_net.env.action_space.n])
        action_prob[action] = 1.0
        return action, action_prob


class ICF_Agent(Agent):

    def __init__(self, tf_icf_agent, policy_index, num_env_actions):
        self.tf_icf_agent = tf_icf_agent
        self.policy_index = policy_index
        self.num_env_actions = num_env_actions

    def get_action(self, s):
        action_probs = self.tf_icf_agent.get_probs([s])[0]  # [num_factors, num_actions]
        policy = action_probs[self.policy_index]
        action = np.random.choice(list(range(self.num_env_actions)), p=policy)
        return action, policy



def compute_value_and_action_stats(env, agent, rollout=500, repeats=10):
    gamma = 0.99
    all_V = []
    probabilities = []
    for _ in range(repeats):
        s = env.reset()
        V = 0
        for t in tqdm.tqdm(range(rollout)):
            a, probs = agent.get_action(s)
            probabilities.append(probs)
            s, r, _, _ = env.step(a)
            V += gamma**t * r
        all_V.append(V)
    average_variance = np.mean(np.std(probabilities, axis=0),axis=0)
    return np.mean(all_V), average_variance



mapping = {
    'seaquest': SeaquestWrapper,
    'assault': AssaultWrapper,
    'pacman': PacmanWrapper
}

def compute_all_values_rd(run_dir, name):
    game, num_rewards = re.match(r'^(.+?)\_(\d+)reward.+?$', name).groups()
    num_rewards = int(num_rewards)
    try:
        env = mapping[game]()
    except KeyError:
        return
    reward_net = RewardPartitionNetwork(env, None, None, num_rewards, env.observation_space.shape[0],
                                        env.action_space.n, 'reward_net', traj_len=10,  gpu_num=-1,
                                        use_gpu=False, num_visual_channels=9, visual=True)
    reward_net.restore(os.path.join(run_dir, name, 'best_weights'), 'reward_net.ckpt')
    for i in range(num_rewards):
        agent = RD_Agent(reward_net.Q_networks[i])
        value, avg_variance = compute_value_and_action_stats(env, agent)
        with open('rd_values_stats.txt', 'a') as f:
            f.write(f'rd,{name},{i},{value},{avg_variance}\n')


def compute_all_values_icf(run_dir, name):
    game, num_rewards = re.match(r'^(.+?)\_(\d+)reward.+?$', name).groups()
    num_rewards = int(num_rewards)
    env = mapping[game]()
    icf = ICF_Policy(2*num_rewards, env.action_space.n, 'tf_icf')
    icf.restore(os.path.join(run_dir, name, 'converted_weights.ckpt'))
    for i in range(2*num_rewards):
        agent = ICF_Agent(icf, i, env.action_space.n)
        value, avg_variance = compute_value_and_action_stats(env, agent)
        with open('icf_values_stats.txt', 'a') as f:
            f.write(f'icf,{name},{i},{value},{avg_variance}\n')

def make_command(run_dir, mode):
    icf_policies = ['assault_2reward_3', 'assault_3reward_2', 'assault_5reward_2', 'assault_8reward_1',
                    'seaquest_2reward_2', 'seaquest_3reward_3', 'seaquest_5reward_1', 'seaquest_8reward_3',
                    'pacman_2reward_3', 'pacman_3reward_3', 'pacman_5reward_3', 'pacman_8reward_2']

    rd_policies = ['assault_2reward_10mult_1', 'assault_3reward_10mult_2', 'assault_5reward_10mult_3',
                   'assault_8reward_10mult_3',
                   'seaquest_2reward_10mult_3', 'seaquest_3reward_10mult_4', 'seaquest_5reward_10mult_4',
                   'seaquest_8reward_10mult_2',
                   'pacman_2reward_10mult_2', 'pacman_3reward_10mult_3', 'pacman_5reward_10mult_1',
                   'pacman_8reward_10mult_2',
                   ]
    selected_policies = icf_policies if mode == 'icf' else rd_policies
    files = [x for x in os.listdir(run_dir) if x in selected_policies]

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


