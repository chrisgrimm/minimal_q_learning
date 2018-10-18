from random import choice
import numpy as np
from replay_buffer import ReplayBuffer
from envs.block_world.block_pushing_domain import BlockPushingDomain
from envs.atari.simple_assault import SimpleAssault
from reward_network import RewardPartitionNetwork
from visualization import produce_two_goal_visualization, produce_assault_ship_histogram_visualization
import argparse
from utils import LOG, build_directory_structure
from baselines.deepq.experiments.training_wrapper import make_dqn
from envs.atari.pacman import PacmanWrapper, AssaultWrapper, QBertWrapper, SeaquestWrapper, AlienWrapper, BreakoutWrapper
from envs.metacontroller_actor import MetaEnvironment
import argparse
from random import choice
import cv2
import os

import numpy as np
from envs.atari.threaded_environment import ThreadedEnvironment
from envs.block_world.block_pushing_domain import BlockPushingDomain
from replay_buffer import ReplayBuffer
from q_learner_agent import QLearnerAgent
from reward_network import RewardPartitionNetwork
from utils import LOG, build_directory_structure

parser = argparse.ArgumentParser()
screen_name = None
if 'STY' in os.environ:
    screen_name = ''.join(os.environ['STY'].split('.')[1:])
    parser.add_argument('--name', type=str, default=screen_name)
else:
    parser.add_argument('--name', type=str, required=True)
parser.add_argument('--mode', type=str, required=True, choices=['SOKOBAN', 'ASSAULT', 'QBERT', 'PACMAN', 'SOKOBAN_META', 'BREAKOUT', 'SEAQUEST', 'ALIEN'])
parser.add_argument('--visual', action='store_true')
parser.add_argument('--gpu-num', type=int, required=True)
parser.add_argument('--meta', action='store_true')
parser.add_argument('--num-partitions', type=int, default=None)
parser.add_argument('--restore-path', type=str, default=None)
parser.add_argument('--run-dir', type=str, default='runs')

args = parser.parse_args()

mode = args.mode
visual = args.visual

observation_mode = 'image' if visual else 'vector'


if mode == 'ASSAULT':
    num_visual_channels = 9
    base_env = AssaultWrapper()
elif mode == 'PACMAN':
    num_visual_channels = 9
    base_env = PacmanWrapper()
elif mode == 'QBERT':
    num_visual_channels = 9
    base_env = QBertWrapper()
elif mode == 'BREAKOUT':
    num_visual_channels = 9
    base_env = BreakoutWrapper()
elif mode == 'ALIEN':
    num_visual_channels = 9
    base_env = AlienWrapper()
elif mode == 'SEAQUEST':
    num_visual_channels = 9
    base_env = SeaquestWrapper()
elif mode == 'SOKOBAN':
    num_visual_channels = 3
    base_env = BlockPushingDomain(observation_mode=observation_mode, configuration='standard')
else:
    raise Exception(f'mode must be in {mode_options}.')

if args.meta:
    assert args.num_partitions is not None
    assert args.restore_path is not None
    reward_net = RewardPartitionNetwork(None, None, None, args.num_partitions, base_env.observation_space.shape[0],
                                        base_env.action_space.n, 'reward_net', traj_len=10,
                                        num_visual_channels=num_visual_channels, visual=visual, gpu_num=args.gpu_num)
    reward_net.restore(args.restore_path, 'reward_net.ckpt')
    env = MetaEnvironment(base_env, reward_net.Q_networks)
else:
    env = base_env

runs_dir = args.run_dir

build_directory_structure('.', {runs_dir: {
    args.name: {
        'images': {},
        'weights': {}}}})
LOG.setup(f'./{runs_dir}/{args.name}')

save_path = os.path.join(runs_dir, args.name, 'weights')

#agent = QLearnerAgent(env.observation_space.shape[0], env.action_space.n)
buffer = ReplayBuffer(100000)

reward_buffer = ReplayBuffer(100000)
#dqn = QLearnerAgent(env.observation_space.shape[0], env.action_space.n, 'q_net', visual=visual, num_visual_channels=num_visual_channels, gpu_num=args.gpu_num)
dqn = make_dqn(env, scope='dqn', gpu_num=args.gpu_num)
batch_size = 32
s = env.reset()
episode_reward = 0
print(env.action_space)

epsilon = 1.0
q_train_freq = 4
min_epsilon = 0.01
learning_starts = 10000
num_epsilon_steps = 1000000
num_steps = 10000000
evaluation_frequency = 10000
start_time = 0

epsilon_delta = (epsilon - min_epsilon) / num_epsilon_steps

# indices of current policy

def get_action(s, eval=False):
    global epsilon
    effective_epsilon = min_epsilon if eval else epsilon
    is_random = np.random.uniform(0, 1) < effective_epsilon
    if is_random:
        action = np.random.randint(0, env.action_space.n)
    else:
        action = dqn.get_action([s])[0]
    return action

# this is a better performance metric. assesses the reward gained over an episode rather than for an arbitrary number of steps.
def evaluate_performance(env, q_network: QLearnerAgent):
    s = env.reset()
    cumulative_reward = 0
    internal_terminal = False
    while not internal_terminal:
        a = get_action(s, eval=True)
        #a = np.random.randint(0, env.action_space.n) if np.random.uniform(0,1) < 0.01 else q_network.get_action([s])[0]
        s, r, t, info = env.step(a)
        cumulative_reward += r
        internal_terminal = info['internal_terminal']

    #quick_visualize_policy(env, q_network)
    return cumulative_reward

for time in range(start_time, num_steps):

    a = get_action(s)
    sp, r, t, _ = env.step(a)

    buffer.append(s, a, r, sp, t)

    if t:
        s = env.reset()
    else:
        s = sp

    if time >= learning_starts:

        if time % q_train_freq == 0:
            s_sample, a_sample, r_sample, sp_sample, t_sample = buffer.sample(batch_size)
            weights, batch_idxes = np.ones_like(t_sample), None

            q_loss = dqn.train_batch(time, s_sample, a_sample, r_sample, sp_sample, t_sample, weights, batch_idxes)

            LOG.add_line(f'q_loss', q_loss)

        if time % evaluation_frequency == 0:
            cum_reward = evaluate_performance(env, dqn)
            print(f'({time}) EVAL: {cum_reward}')
            LOG.add_line('cum_reward', cum_reward)


        log_string = f'({time}) Q_loss: {q_loss}, ({epsilon})'
        print(log_string)
        LOG.add_line(f'time', time)
        # decrement epsilon appropriately
        epsilon = max(min_epsilon, epsilon - epsilon_delta)

