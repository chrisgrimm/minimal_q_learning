from utils import add_implicit_name_arg
from envs.atari.atari_wrapper import PacmanWrapper, AssaultWrapper, QBertWrapper, SeaquestWrapper, AlienWrapper, BreakoutWrapper
from envs.metacontroller_actor import MetaEnvironment
import argparse
import os
import tensorflow as tf
import dill

import numpy as np
from envs.atari.threaded_environment import ThreadedEnvironment
from envs.block_world.block_pushing_domain import BlockPushingDomain
from replay_buffer import ReplayBuffer
from q_learner_agent import QLearnerAgent
from reward_network import RewardPartitionNetwork
from reward_network2 import ReparameterizedRewardNetwork
from utils import LOG, build_directory_structure
from theano_converter import ICF_Policy

parser = argparse.ArgumentParser()
add_implicit_name_arg(parser)

parser.add_argument('--mode', type=str, required=True, choices=['SOKOBAN', 'ASSAULT', 'QBERT', 'PACMAN', 'SOKOBAN_META', 'BREAKOUT', 'SEAQUEST', 'ALIEN', 'SOKOBAN_NO_TOP'])
parser.add_argument('--visual', action='store_true')
parser.add_argument('--gpu-num', type=int, required=True)
parser.add_argument('--meta', action='store_true')
parser.add_argument('--meta-repeat', type=int, default=1)
parser.add_argument('--num-partitions', type=int, default=None)
parser.add_argument('--restore-path', type=str, default=None)
parser.add_argument('--restricted-reward', action='store_true')
parser.add_argument('--allow-base-actions', action='store_true')
parser.add_argument('--stop-at-reward', action='store_true')
parser.add_argument('--run-dir', type=str, default='runs')
parser.add_argument('--augment-trajectories', action='store_true')
parser.add_argument('--prob-augment', type=float, default=0.1)
parser.add_argument('--use-icf-policy', action='store_true')
parser.add_argument('--icf-policy-path', type=str, default=None)
parser.add_argument('--hierarchical-eps-greedy', action='store_true')
parser.add_argument('--hierarchical-eps-greedy-meta-ratio', type=float, default=0.5)
args = parser.parse_args()

if args.hierarchical_eps_greedy:
    # must use meta controller with base actions.
    assert args.meta and args.allow_base_actions

mode = args.mode
visual = args.visual

observation_mode = 'image' if visual else 'vector'

config = tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': args.gpu_num})
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
with sess.as_default():
    from baselines.deepq.experiments.training_wrapper import make_dqn


ATARI_GAMES = ['ASSAULT', 'PACMAN', 'SEAQUEST', 'BREAKOUT', 'QBERT', 'ALIEN']
num_frames = 4 if mode in ATARI_GAMES else 1
num_color_channels = 1 if mode in ATARI_GAMES else 3
num_visual_channels = num_frames * num_color_channels

name_class_mapping = {
        'ASSAULT': AssaultWrapper,
        'PACMAN': PacmanWrapper,
        'SEAQUEST': SeaquestWrapper,
        'QBERT': QBertWrapper,
        'BREAKOUT': BreakoutWrapper,
        'ALIEN': AlienWrapper
    }
if mode in ATARI_GAMES:
    base_env = name_class_mapping[mode](remove_reward_mode=args.restricted_reward)
elif mode == 'SOKOBAN':
    base_env = BlockPushingDomain(observation_mode=observation_mode, configuration='standard')
elif mode == 'SOKOBAN_NO_TOP':
    base_env = BlockPushingDomain(observation_mode=observation_mode, configuration='standard', only_bottom_half=True)
else:
    raise Exception(f'mode must be in {mode_options}.')

def load_icf_policies(icf_policy_path):
    with open(os.path.join(icf_policy_path, 'policy.pickle'), 'rb') as f:
        policy = dill.load(f)
    return policy

if args.meta:

    if not args.use_icf_policy:
        assert args.num_partitions is not None
        assert args.restore_path is not None
        reward_net = ReparameterizedRewardNetwork(base_env, args.num_partitions, 0.0001, None, base_env.action_space.n, 'reward_net', visual=visual,
                                                  num_channels=num_visual_channels, gpu_num=args.gpu_num, reuse=None)
        #reward_net = RewardPartitionNetwork(base_env, None, None, args.num_partitions, base_env.observation_space.shape[0],
        #                                    base_env.action_space.n, 'reward_net', traj_len=10, use_gpu=True,
        #                                    num_visual_channels=num_visual_channels, visual=visual, gpu_num=args.gpu_num)
        reward_net.restore(args.restore_path, 'reward_net.ckpt')
        #Q_networks = reward_net.Q_networks
        icf_policies = None
        tf_icf_agent = None
    else:
        assert args.num_partitions is not None
        assert args.icf_policy_path is not None
        #icf_policies = load_icf_policies(args.icf_policy_path)
        tf_icf_agent = ICF_Policy(2*args.num_partitions, base_env.action_space.n, 'tf_icf')
        tf_icf_agent.restore(os.path.join(args.icf_policy_path, 'converted_weights.ckpt'))
        reward_net = None
        Q_networks = None

    env = MetaEnvironment(base_env, reward_net, None, args.stop_at_reward, args.meta_repeat, allow_base_actions=args.allow_base_actions, tf_icf_agent=tf_icf_agent, num_icf_policies=2*args.num_partitions)
elif args.augment_trajectories:

    if not args.use_icf_policy:
        assert args.num_partitions is not None
        assert args.restore_path is not None
        reward_net = RewardPartitionNetwork(base_env, None, None, args.num_partitions, base_env.observation_space.shape[0],
                                            base_env.action_space.n, 'reward_net', traj_len=10,
                                            num_visual_channels=num_visual_channels, visual=visual, gpu_num=args.gpu_num)
        reward_net.restore(args.restore_path, 'reward_net.ckpt')
    else:
        assert args.num_partitions is not None
        assert args.icf_policy_path is not None
        icf_policies = load_icf_policies(args.icf_policy_path)

    env = base_env
else:
    env = base_env

runs_dir = args.run_dir
q_loss_log_freq = 100

build_directory_structure('.', {runs_dir: {
    args.name: {
        'images': {},
        'weights': {}}}})
LOG.setup(f'./{runs_dir}/{args.name}')

save_path = os.path.join(runs_dir, args.name, 'weights')

#agent = QLearnerAgent(env.observation_space.shape[0], env.action_space.n)
buffer_size = 1000000
buffer = ReplayBuffer(buffer_size, num_frames, num_color_channels)

#dqn = QLearnerAgent(env.observation_space.shape[0], env.action_space.n, 'q_net', visual=visual, num_visual_channels=num_visual_channels, gpu_num=args.gpu_num)

with sess.as_default():
    dqn = make_dqn(env, scope='dqn', gpu_num=args.gpu_num)

#sess.run(tf.global_variables_initializer())
#sess.run(tf.local_variables_initializer())


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
evaluation_frequency = 1000
start_time = 0
augment_frequency = 10000
augment_steps = 100

prob_augment = args.prob_augment
should_augment = False

try:
    policy_num_offset = env.offset
except AttributeError:
    try:
        policy_num_offset = len(reward_net.Q_networks)
    except:
        policy_num_offset=1

augment_policy_num = np.random.randint(0, policy_num_offset)

epsilon_delta = (epsilon - min_epsilon) / num_epsilon_steps

# indices of current policy

def get_action(s, eval=False, augmented=False, augment_policy=None):
    global epsilon
    if augmented:
        return reward_net.Q_networks[augment_policy].get_action([s])[0]
    effective_epsilon = min_epsilon if eval else epsilon
    is_random = np.random.uniform(0, 1) < effective_epsilon
    if is_random:
        if args.hierarchical_eps_greedy:
            num_meta_actions = env.offset
            ratio = args.hierarchical_eps_greedy_meta_ratio
            num_base_actions = env.env.action_space.n
            probs = ([ratio * 1/num_meta_actions] * num_meta_actions) + ([(1 - ratio) * 1/num_base_actions] * num_base_actions)
            action = np.random.choice(list(range(env.action_space.n)), p=probs)
        else:
            action = np.random.randint(0, env.action_space.n)
    else:
        action = dqn.get_action([s], [0])[0]

    return action



# this is a better performance metric. assesses the reward gained over an episode rather than for an arbitrary number of steps.
def evaluate_performance(env, q_network: QLearnerAgent):
    s = env.reset()
    cumulative_reward = 0
    cumulative_reward_env = 0
    internal_terminal = False
    #max_timesteps = 10000
    #for _ in range(max_timesteps):
    while True:
        a = get_action(s, eval=True)
        #a = np.random.randint(0, env.action_space.n) if np.random.uniform(0,1) < 0.01 else q_network.get_action([s])[0]
        s, r, t, info = env.step(a)
        r_env = info['r_env']
        cumulative_reward_env += r_env
        cumulative_reward += r
        internal_terminal = info['internal_terminal']
        if internal_terminal:
            break

    #quick_visualize_policy(env, q_network)
    return cumulative_reward, cumulative_reward_env

for time in range(start_time, num_steps):

    a = get_action(s, augmented=should_augment, augment_policy=augment_policy_num)
    sp, r, t, info = env.step(a)

    buffer.append(s, a, r, sp, t)

    if t:
        s = env.reset()
    else:
        s = sp

    if args.augment_trajectories and info['internal_terminal']:
        should_augment = np.random.uniform(0, 1) < prob_augment
        augment_policy_num = np.random.randint(0, policy_num_offset)



    if time >= learning_starts:

        if time % q_train_freq == 0:
            s_sample, a_sample, r_sample, sp_sample, t_sample = buffer.sample(batch_size)

            weights, batch_idxes = np.ones_like(t_sample), None

            q_loss = dqn.train_batch(time, s_sample, a_sample, r_sample, sp_sample, t_sample, weights, batch_idxes, np.zeros_like(t_sample, dtype=np.int32))
            if time % q_loss_log_freq == 0:
                LOG.add_line(f'q_loss', q_loss)

        if time % evaluation_frequency == 0:
            cum_reward, cum_reward_env = evaluate_performance(env, dqn)
            print(f'({time}) EVAL: {cum_reward}')
            LOG.add_line('cum_reward', cum_reward)
            LOG.add_line('cum_reward_env', cum_reward_env)


        log_string = f'({time}) Q_loss: {q_loss}, ({epsilon})'
        print(log_string)
        #LOG.add_line(f'time', time)
        # decrement epsilon appropriately
        epsilon = max(min_epsilon, epsilon - epsilon_delta)
