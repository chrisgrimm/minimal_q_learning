import numpy as np
from envs.atari.atari_wrapper import PacmanWrapper, QBertWrapper, AssaultWrapper, AlienWrapper, SeaquestWrapper, BreakoutWrapper
from q_learner_agent import QLearnerAgent
from envs.metacontroller_actor import MetaEnvironment
from envs.atari.simple_assault import SimpleAssault
from visualization import produce_two_goal_visualization, produce_assault_ship_histogram_visualization, produce_assault_reward_visualization, produce_reward_statistics, visualize_all_representations_all_reward_images, record_value_matrix, approximate_disentanglement_terms
from utils import LOG, build_directory_structure, add_implicit_name_arg
from reward_prob_tracker import RewardProbTracker
import argparse
from random import choice
import cv2
import tqdm
import os, re
import tensorflow as tf
from replay_buffer import StateReplayBuffer

from envs.atari.threaded_environment import ThreadedEnvironment
from envs.block_world.block_pushing_domain import BlockPushingDomain
from replay_buffer import ReplayBuffer
#from reward_network import RewardPartitionNetwork
from reward_network2 import ReparameterizedRewardNetwork
from utils import LOG, build_directory_structure

parser = argparse.ArgumentParser()
add_implicit_name_arg(parser)

parser.add_argument('--reuse-visual', action='store_true')
parser.add_argument('--traj-len', type=int, default=10)
parser.add_argument('--max-value-mult', type=float, default=10.0)
parser.add_argument('--dynamic-weighting-disentangle', action='store_true')
parser.add_argument('--mode', type=str, required=True, choices=['SOKOBAN', 'SOKOBAN_REWARD_ALWAYS_ONE', 'SOKOBAN_OBSTACLE', 'SOKOBAN_FOUR_ROOM', 'ASSAULT', 'PACMAN', 'QBERT', 'ALIEN', 'BREAKOUT', 'SEAQUEST'])
parser.add_argument('--visual', action='store_true')
parser.add_argument('--learning-rate', type=float, default=0.00005)
parser.add_argument('--gpu-num', type=int, required=True)
parser.add_argument('--separate-reward-repr', action='store_true')
parser.add_argument('--bayes-reward-filter', action='store_true')
parser.add_argument('--use-ideal-filter', action='store_true')
parser.add_argument('--num-partitions', type=int, required=True)
parser.add_argument('--use-meta-controller', action='store_true')
parser.add_argument('--clip-gradient', type=float, default=-1)
parser.add_argument('--restore-dead-run', type=str, default=None)
parser.add_argument('--softmin-temp', type=float, default=1.0)
parser.add_argument('--stop-softmin-gradient', action='store_true')
parser.add_argument('--run-dir', type=str, default='new_runs')
parser.add_argument('--regularize', action='store_true')
parser.add_argument('--regularization-weight', type=float, default=1.0)
parser.add_argument('--hybrid-reward', action='store_true')
parser.add_argument('--hybrid-reward-mode', type=str, default='sum', choices=['sum', 'max'])



args = parser.parse_args()

use_gpu = args.gpu_num >= 0
mode = args.mode
visual = args.visual

observation_mode = 'image' if visual else 'vector'

def default_visualizations(network, env, value_matrix, name):
    [path, name] = os.path.split(name)
    [name, name_extension] = name.split('.')
    statistics_full_name = os.path.join(path, name + '_statistics.txt')
    behavior_name = name + '_behavior_file.pickle'
    behavior_full_name = os.path.join(path, behavior_name)
    value_matrix_full_name = os.path.join(path, name+'_value_matrix.pickle')

    #produce_reward_statistics(network, env, statistics_full_name, behavior_full_name)
    #record_value_matrix(value_matrix, value_matrix_full_name)
    file_number = re.match(r'^policy\_vis\_(\d+)\_behavior_file.pickle$', behavior_name).groups()[0]
    #produce_all_videos(path, file_number)




def default_on_reward_print_func(r, sp, info, network, reward_buffer):
    partitioned_r = network.get_partitioned_reward([sp], [r])[0]
    print(reward_buffer.length(), partitioned_r)

ATARI_GAMES = ['ASSAULT', 'PACMAN', 'SEAQUEST', 'BREAKOUT', 'QBERT', 'ALIEN']
num_frames = 4 if mode in ATARI_GAMES else 1
num_color_channels = 1 if mode in ATARI_GAMES else 3

display_freq = 10000

def setup_atari(name):
    name_class_mapping = {
        'ASSAULT': AssaultWrapper,
        'PACMAN': PacmanWrapper,
        'SEAQUEST': SeaquestWrapper,
        'QBERT': QBertWrapper,
        'BREAKOUT': BreakoutWrapper,
        'ALIEN': AlienWrapper
    }
    num_partitions = args.num_partitions
    num_visual_channels = num_frames * num_color_channels
    on_reward_print_func = default_on_reward_print_func
    visualization_func = default_visualizations
    assert visual
    env = name_class_mapping[name]()
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: name_class_mapping[name](),
                                            name_class_mapping[name])
    dummy_env_cluster('reset', args=[])
    dummy_env = name_class_mapping[name]()
    dummy_env.reset()
    return {'num_partitions': num_partitions,
            'num_visual_channels': num_visual_channels,
            'on_reward_print_func': on_reward_print_func,
            'visualization_func': visualization_func,
            'env': env,
            'dummy_env_cluster': dummy_env_cluster,
            'dummy_env': dummy_env
            }



if mode in ATARI_GAMES:
    settings = setup_atari(mode)
    num_partitions = settings['num_partitions']
    num_visual_channels = settings['num_visual_channels']
    on_reward_print_func = settings['on_reward_print_func']
    visualization_func = settings['visualization_func']
    env = settings['env']
    dummy_env_cluster = settings['dummy_env_cluster']
    dummy_env = settings['dummy_env']

elif mode == 'SOKOBAN':
    num_partitions = args.num_partitions
    num_visual_channels = num_frames * num_color_channels
    visualization_func = produce_two_goal_visualization
    on_reward_print_func = default_on_reward_print_func

    config = 'standard'
    env = BlockPushingDomain(observation_mode=observation_mode, configuration=config)
    # dummy_env_cluster = ThreadedEnvironment(32,
    #                                         lambda i: BlockPushingDomain(observation_mode=observation_mode,
    #                                                                      configuration=config),
    #                                         BlockPushingDomain)
    #dummy_env_cluster('reset', args=[])
    dummy_env = BlockPushingDomain(observation_mode=observation_mode, configuration=config)
    dummy_env.reset()
    dummy_env_cluster = None
    ##dummy_env_cluster = dummy_env = None
elif mode == 'SOKOBAN_REWARD_ALWAYS_ONE':
    num_partitions = args.num_partitions
    num_visual_channels = num_frames * num_color_channels
    visualization_func = produce_two_goal_visualization
    on_reward_print_func = default_on_reward_print_func

    config = 'standard_all_reward'
    env = BlockPushingDomain(observation_mode=observation_mode, configuration=config)
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: BlockPushingDomain(observation_mode=observation_mode,
                                                                         configuration=config),
                                            BlockPushingDomain)
    dummy_env_cluster('reset', args=[])
    dummy_env = BlockPushingDomain(observation_mode=observation_mode, configuration=config)
    dummy_env.reset()

elif mode == 'SOKOBAN_OBSTACLE':
    num_partitions = args.num_partitions

    num_visual_channels = num_frames * num_color_channels
    visualization_func = produce_two_goal_visualization
    on_reward_print_func = default_on_reward_print_func

    config = 'obstacle'
    env = BlockPushingDomain(observation_mode=observation_mode, configuration=config)
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: BlockPushingDomain(observation_mode=observation_mode,
                                                                         configuration=config),
                                            BlockPushingDomain)
    dummy_env_cluster('reset', args=[])
    dummy_env = BlockPushingDomain(observation_mode=observation_mode, configuration=config)
    dummy_env.reset()
elif mode == 'SOKOBAN_FOUR_ROOM':
    num_partitions = args.num_partitions

    num_visual_channels = num_frames * num_color_channels
    visualization_func = produce_two_goal_visualization
    on_reward_print_func = default_on_reward_print_func

    config = 'four_room'
    display_freq = 1000
    env = BlockPushingDomain(observation_mode=observation_mode, configuration=config)
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: BlockPushingDomain(observation_mode=observation_mode,
                                                                         configuration=config),
                                            BlockPushingDomain)
    dummy_env_cluster('reset', args=[])
    dummy_env = BlockPushingDomain(observation_mode=observation_mode, configuration=config)
    dummy_env.reset()

else:
    raise Exception(f'mode must be in {mode_options}.')

run_dir = args.run_dir

build_directory_structure('.', {run_dir: {
                                    args.name: {
                                        'images': {},
                                        'weights': {},
                                        'best_weights': {},}}})
LOG.setup(f'./{run_dir}/{args.name}')
save_path = os.path.join(run_dir, args.name, 'weights')
best_save_path = os.path.join(run_dir, args.name, 'best_weights')


#agent = QLearnerAgent(env.observation_space.shape[0], env.action_space.n)
buffer = ReplayBuffer(100000, num_frames, num_color_channels)

#state_replay_buffer = StateReplayBuffer(1000000)

# reward_net = RewardPartitionNetwork(env, buffer, state_replay_buffer, num_partitions, env.observation_space.shape[0],
#                                     env.action_space.n, 'reward_net', traj_len=args.traj_len,  gpu_num=args.gpu_num,
#                                     use_gpu=use_gpu, num_visual_channels=num_visual_channels, visual=visual,
#                                     max_value_mult=args.max_value_mult, use_dynamic_weighting_disentangle_value=args.dynamic_weighting_disentangle,
#                                     lr=args.learning_rate, reuse_visual_scoping=args.reuse_visual, separate_reward_repr=args.separate_reward_repr,
#                                     use_ideal_threshold=args.use_ideal_filter, clip_gradient=args.clip_gradient,
#                                     softmin_temperature=args.softmin_temp, stop_softmin_gradients=args.stop_softmin_gradient,
#                                     regularize=args.regularize, regularization_weight=args.regularization_weight)

reward_net = ReparameterizedRewardNetwork(env, num_partitions, args.learning_rate, buffer, env.action_space.n, 'reward_net',
                                          num_channels=num_visual_channels, gpu_num=args.gpu_num)


(height, width, depth) = env.observation_space.shape
tracker = RewardProbTracker(height, width, depth)

learning_starts = 10000
batch_size = 32
q_train_freq = 4
q_loss_log_freq = 100
episode_reward = 0
print(env.action_space)
epsilon = 1.0
min_epsilon = 0.01
num_epsilon_steps = 1000000
min_reward_experiences = 500
num_reward_steps = 30000
save_freq = 10000
evaluation_frequency = 1000
current_reward_training_step = 0 if args.separate_reward_repr else num_reward_steps
epsilon_delta = (epsilon - min_epsilon) / num_epsilon_steps
time = 0
num_steps = 10000000

# indices of current policy
policy_indices = list(range(num_partitions)) + [-1]
current_policy = choice(policy_indices)

class Actor:

    def __init__(self, network: ReparameterizedRewardNetwork):
        self.network = network
        self.eps = [1.0 for _ in range(self.network.num_rewards)]
        self.current_policy = np.random.randint(0, self.network.num_rewards)

    def act(self, s, eval=False):
        return np.random.randint(0, self.network.num_actions)
        # handle evaluation
        if eval:
            if np.random.randint(0, 1) < min_epsilon:
                a = np.random.randint(0, self.network.num_actions)
            else:
                a = self.network.get_state_actions([s])[self.current_policy][0]
            return a
        # handle non-evaluation
        if np.random.uniform(0, 1) < self.eps[self.current_policy]:
            a = np.random.randint(0, self.network.num_actions)
        else:
            a = self.network.get_state_actions([s])[self.current_policy][0]
        self.eps[self.current_policy] = max(min_epsilon, self.eps[self.current_policy] - epsilon_delta)
        return a

    def switch_policy(self, policy_number=-1):
        if policy_number >= 0:
            self.current_policy = policy_number
        else:
            self.current_policy = np.random.randint(0, self.network.num_rewards)


def get_action(s, eval=False):
    global epsilon
    global current_policy
    is_random = np.random.uniform(0, 1) < (min_epsilon if eval else epsilon)
    if is_random or True:
        action = np.random.randint(0, env.action_space.n)
    else:
        if args.hybrid_reward:
            action = reward_net.get_hybrid_actions([s], mode=args.hybrid_reward_mode)[0]
        else:
            action = reward_net.get_state_actions([s])[current_policy][0]

    return action


# this is a better performance metric. assesses the reward gained over an episode rather than for an arbitrary number of steps.
def evaluate_performance(env, actor: Actor):
    s = env.reset()
    cumulative_reward = 0
    cumulative_reward_env = 0
    internal_terminal = False
    #max_timesteps = 10000
    #for _ in range(max_timesteps):
    while True:
        a = actor.act(s, eval=True)
        #a = get_action(s, eval=True)
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






def main():
    actor = Actor(reward_net)
    global learning_starts, batch_size,q_train_freq,q_loss_log_freq,episode_reward,epsilon,min_epsilon, \
        num_epsilon_steps,min_reward_experiences,num_reward_steps,save_freq,evaluation_frequency, \
        current_reward_training_step,epsilon_delta,time,num_steps

    current_episode_length = 0
    max_length_before_policy_switch = -1
    update_threshold_frequency = 100
    (h, w, d) = env.observation_space.shape
    last_100_scores = [np.inf]
    best_score = np.inf
    s = env.reset()

    starting_time = 0
    for time in range(starting_time, num_steps):

        a = actor.act(s)
        sp, r, t, info = env.step(a)

        #state_replay_buffer.append(env.get_current_state())



        episode_reward += r
        buffer.append(s, a, r, sp, t)

        if info['internal_terminal']:
            current_episode_length = 0
            #current_policy = choice(policy_indices)
            actor.switch_policy()

        if t:
            s = env.reset()
            #current_policy = choice(policy_indices)
            current_episode_length = 0
            print(f'Episode Reward: {episode_reward}')
            #print(f'Epsilon {epsilon}')
            episode_reward = 0
        else:
            s = sp

        if time >= learning_starts:

            # if time % q_train_freq == 0:
            #
            #     q_losses = reward_net.train_Q_networks(time)
            #     # tensorboard logging.
            #     if time % q_loss_log_freq == 0:
            #         for j in range(num_partitions):
            #             LOG.add_line(f'q_loss{j}', q_losses[j])

            if time % q_train_freq == 0:
                for j in range(1):
                    sums_to_R, greater_than_0, reward_consistency, J_indep, J_nontriv, q_loss = reward_net.train_R_functions(time)
                    LOG.add_line('sums_to_R', sums_to_R)
                    LOG.add_line('greater_than_0', greater_than_0)
                    LOG.add_line('reward_consistency', reward_consistency)
                    LOG.add_line('J_indep', J_indep)
                    LOG.add_line('J_nontriv', J_nontriv)
                    LOG.add_line('J_disentangled', J_indep - J_nontriv)
                    LOG.add_line('q_loss', q_loss)
                    #LOG.add_line('time', time)
                    # # TODO actually log the value_partition
                    # if len(last_100_scores) < 100:
                    #     last_100_scores.append(J_indep - J_nontrivial)
                    # else:
                    #     last_100_scores = last_100_scores[1:] + [J_indep - J_nontrivial]
            log_string = f'({time}, eps: {epsilon})'
            # log_string = f'({time}, eps: {epsilon}) ' + \
            #              ''.join([f'Q_{j}_loss: {q_losses[j]}\t' for j in range(num_partitions)]) + \
            #              f'Reward Loss: {reward_loss}' + \
            #              f'(MaxValConst: {max_value_constraint}, ValConst: {value_constraint})'
            print(log_string)

            if time % display_freq == 0:
                print('displaying!')
                value_matrix = np.zeros([num_partitions, num_partitions], dtype=np.float32)
                visualization_func(reward_net, dummy_env, value_matrix, f'./{run_dir}/{args.name}/images/policy_vis_{time}.png')
                approx_J_nontriv, approx_J_indep, policy_value_vector = approximate_disentanglement_terms(reward_net, dummy_env)
                for reward_num in range(reward_net.num_rewards):
                    Vii = policy_value_vector[reward_num]
                    LOG.add_line(f'Value_{reward_num}', Vii)
                approx_J_disentangled = approx_J_indep - approx_J_nontriv
                LOG.add_line('approx_J_nontriv', approx_J_nontriv)
                LOG.add_line('approx_J_indep', approx_J_indep)
                LOG.add_line('approx_J_disentangled', approx_J_disentangled)


            if time % save_freq == 0:
                reward_net.save(save_path, 'reward_net.ckpt')

            if time % evaluation_frequency == 0:
                cum_reward, cum_reward_env = evaluate_performance(env, actor)
                print(f'({time}) EVAL: {cum_reward}')
                LOG.add_line('cum_reward', cum_reward)
                LOG.add_line('cum_reward_env', cum_reward_env)

            if time % 10000 == 0 and np.mean(last_100_scores) < best_score:
                best_score = np.mean(last_100_scores)
                reward_net.save(best_save_path, 'reward_net.ckpt')



            epsilon = max(min_epsilon, epsilon - epsilon_delta)

        current_episode_length += 1

if __name__ == '__main__':
    main()

