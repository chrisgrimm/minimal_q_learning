from random import choice
import numpy as np
from replay_buffer import ReplayBuffer
from envs.block_world.block_pushing_domain import BlockPushingDomain
from envs.atari.pacman import PacmanWrapper, QBertWrapper, AssaultWrapper, AlienWrapper, SeaquestWrapper, BreakoutWrapper
from q_learner_agent import QLearnerAgent
from envs.metacontroller_actor import MetaEnvironment
from envs.atari.simple_assault import SimpleAssault
from reward_network import RewardPartitionNetwork
from visualization import produce_two_goal_visualization, produce_assault_ship_histogram_visualization, produce_assault_reward_visualization, produce_reward_statistics, visualize_all_representations_all_reward_images
import argparse
from utils import LOG, build_directory_structure
from reward_prob_tracker import RewardProbTracker
import argparse
from random import choice
import cv2
import os
from replay_buffer import StateReplayBuffer

from envs.atari.threaded_environment import ThreadedEnvironment
from envs.block_world.block_pushing_domain import BlockPushingDomain
from replay_buffer import ReplayBuffer
from reward_network import RewardPartitionNetwork
from utils import LOG, build_directory_structure

parser = argparse.ArgumentParser()
screen_name = None
if 'STY' in os.environ:
    screen_name = ''.join(os.environ['STY'].split('.')[1:])
    parser.add_argument('--name', type=str, default=screen_name)
else:
    parser.add_argument('--name', type=str, required=True)

parser.add_argument('--reuse-visual', action='store_true')
parser.add_argument('--traj-len', type=int, default=10)
parser.add_argument('--max-value-mult', type=float, default=10.0)
parser.add_argument('--dynamic-weighting-disentangle', action='store_true')
parser.add_argument('--mode', type=str, required=True, choices=['SOKOBAN', 'ASSAULT', 'PACMAN', 'QBERT', 'ALIEN', 'BREAKOUT', 'SEAQUEST'])
parser.add_argument('--visual', action='store_true')
parser.add_argument('--learning-rate', type=float, default=0.00005)
parser.add_argument('--gpu-num', type=int, required=True)
parser.add_argument('--separate-reward-repr', action='store_true')
parser.add_argument('--bayes-reward-filter', action='store_true')
parser.add_argument('--use-ideal-filter', action='store_true')
parser.add_argument('--num-partitions', type=int, required=True)
parser.add_argument('--use-meta-controller', action='store_true')

args = parser.parse_args()

use_gpu = args.gpu_num >= 0
mode = args.mode
visual = args.visual

observation_mode = 'image' if visual else 'vector'

def default_visualizations(network, env, name):
    [path, name] = os.path.split(name)
    [name, name_extension] = name.split('.')
    statistics_full_name = os.path.join(path, name + '_statistics.txt')
    behavior_full_name = os.path.join(path, name + '_behavior_file.pickle')
    produce_reward_statistics(network, env, statistics_full_name, behavior_full_name)

def default_on_reward_print_func(r, sp, info, network, reward_buffer):
    partitioned_r = network.get_partitioned_reward([sp], [r])[0]
    print(reward_buffer.length(), partitioned_r)


if mode == 'ASSAULT':
    num_partitions = args.num_partitions
    num_visual_channels = 9
    on_reward_print_func = default_on_reward_print_func
    visualization_func = default_visualizations
    # visual mode must be on for Assault domain.
    assert visual
    env = AssaultWrapper()
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: AssaultWrapper(),
                                            AssaultWrapper)
    dummy_env_cluster('reset', args=[])
    dummy_env = SimpleAssault(initial_states_file=None)
    dummy_env.reset()
elif mode == 'SOKOBAN':
    num_partitions = args.num_partitions
    num_visual_channels = 3
    visualization_func = produce_two_goal_visualization
    on_reward_print_func = default_on_reward_print_func


    env = BlockPushingDomain(observation_mode=observation_mode)
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: BlockPushingDomain(observation_mode=observation_mode),
                                            BlockPushingDomain)
    dummy_env_cluster('reset', args=[])
    dummy_env = BlockPushingDomain(observation_mode=observation_mode)
    dummy_env.reset()
elif mode == 'PACMAN':
    num_partitions = args.num_partitions
    num_visual_channels = 9

    on_reward_print_func = default_on_reward_print_func
    visualization_func = default_visualizations

    env = PacmanWrapper()
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: PacmanWrapper(),
                                            PacmanWrapper)
    dummy_env_cluster('reset', args=[])

    dummy_env = PacmanWrapper()
    dummy_env.reset()
elif mode == 'SEAQUEST':
    num_partitions = args.num_partitions
    num_visual_channels = 9

    on_reward_print_func = default_on_reward_print_func
    visualization_func = default_visualizations

    env = SeaquestWrapper()
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: SeaquestWrapper(),
                                            SeaquestWrapper)
    dummy_env_cluster('reset', args=[])

    dummy_env = SeaquestWrapper()
    dummy_env.reset()
elif mode == 'BREAKOUT':
    num_partitions = args.num_partitions
    num_visual_channels = 9

    on_reward_print_func = default_on_reward_print_func
    visualization_func = default_visualizations

    env = BreakoutWrapper()
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: BreakoutWrapper(),
                                            BreakoutWrapper)
    dummy_env_cluster('reset', args=[])

    dummy_env = BreakoutWrapper()
    dummy_env.reset()
elif mode == 'ALIEN':
    num_partitions = args.num_partitions
    num_visual_channels = 9

    on_reward_print_func = default_on_reward_print_func
    visualization_func = default_visualizations

    env = AlienWrapper()
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: AlienWrapper(),
                                            AlienWrapper)
    dummy_env_cluster('reset', args=[])

    dummy_env = AlienWrapper()
    dummy_env.reset()
elif mode == 'QBERT':
    num_partitions = args.num_partitions
    num_visual_channels = 9

    on_reward_print_func = default_on_reward_print_func
    visualization_func = default_visualizations


    env = QBertWrapper()
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: QBertWrapper(),
                                            QBertWrapper)
    dummy_env_cluster('reset', args=[])

    dummy_env = QBertWrapper()
    dummy_env.reset()

else:
    raise Exception(f'mode must be in {mode_options}.')

build_directory_structure('.', {'runs': {
                                    args.name: {
                                        'images': {},
                                        'weights': {}}}})
LOG.setup(f'./runs/{args.name}')
save_path = os.path.join('runs', args.name, 'weights')

#agent = QLearnerAgent(env.observation_space.shape[0], env.action_space.n)
buffer = ReplayBuffer(10000)

state_replay_buffer = StateReplayBuffer(10000)

reward_net = RewardPartitionNetwork(env, buffer, state_replay_buffer, num_partitions, env.observation_space.shape[0],
                                    env.action_space.n, 'reward_net', traj_len=args.traj_len,  gpu_num=args.gpu_num,
                                    use_gpu=use_gpu, num_visual_channels=num_visual_channels, visual=visual,
                                    max_value_mult=args.max_value_mult, use_dynamic_weighting_disentangle_value=args.dynamic_weighting_disentangle,
                                    lr=args.learning_rate, reuse_visual_scoping=args.reuse_visual, separate_reward_repr=args.separate_reward_repr,
                                    use_ideal_threshold=args.use_ideal_filter)

meta_env = MetaEnvironment(env, reward_net.Q_networks)
meta_controller_buffer = ReplayBuffer(10000)
reward_meta_controller_buffer = ReplayBuffer(10000)
meta_controller = QLearnerAgent(meta_env.observation_space.shape[0], meta_env.action_space.n, 'meta_q_net', visual=visual, num_visual_channels=num_visual_channels, gpu_num=args.gpu_num)

(height, width, depth) = env.observation_space.shape
tracker = RewardProbTracker(height, width, depth)


learning_starts = 10000
batch_size = 32
q_train_freq = 4
epsilon = 0.01
episode_reward = 0
print(env.action_space)
epsilon = 1.0
min_epsilon = 0.01
num_epsilon_steps = 1000000
min_reward_experiences = 500
num_reward_steps = 30000
save_freq = 1000
evaluation_frequency = 100
current_reward_training_step = 0 if args.separate_reward_repr else num_reward_steps
epsilon_delta = (epsilon - min_epsilon) / num_epsilon_steps
time = 0

# indices of current policy
policy_indices = list(range(num_partitions)) + [-1]
current_policy = choice(policy_indices)

def get_action_meta_controller(s):
    global epsilon, meta_controller
    is_random = np.random.uniform(0,1) < epsilon
    if is_random:
        # flip a coin to decide if the random action will be low-level or high-level
        action = -1 if np.random.uniform(0,1) < 0.5 else np.random.randint(0,meta_env.action_space.n)
    else:
        action = meta_controller.get_action([s])[0]
    return action


def get_action(s):
    global epsilon
    global current_policy
    is_random = np.random.uniform(0, 1) < epsilon
    if current_policy == -1 or is_random:
        action = np.random.randint(0, env.action_space.n)
    else:
        action = reward_net.get_state_actions([s])[current_policy][0]
    return action

def evaluate_performance(env, q_network: QLearnerAgent):
    max_steps = 1000
    s = env.reset()
    cumulative_reward = 0
    for i in range(max_steps):
        a = np.random.randint(0, env.action_space.n) if np.random.uniform(0,1) < 0.01 else q_network.get_action([s])[0]
        s, r, t, _ = env.step(a)
        cumulative_reward += r
    return cumulative_reward, env.reset()


current_episode_length = 0
max_length_before_policy_switch = -1
update_threshold_frequency = 100
(h, w, d) = env.observation_space.shape

s = env.reset()

ideal_threshold = (cv2.imread('./ideal_threshold.png')[:, :, [0]] / 255).astype(np.uint8)

reward_tracker_zero_filter = 0
while True:
    # take random action
    #a = np.random.randint(0, env.action_space.n)
    if args.use_meta_controller:
        meta_a = get_action_meta_controller(s)
        if meta_a == -1:
            a = np.random.randint(0, env.action_space.n)
            sp, r, t, info = env.step(a)
        else:
            sp, r, t, info = meta_env.step(meta_a)
            a = info['a']
    else:
        a = get_action(s)
        sp, r, t, info = env.step(a)

    state_replay_buffer.append(env.get_current_state())

    #if r > 0:
        #partitioned_r = reward_net.get_partitioned_reward([sp], [r])[0]
        #print(f'{reward_buffer.length()}/{1000}')
        #reward_buffer.append(s, a, r, sp, t)
        #if args.use_meta_controller and meta_a != -1:
        #    reward_meta_controller_buffer.append(s, meta_a, r, sp, t)
        #on_reward_print_func(r, sp, info, reward_net, reward_buffer)
        #if args.bayes_reward_filter:
        #    tracker.add(sp, r)
        #LOG.add_line('max_reward_on_positive', np.max(partitioned_r))
        #image = np.concatenate([sp[:,:,0:3], sp[:,:,3:6], sp[:,:,6:9]], axis=1)
        #cv2.imwrite(f'pos_reward_{i}.png', cv2.resize(image, (400*3, 400), interpolation=cv2.INTER_NEAREST))
        #print(r, partitioned_r)
    #else:
    #    if args.bayes_reward_filter and reward_tracker_zero_filter % 100 == 0:
    #        tracker.add(sp, r)

    reward_tracker_zero_filter += 1


    episode_reward += r
    #env.render()
    buffer.append(s, a, r, sp, t)
    if args.use_meta_controller and meta_a != -1:
        meta_controller_buffer.append(s, meta_a, r, sp, t)

    if info['internal_terminal']:
        current_episode_length = 0
        current_policy = choice(policy_indices)

    if t:
        s = env.reset()
        current_policy = choice(policy_indices)
        current_episode_length = 0
        print(f'Episode Reward: {episode_reward}')
        #print(f'Epsilon {epsilon}')
        episode_reward = 0
    else:
        s = sp

    #epsilon = max(min_epsilon, epsilon - epsilon_delta)

    #if args.bayes_reward_filter and (reward_buffer.length() >= min_reward_experiences) and (time % update_threshold_frequency == 0):
    #    threshold = np.max(tracker.compute_threshold_image(0.09), axis=2, keepdims=True)
    #    reward_net.update_threshold_image(threshold)

    # need to  figure out what these constraints imply. They should only ever be active when we're in meta-controller mode.
    extra_meta_controller_constraints = \
        (current_reward_training_step >= num_reward_steps and # lets the reward predictor train before the agents start (if reward prediction is not enabled, this is always true)
        (meta_controller_buffer.length() >= batch_size or not args.use_meta_controller) and  # wait for the meta-controller's buffer to be sufficiently full.
        (reward_meta_controller_buffer.length() >= min_reward_experiences or not args.use_meta_controller)) # same deal but with its reward buffer.




    if time >= learning_starts and extra_meta_controller_constraints:

        if time % q_train_freq == 0:
            q_losses = reward_net.train_Q_networks(time)
            # tensorboard logging.
            for j in range(num_partitions):
                LOG.add_line(f'q_loss{j}', q_losses[j])

        if args.use_meta_controller:
            no_reward_S, no_reward_A, no_reward_R, no_reward_SP, no_reward_T = meta_controller_buffer.sample(batch_size // 2)
            reward_S, reward_A, reward_R, reward_SP, reward_T = reward_meta_controller_buffer.sample(batch_size // 2)
            S = no_reward_S + reward_S
            A = no_reward_A + reward_A
            R = no_reward_R + reward_R
            SP = no_reward_SP + reward_SP
            T = no_reward_T + reward_T
            meta_controller_loss = meta_controller.train_batch(S, A, R, SP, T)
        if time % (q_train_freq * 10) == 0:
            for j in range(1):
                reward_loss, max_value_constraint, value_constraint, J_indep, J_nontrivial = reward_net.train_R_function(dummy_env_cluster)
                LOG.add_line('reward_loss', reward_loss)
                LOG.add_line('max_value_constraint', max_value_constraint)
                LOG.add_line('value_constraint', value_constraint)
                LOG.add_line('J_indep', J_indep)
                LOG.add_line('J_nontrivial', J_nontrivial)
                if args.use_meta_controller:
                    LOG.add_line('meta_controller_loss', meta_controller_loss)

        #if args.separate_reward_repr:
        #    pred_reward_loss = reward_net.train_predicted_reward()




        log_string = f'({time}, eps: {epsilon}) ' + \
                     ''.join([f'Q_{j}_loss: {q_losses[j]}\t' for j in range(num_partitions)]) + \
                     f'Reward Loss: {reward_loss}' + \
                     f'(MaxValConst: {max_value_constraint}, ValConst: {value_constraint})'
        print(log_string)

        if time % 100 == 0:
            visualization_func(reward_net, dummy_env, f'./runs/{args.name}/images/policy_vis_{time}.png')

        if time % save_freq == 0:
            reward_net.save(save_path, 'reward_net.ckpt')

        if time % evaluation_frequency == 0:
            # evaluate the performance and reset the environment.
            eval_cum_reward, s = evaluate_performance(meta_env, meta_controller)
            LOG.add_line('eval_cum_reward', eval_cum_reward)

        epsilon = max(min_epsilon, epsilon - epsilon_delta)
    time += 1







    current_episode_length += 1



