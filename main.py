import numpy as np
from envs.atari.atari_wrapper import PacmanWrapper, QBertWrapper, AssaultWrapper, AlienWrapper, SeaquestWrapper, BreakoutWrapper
from q_learner_agent import QLearnerAgent
from envs.metacontroller_actor import MetaEnvironment
from envs.atari.simple_assault import SimpleAssault
from visualization import produce_two_goal_visualization, produce_assault_ship_histogram_visualization, produce_assault_reward_visualization, produce_reward_statistics, visualize_all_representations_all_reward_images, record_value_matrix
from utils import LOG, build_directory_structure, add_implicit_name_arg
import argparse
from random import choice
from switch_world_visualizations.switchworld_vis_reward import switchworld_vis_reward
import cv2
import tqdm
import os, re
import tensorflow as tf
from replay_buffer import StateReplayBuffer

from envs.atari.threaded_environment import ThreadedEnvironment
from envs.block_world.block_pushing_domain import BlockPushingDomain
from replay_buffer import ReplayBuffer
from reward_network import RewardPartitionNetwork
from envs.switch_world import SwitchWorld
from utils import LOG, build_directory_structure

parser = argparse.ArgumentParser()
add_implicit_name_arg(parser)

parser.add_argument('--product-mode', action='store_true')
parser.add_argument('--traj-len', type=int, default=20)
parser.add_argument('--max-value-mult', type=float, default=10.0)
parser.add_argument('--dynamic-weighting-disentangle', action='store_true')
parser.add_argument('--mode', type=str, required=True, choices=['SOKOBAN', 'SOKOBAN_REWARD_ALWAYS_ONE', 'SOKOBAN_OBSTACLE', 'SOKOBAN_FOUR_ROOM', 'ASSAULT', 'PACMAN', 'QBERT', 'ALIEN', 'BREAKOUT', 'SEAQUEST', 'SWITCHWORLD'])
parser.add_argument('--visual', action='store_true')
parser.add_argument('--learning-rate', type=float, default=0.00005)
parser.add_argument('--gpu-num', type=int, required=True)
parser.add_argument('--num-partitions', type=int, required=True)
parser.add_argument('--use-meta-controller', action='store_true')
parser.add_argument('--clip-gradient', type=float, default=-1)
parser.add_argument('--restore-dead-run', type=str, default=None)
parser.add_argument('--softmin-temp', type=float, default=1.0)
parser.add_argument('--stop-softmin-gradient', action='store_true')
parser.add_argument('--run-dir', type=str, default='new_runs')


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

    produce_reward_statistics(network, env, statistics_full_name, behavior_full_name)
    record_value_matrix(value_matrix, value_matrix_full_name)
    file_number = re.match(r'^policy\_vis\_(\d+)\_behavior_file.pickle$', behavior_name).groups()[0]
    #produce_all_videos(path, file_number)


def default_on_reward_print_func(r, sp, info, network, reward_buffer):
    partitioned_r = network.get_partitioned_reward([sp], [r])[0]
    print(reward_buffer.length(), partitioned_r)


display_freq = 10000


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

    config = 'standard'
    env = BlockPushingDomain(observation_mode=observation_mode, configuration=config)
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: BlockPushingDomain(observation_mode=observation_mode,
                                                                         configuration=config),
                                            BlockPushingDomain)
    dummy_env_cluster('reset', args=[])
    dummy_env = BlockPushingDomain(observation_mode=observation_mode, configuration=config)
    dummy_env.reset()
elif mode == 'SOKOBAN_REWARD_ALWAYS_ONE':
    num_partitions = args.num_partitions
    num_visual_channels = 3
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

    num_visual_channels = 3
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

    num_visual_channels = 3
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
elif mode == 'SWITCHWORLD':
    num_partitions = args.num_partitions

    num_visual_channels = 3
    #visualization_func = produce_two_goal_visualization
    visualization_func = switchworld_vis_reward
    on_reward_print_func = default_on_reward_print_func

    display_freq = 1000
    env = SwitchWorld()
    dummy_env_cluster = ThreadedEnvironment(32,
                                            lambda i: SwitchWorld(),
                                            SwitchWorld)
    dummy_env_cluster('reset', args=[])
    dummy_env = SwitchWorld()
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
buffer = ReplayBuffer(100000)

state_replay_buffer = StateReplayBuffer(100000)

reward_net = RewardPartitionNetwork(env, buffer, state_replay_buffer, num_partitions, env.observation_space.shape[0],
                                    env.action_space.n, 'reward_net', traj_len=args.traj_len,  gpu_num=args.gpu_num,
                                    use_gpu=use_gpu, num_visual_channels=num_visual_channels, visual=visual,
                                    max_value_mult=args.max_value_mult, use_dynamic_weighting_disentangle_value=args.dynamic_weighting_disentangle,
                                    lr=args.learning_rate, product_mode=args.product_mode,
                                    softmin_temperature=args.softmin_temp, stop_softmin_gradients=args.stop_softmin_gradient,
                                    )

meta_env = MetaEnvironment(env, reward_net, reward_net.Q_networks, True, 1)
meta_controller_buffer = ReplayBuffer(10000)
reward_meta_controller_buffer = ReplayBuffer(10000)
meta_controller = QLearnerAgent(meta_env.observation_space.shape[0], meta_env.action_space.n, 'meta_q_net', visual=visual, num_visual_channels=num_visual_channels, gpu_num=args.gpu_num)

(height, width, depth) = env.observation_space.shape

learning_starts = 10000
batch_size = 32
q_train_freq = 4
q_loss_log_freq = 100
epsilon = 0.01
episode_reward = 0
print(env.action_space)
epsilon = 1.0
min_epsilon = 0.01
num_epsilon_steps = 1000000
min_reward_experiences = 500
num_reward_steps = 30000
save_freq = 10000
evaluation_frequency = 10000
current_reward_training_step = num_reward_steps
epsilon_delta = (epsilon - min_epsilon) / num_epsilon_steps
time = 0
num_steps = 10000000

# indices of current policy
policy_indices = list(range(num_partitions)) + [-1]
current_policy = choice(policy_indices)


def restore_dead_run():
    run_path = os.path.join(run_dir, args.restore_dead_run)
    # first get the models loaded up
    reward_net.restore(os.path.join(run_path, 'weights'), 'reward_net.ckpt')
    # get the time that we made it to in the dead run.
    event_file_names = [x for x in os.listdir(run_path) if x.startswith('events')]
    if len(event_file_names) > 1:
        raise Exception('Too many event files. Disambiguation required.')
    elif len(event_file_names) == 0:
        raise Exception(f'Cannot find event file in directory: {run_path}')
    event_file_name = event_file_names[0]
    event_path = os.path.join(run_path, event_file_name)
    largest_time = 0
    for s in tf.train.summary_iterator(event_path):
        for e in s.summary.value:
            if e.tag == 'time':
                print(e.simple_value)
                largest_time = max(e.simple_value, largest_time)
    # fill the replay buffer and state-buffers with experiences
    replay_buffer_size = 100000
    fill_steps = min(replay_buffer_size, int(largest_time))
    epsilon = max(1.0 - largest_time * epsilon_delta, min_epsilon)
    s = env.reset()
    for t in tqdm.tqdm(range(fill_steps)):
        a = get_action(s)
        sp, r, t, _ = env.step(a)
        buffer.append(s, a, r, sp, t)
        state_replay_buffer.append(env.get_current_state())
        if t:
            s = env.reset()
        else:
            s = sp
    return int(largest_time), epsilon







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
(h, w, d) = env.observation_space.shape
last_100_scores = [np.inf]
best_score = np.inf
s = env.reset()

starting_time = 0
if args.restore_dead_run is not None:
    starting_time, epsilon = restore_dead_run()


for time in range(starting_time, num_steps):
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

    # need to  figure out what these constraints imply. They should only ever be active when we're in meta-controller mode.
    extra_meta_controller_constraints = \
        (current_reward_training_step >= num_reward_steps and # lets the reward predictor train before the agents start (if reward prediction is not enabled, this is always true)
        (meta_controller_buffer.length() >= batch_size or not args.use_meta_controller) and  # wait for the meta-controller's buffer to be sufficiently full.
        (reward_meta_controller_buffer.length() >= min_reward_experiences or not args.use_meta_controller)) # same deal but with its reward buffer.




    if time >= learning_starts and extra_meta_controller_constraints:

        if time % q_train_freq == 0:
            q_losses = reward_net.train_Q_networks(time)
            # tensorboard logging.
            if time % q_loss_log_freq == 0:
                for j in range(num_partitions):
                    LOG.add_line(f'q_loss{j}', q_losses[j])

        if time % (q_train_freq * 5) == 0:
            for j in range(1):
                loss, J_indep, J_nontriv = reward_net.train_R_function(dummy_env_cluster)
                LOG.add_line('loss', loss)
                #LOG.add_line('value_constraint', value_constraint)
                LOG.add_line('J_indep', J_indep)
                LOG.add_line('J_nontriv', J_nontriv)
                #LOG.add_line('J_disentangled', J_indep - J_nontrivial)
                #LOG.add_line('time', time)
                # TODO actually log the value_partition
                if len(last_100_scores) < 100:
                    last_100_scores.append(loss)
                else:
                    last_100_scores = last_100_scores[1:] + [loss]

        #if args.separate_reward_repr:
        #    pred_reward_loss = reward_net.train_predicted_reward()




        log_string = f'({time}, eps: {epsilon}) ' + \
                     ''.join([f'Q_{j}_loss: {q_losses[j]}\t' for j in range(num_partitions)]) + \
                     f'Reward Loss: {loss}' + \
                     f'(J_indep: {J_indep}, J_nontriv: {J_nontriv})'
        print(log_string)

        if time % display_freq == 0:
            visualization_func(reward_net, dummy_env, None, f'./{run_dir}/{args.name}/images/policy_vis.txt')

        if time % save_freq == 0:
            reward_net.save(save_path, 'reward_net.ckpt')

        if time % evaluation_frequency == 0:
            # evaluate the performance and reset the environment.
            eval_cum_reward, s = evaluate_performance(meta_env, meta_controller)
            LOG.add_line('eval_cum_reward', eval_cum_reward)

        if time % 10000 == 0 and np.mean(last_100_scores) < best_score:
            best_score = np.mean(last_100_scores)
            reward_net.save(best_save_path, 'reward_net.ckpt')



        epsilon = max(min_epsilon, epsilon - epsilon_delta)







    current_episode_length += 1



