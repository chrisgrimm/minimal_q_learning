from reward_network import RewardPartitionNetwork
from envs.atari.atari_wrapper import PacmanWrapper, AssaultWrapper, SeaquestWrapper
from utils import build_directory_structure
import os, numpy as np
import cv2
import tqdm

wrapper_mapping = {'assault': AssaultWrapper, 'pacman': PacmanWrapper, 'seaquest': SeaquestWrapper}
snapshot_dir = 'new_dqn_results/completed_runs/assault_snapshots'

#build_directory_structure(snapshot_dir, {''})

def store_snapshot(snapshot_dir, snapshot, reward_partition, partition_counts):
    reward_partition = np.argmax(reward_partition)
    file_num = partition_counts[reward_partition]
    partition_counts[reward_partition] += 1
    snapshot_path = os.path.join(snapshot_dir, f'part{reward_partition}', f'{file_num}.png')
    cv2.imwrite(snapshot_path, snapshot)

class KilledException(Exception):
    pass

def shoot_from_position(env, reward_net, position, reward_per_position):
    action_set = {' ': 2, 'd': 3, 'a': 4, '': 0}
    num_rewards_collected = 0
    for i in range(7):
        # take 7 steps to the right.
        s, r, t, info = env.step(action_set['d'])
        env.render()
        if info['internal_terminal']:
            return num_rewards_collected



    for i in range(position):
        # take position steps to the left.
        s, r, t, info = env.step(action_set['a'])
        env.render()
        if info['internal_terminal']:
            return num_rewards_collected
    #cv2.imwrite(f'new_dqn_results/completed_runs/assault_snapshots/position{position}.png', env.get_unprocessed_obs())
    while True:
        s, r, t, info = env.step(action_set[' '])
        env.render()
        if info['internal_terminal']:
            return num_rewards_collected
        if r == 1:
            partition = reward_net.get_partitioned_reward([s], [r])[0]
            reward_per_position[position] += np.array(partition)
            num_rewards_collected += 1

def collect_position_data(env, reward_net, position, reward_per_position):
    num_rewards_collected = 0
    while num_rewards_collected < 1:
        num_rewards_collected += shoot_from_position(env, reward_net, position, reward_per_position)
        print(f'Collected {num_rewards_collected}')
        env.reset()


def collect_rewards_for_all_positions(env, reward_net, num_partitions):
    num_positions = 12
    reward_per_position = [np.zeros([num_partitions], dtype=np.float32) for _ in range(num_positions)]
    for position in range(12):
        print(f'Shooting from position {position}')
        env.reset()
        collect_position_data(env, reward_net, position, reward_per_position)
    return reward_per_position

def color_sections(env_state, reward_per_position, reward_index):
    mask_alpha = np.zeros(list(env_state.shape[:2]), dtype=np.float32)
    mask_color = (0, 119, 255)
    mask = np.tile([[list(mask_color)]], list(env_state.shape[:2]) + [1])
    for position in range(12):
        start = 130 - 10*position
        end = 130 - 10*(position - 1)
        value = reward_per_position[position][reward_index] / np.sum(reward_per_position[position])
        mask_alpha[:, start:end] = value
    mask_alpha = np.reshape(mask_alpha, list(env_state.shape[:2]) + [1])
    mask_alpha = np.minimum(mask_alpha, 0.8)
    grayscale = np.tile(np.reshape(cv2.cvtColor(env_state, cv2.COLOR_BGR2GRAY), list(env_state.shape[:2]) + [1]), [1, 1, 3])

    result = mask * mask_alpha + grayscale * (1 - mask_alpha)
    #result = 255*mask_alpha
    return result

def generate_masks(path, env, reward_net, num_partitions):
    reward_per_position = collect_rewards_for_all_positions(env, reward_net, num_partitions)
    #reward_per_position = [np.random.uniform(0, 1, size=[num_partitions]) for _ in range(12)]
    for partition in range(num_partitions):
        #path = 'new_dqn_results/completed_runs/assault_snapshots/'
        res = color_sections(env.get_unprocessed_obs(), reward_per_position, partition)
        #print(res)
        image_path = os.path.join(path, f'partition{partition}.png')
        print(f'making image {partition} {path}')
        print('imwrite res:', cv2.imwrite(image_path, res))



if __name__ == '__main__':
    game = 'assault'
    env = wrapper_mapping[game]()
    num_partitions = 3
    run_number = 4
    build_directory_structure(snapshot_dir, {f'assault_{num_partitions}reward_{run_number}': {} })

    policy_num = 1
    num_visual_channels = 9
    path = f'new_dqn_results/new_weights/{game}_{num_partitions}reward_10mult_{run_number}/best_weights/'
    name = 'reward_net.ckpt'
    reward_net = reward_net = RewardPartitionNetwork(env, None, None, num_partitions, env.observation_space.shape[0],
                                                     env.action_space.n, 'reward_net', traj_len=10,
                                                     num_visual_channels=num_visual_channels, visual=True, gpu_num=-1)
    reward_net.restore(path, name)
    #collect_rewards_for_all_positions(env, reward_net, num_partitions)
    image_path = os.path.join(snapshot_dir, f'assault_{num_partitions}reward_{run_number}')
    generate_masks(image_path, env, reward_net, num_partitions)
    # s = env.reset()
    # print('num_actions', env.action_space.n)
    # partition_counts = [0 for _ in range(num_partitions)]
    # num_steps = 0
    # while True:
    #     try:
    #         action_set = {'w': 6, 'a': 3, 'd': 2, 's': 4}
    #         action_set = {' ': 2, 'd': 3, 'a': 4, '': 0}
    #         a = input()
    #         if a == 't':
    #             s = env.reset()
    #             continue
    #         if a in action_set:
    #            a = action_set[a]
    #         else:
    #            continue
    #         # a = reward_net.Q_networks[policy_num].get_action([s])[0]
    #     except:
    #         continue
    #     s, r, t, _ = env.step(a)
    #     env.render()
    #     #input('...')
    #     partition = reward_net.get_partitioned_reward([s], [r])[0]
    #     if r == 1:
    #         store_snapshot(os.path.join(snapshot_dir, f'assault_{num_partitions}reward_{run_number}'),
    #                        env.get_unprocessed_obs(), partition, partition_counts)
    #     print(r, partition)
    #     num_steps += 1
    #     #if num_steps > 1000:
    #     #    s = env.reset()
    #     #    policy_num = np.random.randint(0, num_partitions)



