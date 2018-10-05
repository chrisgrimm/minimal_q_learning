from reward_network import RewardPartitionNetwork
from envs.atari.pacman import PacmanWrapper

if __name__ == '__main__':
    env = PacmanWrapper()
    num_partitions = 2
    num_visual_channels = 9
    path = 'pacman_new_dqn_5x_freq/part2'
    name = 'reward_net.ckpt'
    reward_net = reward_net = RewardPartitionNetwork(env, None, None, num_partitions, env.observation_space.shape[0],
                                                     env.action_space.n, 'reward_net', traj_len=10,
                                                     num_visual_channels=num_visual_channels, visual=True, gpu_num=-1)
    reward_net.restore(path, name)

    s = env.reset()
    print('num_actions', env.action_space.n)
    while True:
        try:
            #action_set = {'w': 6, 'a': 3, 'd': 2, 's': 4}
            #a = input()
            #if a in action_set:
            #    a = action_set[a]
            #else:
            #    continue
            a = reward_net.Q_networks[1].get_action([s])[0]
        except:
            continue
        s, r, t, _ = env.step(a)
        env.render()
        input('...')
        partition = reward_net.get_partitioned_reward([s], [r])[0]
        print(r, partition)

