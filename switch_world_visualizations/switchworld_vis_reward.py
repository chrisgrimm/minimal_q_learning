from envs.switch_world import SwitchWorld
from reward_network import RewardPartitionNetwork
import numpy as np

def switchworld_vis_reward(reward_net : RewardPartitionNetwork, env : SwitchWorld, value_matrix, filepath: str):
    policy_steps = 20
    all_switch_histograms = []
    for part_num in range(reward_net.num_partitions):
        switch_histograms = np.array([0, 0, 0, 0])
        s = env.restore_state(([False]*4, (0,0), 0))
        got_reward = False
        for _ in range(policy_steps):
            a = reward_net.Q_networks[part_num].get_action([s])[0]
            s, r, t, info = env.step(a)
            if r == 1:
                got_reward = True
                break
        if got_reward:
            switch_states = [True, True, True, True]
        else:
            switch_states = env.switch_states
        for i, switch_flipped in enumerate(switch_states):
            switch_histograms[i] += int(switch_flipped)
        all_switch_histograms.append(switch_histograms)
    with open(filepath, 'a+') as f:
        f.write('-------')
        for i, histogram in enumerate(all_switch_histograms):
            f.write(f'\t({i}) : {histogram}\n')
