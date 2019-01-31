from envs.switch_world import SwitchWorld
from reward_network import RewardPartitionNetwork
import numpy as np

# def switchworld_vis_reward(reward_net : RewardPartitionNetwork, env : SwitchWorld, value_matrix, filepath: str):
#     policy_steps = 20
#     all_switch_histograms = []
#     for part_num in range(reward_net.num_partitions):
#         switch_histograms = np.array([0, 0, 0, 0])
#         s = env.restore_state(([False]*4, (2,2), 0))
#         got_reward = False
#         for _ in range(policy_steps):
#             a = reward_net.Q_networks[part_num].get_action([s])[0]
#             s, r, t, info = env.step(a)
#             if r == 1:
#                 got_reward = True
#                 break
#         if got_reward:
#             switch_states = [True, True, True, True]
#         else:
#             switch_states = env.switch_states
#         for i, switch_flipped in enumerate(switch_states):
#             switch_histograms[i] += int(switch_flipped)
#         all_switch_histograms.append(switch_histograms)
#     with open(filepath, 'a+') as f:
#         f.write('\n-------\n')
#         for i, histogram in enumerate(all_switch_histograms):
#             f.write(f'\t({i}) : {histogram}\n')

def switchworld_vis_reward(reward_net : RewardPartitionNetwork, env: SwitchWorld, value_matrix, filepath: str):
    action_sequence = ['a','a',
                       'w','w', # get top-left
                       'd','d','d','d', # get top-right
                       's','s','s','s']#, # get bottom-right
                       #'a','a','a','a'] # get bottom-left

    s = env.restore_state(([False] * 4, (2, 2), 0))
    activies = []
    for a_human in action_sequence:
        a = env.human_mapping[a_human]
        s, r, t, info = env.step(a)
        num_active = np.sum(reward_net.get_partitioned_reward([s], [r])[0])
        num_active = int(np.round(num_active))
        activies.append(num_active)
    with open(filepath, 'a+') as f:
        f.write(f'{",".join([str(x) for x in activies])}\n')

