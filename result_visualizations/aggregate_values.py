import numpy as np
import matplotlib.pyplot as plt
import re

def read_in_values(txt):
    mapping = dict()
    for line in txt:
        [mode, name, reward_num, value, variance] = line.split(',')
        if name in mapping:
            mapping[name].append((float(value), float(variance)))
        else:
            mapping[name] = [(float(value), float(variance))]
    for key, pairs in mapping.items():
        value = [p[0] for p in pairs]
        variance = [p[1] for p in pairs]
        mapping[key] = (np.mean(value), np.mean(variance))
    return mapping


with open('icf_values_stats.txt', 'r') as f:
    icf_values = f.readlines()
    icf_values = read_in_values(icf_values)

with open('rd_values_stats.txt', 'r') as f:
    rd_values = f.readlines()
    rd_values = read_in_values(rd_values)



icf_policies = ['assault_2reward_3', 'assault_3reward_2', 'assault_5reward_2', 'assault_8reward_1',
                'seaquest_2reward_2', 'seaquest_3reward_3', 'seaquest_5reward_1', 'seaquest_8reward_3',
                'pacman_2reward_3', 'pacman_3reward_3', 'pacman_5reward_3', 'pacman_8reward_2']

rd_policies = ['assault_2reward_10mult_1', 'assault_3reward_10mult_2', 'assault_5reward_10mult_3', 'assault_8reward_10mult_3',
               'seaquest_2reward_10mult_3', 'seaquest_3reward_10mult_4', 'seaquest_5reward_10mult_4', 'seaquest_8reward_10mult_2',
               'pacman_2reward_10mult_2', 'pacman_3reward_10mult_3', 'pacman_5reward_10mult_1', 'pacman_8reward_10mult_2',
               ]

for icf, rd in zip(icf_policies, rd_policies):
    name = icf.split('_')[0]
    print(name, rd_values[rd], icf_values[icf])


def make_barchart(game_name, icf_policies, rd_policies):
    entries = []
    for icf, rd in zip(icf_policies, rd_policies):
        reward = int(re.match(r'^.+?(\d+)reward.+?$', icf).groups()[0])
        entries.append((reward, (rd_values[rd], icf_values[icf])))
    entries = sorted(entries, key=lambda x: x[0])
    icf_means = [icf for (reward, (rd, icf)) in entries]
    icf_means = [rd for (reward, (rd, icf)) in entries]
    rewards = [reward for (reward, (rd, icf)) in entries]
    plt.bar()





# n_groups = 4
# means_frank = (90, 55, 40, 65)
# means_guido = (85, 62, 54, 20)
#
# # create plot
# fig, ax = plt.subplots()
# index = np.arange(n_groups)
# bar_width = 0.35
# opacity = 0.8
#
# rects1 = plt.bar(index, means_frank, bar_width,
#                  alpha=opacity,
#                  color='b',
#                  label='Frank')
#
# rects2 = plt.bar(index + bar_width, means_guido, bar_width,
#                  alpha=opacity,
#                  color='g',
#                  label='Guido')
#
# plt.xlabel('Person')
# plt.ylabel('Scores')
# plt.title('Scores by person')
# plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
# plt.legend()
#
# plt.tight_layout()
# plt.show()
