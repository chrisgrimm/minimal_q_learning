import numpy as np
import csv
import matplotlib.pyplot as plt
import os


def read_csv(csv_file):
    data = []
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for line, row in enumerate(csv_reader):
            if line == 0:
                print(line)
            else:
                data.append((int(row[1]), float(row[2])))
    return data

J_indep_data = read_csv('/Users/chris/Downloads/run_part4_sokoban_.-tag-J_indep.csv')
J_nontriv_data = read_csv('/Users/chris/Downloads/run_part4_sokoban_.-tag-J_nontrivial.csv')

def get_disentangled_data(J_indep_data, J_nontriv_data):
    data_X, data_Y = [], []
    for (pos1, indep), (pos2, nontriv) in zip(J_indep_data, J_nontriv_data):
        print((pos1, indep), (pos2, nontriv))
        #assert pos1 == pos2
        data_X.append(pos1)
        data_Y.append(indep - nontriv)
    return data_X, data_Y


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def create_disentanglement_plot(plot_name, path, run_names, run_labels, master_title=None):
    for run_name, run_label in zip(run_names, run_labels):
        J_indep_path = os.path.join(path, f'{run_name}_.-tag-J_indep.csv')
        J_nontriv_path = os.path.join(path, f'{run_name}_.-tag-J_nontrivial.csv')
        J_indep_data = read_csv(J_indep_path)
        J_nontriv_data = read_csv(J_nontriv_path)
        n = 25
        x, y = get_disentangled_data(J_indep_data, J_nontriv_data)
        x, y = x[n-1:], moving_average(y, n=n)
        plt.hold(True)
        plt.plot(x, y, label=run_label)
        plt.xlabel('Timesteps')
        plt.ylabel('Disentanglement Score')
    if master_title is not None:
        plt.title(master_title)
    plt.legend(loc='bottom right')
    plt.savefig(plot_name)



def filter_by_step(x_data, y_data, length):
    x_new, y_new = [], []
    for x, y in zip(x_data, y_data):
        if x <= length:
            x_new.append(x)
            y_new.append(y)
    return x_new, y_new

def create_metacontroller_plot(plot_name, path, run_name, title=None):
    dqn_baseline_path = os.path.join(path, f'run_dqn_{run_name}_.-tag-cum_reward.csv')
    meta2_path = os.path.join(path, f'run_dqn_{run_name}_meta_part2_.-tag-cum_reward.csv')
    meta3_path = os.path.join(path, f'run_dqn_{run_name}_meta_part3_.-tag-cum_reward.csv')
    meta4_path = os.path.join(path, f'run_dqn_{run_name}_meta_part4_.-tag-cum_reward.csv')


    dqn_baseline_data = read_csv(dqn_baseline_path)
    meta2_data = read_csv(meta2_path)
    meta3_data = read_csv(meta3_path)
    meta4_data = read_csv(meta4_path)

    max_length = 500
    length = min([len(meta2_data), len(meta3_data), len(dqn_baseline_data), max_length])
    if length < max_length:
        print('Warning: data is small.')

    print('max_length')
    plt.hold(True)
    n = 25
    x, y = zip(*dqn_baseline_data)
    x, y = x[n-1:], moving_average(y, n=n)
    x, y = filter_by_step(x, y, length)
    plt.plot(x, y, label='DQN')

    x, y = zip(*meta2_data)
    x, y = x[n-1:], moving_average(y, n=n)
    x, y = filter_by_step(x, y, length)
    plt.plot(x, y, label='2 Reward')

    x, y = zip(*meta3_data)
    x, y = x[n-1:], moving_average(y, n=n)
    x, y = filter_by_step(x, y, length)
    plt.plot(x, y, label='3 Reward')

    x, y = zip(*meta4_data)
    x, y = x[n - 1:], moving_average(y, n=n)
    x, y = filter_by_step(x, y, length)
    plt.plot(x, y, label='4 Reward')

    plt.xlabel('Evaluation Epochs')
    plt.ylabel('Cumulative Reward')
    if title is not None:
        plt.title(title)
    plt.legend(loc='bottom right')
    plt.savefig(plot_name)

path = '/Users/chris/Downloads'
#runs = ['run_part2_sokoban', 'run_part3_sokoban', 'run_part4_sokoban']
runs = ['run_part2_qbert', 'run_part3_qbert']
#labels = ['2 Rewards', '3 Rewards']#, '4 Rewards']
#create_disentanglement_plot('all_qbert_disent.pdf', path, runs, labels, master_title='QBert Disentanglements')

run = 'sokoban'
create_metacontroller_plot('sokoban_metacontroller_plot.pdf', path, run, title='Gridworld Metacontroller')
#data_X, data_Y = get_disentangled_data(J_indep_data, J_nontriv_data)
#print(data_X)
#plt.hold(True)
#plt.plot(data_X, data_Y)
#plt.savefig('./disentangle_plot.pdf')