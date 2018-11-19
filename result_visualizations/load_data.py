import os, re, tensorflow as tf
import pickle, numpy as np, tqdm
import matplotlib.pyplot as plt

def load_mapping():
    file = '/Users/chris/projects/q_learning/result_visualizations/new_commands'
    with open(file, 'r') as f:
        text = f.read()
    paths = [os.path.split(x[:-1]) for x in re.findall(r'\/home.*?[\,\}]', text.replace('\n',''))]
    paths = [(path.split('/')[-1], name) for (path, name) in paths]
    paths = {event_name: run_name for (run_name, event_name) in paths}
    return paths

def load_tb_data(file_path, fields):
    data = {f: [] for f in fields}
    fields_set = set(fields)
    for s in tf.train.summary_iterator(file_path):
        step = s.step
        for v in s.summary.value:
            if v.tag in fields_set:
                data[v.tag].append((step, v.simple_value))
    return data

def cut_down_meta_data():
    meta_runs_path = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/meta_runs/'
    cut_down_path = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/cut_down_meta'
    dirs = os.listdir(meta_runs_path)
    for dir in dirs:
        match = re.match(r'^meta\_(.+?)\_(\d)reward\_10mult\_(\d)$', dir)
        if not match:
            continue
        (game, reward_partitions, run_num) = match.groups()
        print(game, reward_partitions, run_num)
        # get the events file
        event_files = [x for x in os.listdir(os.path.join(meta_runs_path, dir)) if 'events' in x]
        if len(event_files) != 1:
            error = f'Found {len(event_files)} event files in {dir}. Skipping.'
            print(error)
            continue
        event_file = event_files[0]
        file_path = os.path.join(meta_runs_path, dir, event_file)
        data = load_tb_data(file_path, ['cum_reward', 'time'])
        with open(os.path.join(cut_down_path, dir), 'wb') as f:
            pickle.dump(data, f)

def cut_down_baseline_data():
    meta_runs_path = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/new_baselines/'
    cut_down_path = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/cut_down_baselines'
    dirs = os.listdir(meta_runs_path)
    for dir in dirs:
        match = re.match(r'^baseline\_(.+?)\_(\d)reward\_10mult\_(\d)$', dir)
        if not match:
            continue
        (game, reward_partitions, run_num) = match.groups()
        print(game, reward_partitions, run_num)
        # get the events file
        event_files = [x for x in os.listdir(os.path.join(meta_runs_path, dir)) if 'events' in x]
        if len(event_files) != 1:
            error = f'Found {len(event_files)} event files in {dir}. Skipping.'
            print(error)
            continue
        event_file = event_files[0]
        file_path = os.path.join(meta_runs_path, dir, event_file)
        data = load_tb_data(file_path, ['cum_reward', 'time'])
        with open(os.path.join(cut_down_path, dir), 'wb') as f:
            pickle.dump(data, f)



def cut_down_data(fields, dont_repeat_work=True,
                  base_path='/Users/chris/projects/q_learning/new_dqn_results/completed_runs/',
                  cut_down_path='/Users/chris/projects/q_learning/new_dqn_results/completed_runs/new_cut_down',
                  directories=('new_rldl3_runs', 'new_rldl5_runs', 'new_rldl11_runs', 'new_rldl4_runs')):
    mapping = load_mapping()
    for d in directories:
        path = os.path.join(base_path, d)
        event_names = [x for x in os.listdir(path) if x.startswith('events')]
        for event_name in event_names:
            full_path = os.path.join(path, event_name)
            run_name = mapping[event_name]
            print(run_name, full_path)
            # dont repeat
            if dont_repeat_work:
                if os.path.isfile(os.path.join(cut_down_path, run_name)):
                    print(f'skipping {run_name}...')
                    continue
            data = load_tb_data(full_path, fields)
            with open(os.path.join(cut_down_path, run_name), 'wb') as f:
                pickle.dump(data, f)



def load_cutdown_data():
    cut_down_path = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/new_cut_down'
    names = os.listdir(cut_down_path)
    all_data = {}
    for name in tqdm.tqdm(names):
        with open(os.path.join(cut_down_path, name), 'rb') as f:
            data = replace_step_with_time(pickle.load(f))
        all_data[name] = data
    return all_data


def replace_step_with_time(data):
    # there should be a time for every step
    time_mapping = {step: time for step, time in data['time']}
    new_data = {field: [(time_mapping[step], value) for step, value in data[field]]
                for field in data.keys()}
    return new_data


def create_merging_bins(all_data):
    bins = {}
    bin_names = {}
    for full_name in all_data.keys():

        match = re.match(r'^(restore\d?\_)?(.+?)\_(\d)reward\_(\d+)mult(\_\d)?$', full_name)
        if match is None:
            print(f'Rejecting name: {full_name}')
            continue

        (restored, name, num_rewards, num_mult, run_flag) = match.groups()
        # priority system should define how the run data should be merged if a run had to be restarted.
        if restored is None:
            priority = -2
        else:
            priority = re.match(r'^restore(\d?)\_$', restored).groups()[0]
            if priority is None:
                priority = -1
        merge_id = (name,num_rewards,num_mult,run_flag)
        if merge_id not in bins:
            bins[merge_id] = [all_data[full_name]]
            bin_names[merge_id] = [full_name]
        else:
            bins[merge_id].append(all_data[full_name])
            bin_names[merge_id].append(full_name)
    for merge_id, full_names in bin_names.items():
        print(merge_id, full_names)
    return bins

def merge_data(bin):
    print()
    fields = bin[0].keys()
    merged_data = {}
    for field in fields:
        merged_data[field] = []
        for data in bin:
            merged_data[field].extend(data[field])
        merged_data[field] = sorted(merged_data[field], key=lambda x: x[0])
    return merged_data



def load_and_merge_data():
    data = load_cutdown_data()
    bins = create_merging_bins(data)
    merged_data = {}
    for name, bin in bins.items():
        print(name, len(bin))
        merged_data[name] = merge_data(bin)
    return merged_data

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def make_J_disentangled_plots(merged_data, n=2000):
    path = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/j_disentangled_plots'
    game_bins = {}
    for (name, num_rewards, mult, run_num), data in merged_data.items():
        if (name, num_rewards) not in game_bins:
            game_bins[(name, num_rewards)] = [data]
        else:
            game_bins[(name, num_rewards)].append(data)
    for (name, num_rewards), data_list in game_bins.items():
        plt.clf()
        plt.hold(True)
        for data in data_list:
            x = [time for time, J in data['J_disentangled']][n-1:]
            y = moving_average([J for time, J in data['J_disentangled']],n=n)
            plt.plot(x, y)
        plt.hold(False)
        plt.savefig(os.path.join(path, f'{name}_{num_rewards}reward.pdf'))




if __name__ == '__main__':
    #data = cut_down_data(['J_disentangled', 'J_indep', 'J_nontrivial', 'time'])
    data = cut_down_meta_data()
    #merged_data = load_and_merge_data()
    #make_J_disentangled_plots(merged_data)
