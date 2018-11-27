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
    baseline_runs_path = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/new_baselines2/'
    cut_down_path = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/cut_down_baselines'
    dirs = os.listdir(baseline_runs_path)
    for dir in dirs:
        match = re.match(r'^baseline\_(.+?)\_(\d)$', dir)
        if not match:
            continue
        (game, run_num) = match.groups()
        # get the events file
        event_files = [x for x in os.listdir(os.path.join(baseline_runs_path, dir)) if 'events' in x]
        if len(event_files) != 1:
            error = f'Found {len(event_files)} event files in {dir}. Skipping.'
            print(error)
            continue
        print(game, run_num)
        event_file = event_files[0]
        file_path = os.path.join(cut_down_path, dir)
        print(file_path)
        if os.path.isfile(file_path):
            print(f'Found {dir}. Skipping.')
            continue
        event_file = event_files[0]
        file_path = os.path.join(baseline_runs_path, dir, event_file)
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



def load_cutdown_data(filter_regex=None):
    cut_down_path = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/new_cut_down'
    names = os.listdir(cut_down_path)
    all_data = {}
    for name in tqdm.tqdm(names):
        print(name)
        if filter_regex is None or re.match(filter_regex, name):
            with open(os.path.join(cut_down_path, name), 'rb') as f:
                data = replace_step_with_time(pickle.load(f))
            all_data[name] = data
    return all_data

def load_metacontroller_and_baseline_data(single_game=None):
    baselines = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/cut_down_baselines'
    meta = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/cut_down_meta'
    meta_runs = dict()
    baseline_runs = dict()
    for name in tqdm.tqdm(os.listdir(baselines)):
        match = re.match(r'^baseline\_(.+?)\_(\d)$', name)
        if not match:
            print(f'{name} didnt match. Skipping...')
            continue
        (game, run_num) = match.groups()
        if single_game is not None and single_game != game:
            continue
        with open(os.path.join(baselines, name), 'rb') as f:
            data = replace_step_with_time(pickle.load(f))
            print(data.keys())
        if game in baseline_runs:
            baseline_runs[game].append(data)
        else:
            baseline_runs[game] = [data]
    for name in tqdm.tqdm(os.listdir(meta)):
        match = re.match(r'^meta\_(.+?)\_(\d)reward\_.+?\_(\d)$', name)
        if not match:
            print(f'{name} didnt match. Skipping...')
            continue
        (game, num_rewards, run_num) = match.groups()
        if single_game is not None and single_game != game:
            continue
        with open(os.path.join(meta, name), 'rb') as f:
            data = replace_step_with_time(pickle.load(f))
            print(data.keys())

        if game in meta_runs:
            if num_rewards in meta_runs[game]:
                meta_runs[game][num_rewards].append(data)
            else:
                meta_runs[game][num_rewards] = [data]
        else:
            meta_runs[game] = {num_rewards: [data]}
    return meta_runs, baseline_runs



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



def load_and_merge_data(filter_regex=None):
    data = load_cutdown_data(filter_regex=filter_regex)
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
            game_bins[(name, num_rewards)] = [(run_num, data)]
        else:
            game_bins[(name, num_rewards)].append((run_num, data))
    colors = {'_1': 'blue', '_2': 'red', '_3': 'orange', '_4': 'green'}
    for (name, num_rewards), data_list in game_bins.items():
        print(f'Plotting {name}')
        plt.clf()
        plt.hold(True)
        for (run_num, data) in data_list:
            print('run_num', run_num)
            x = [time for time, J in data['J_disentangled']][n-1:]
            mv = np.array([mv for _, mv in data['max_value_constraint']])
            indep = np.array([indep for _, indep in data['J_indep']])
            loss = indep - mv
            y = moving_average(loss, n=n)
            #y = moving_average([J for time, J in data['J_disentangled']], n=n)
            plt.plot(x, y, label=f'A{run_num}', color=colors[run_num])
        #plt.legend()
        #plt.title('Disentanglement Score')
        plt.xlabel('Timesteps')
        plt.ylabel('Disentanglement Score')
        plt.hold(False)
        plt.savefig(os.path.join(path, f'{name}_{num_rewards}reward.pdf'))

def make_meta_controller_plots(meta_runs, baseline_runs, n=20, y_range=None):
    path = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/meta_plots'
    all_keys = set(list(meta_runs.keys()) + list(baseline_runs.keys()))
    for game in all_keys:
        plt.clf()
        plt.hold(True)
        # baselines
        all_ys = []
        for data in baseline_runs[game]:
            print(len(data))
            print(len(data['cum_reward']))
            x = [time for time, J in data['cum_reward']][n-1:]
            y = moving_average([J for time, J in data['cum_reward']],n=n)
            print('shapes', len(data['cum_reward']), np.shape(y))
            all_ys.append(y)
        min_len = min([len(y) for y in all_ys])
        all_ys = [y[-min_len:] for y in all_ys]

        mean = np.mean(all_ys, axis=0)
        err = np.std(all_ys, axis=0)
        plt.plot(x[-min_len:], mean, color='blue', label=f'baseline')
        if y_range is not None:
            plt.ylim(y_range[0], y_range[1])
        plt.ylim()
        plt.fill_between(x[-min_len:], mean-err, mean+err, color='blue', alpha=0.5)

        color_mapping = {'2': 'red', '3': 'green', '4': 'orange'}
        for reward, data_runs in meta_runs[game].items():
            all_ys = []
            for data in data_runs:
                x = [time for time, J in data['cum_reward']][n-1:]
                y = moving_average([J for time, J in data['cum_reward']], n=n)
                all_ys.append(y)
                #plt.plot(x, y, color=color_mapping[reward], label=f'{reward} rewards')
            print(np.shape(all_ys))
            err = np.std(all_ys, axis=0)
            mean = np.mean(all_ys, axis=0)
            plt.plot(x, mean, color=color_mapping[reward], label=f'{reward} rewards')
            plt.fill_between(x, mean-err, mean+err, color=color_mapping[reward], alpha=0.5)
        plt.legend()
        plt.hold(False)
        plt.savefig(os.path.join(path, f'{game}_meta.pdf'))




if __name__ == '__main__':
    #data = cut_down_data(['J_disentangled', 'J_indep', 'J_nontrivial', 'max_value_constraint', 'time'], dont_repeat_work=False)
    #data = cut_down_meta_data()
    #data = cut_down_baseline_data()
    #meta_runs, baseline_runs = load_metacontroller_and_baseline_data('sokoban')
    #make_meta_controller_plots(meta_runs, baseline_runs, y_range=[30, 40])
    merged_data = load_and_merge_data(filter_regex=r'^.*?sokoban\_4reward.*?\_[234]$')
    #merged_data = load_and_merge_data()
    make_J_disentangled_plots(merged_data)
