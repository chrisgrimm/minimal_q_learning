import os, re, tensorflow as tf
import pickle, numpy as np, tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

def load_mapping(command_file='/Users/chris/projects/q_learning/result_visualizations/new_commands'):
    with open(command_file, 'r') as f:
        text = f.read()
    paths = [os.path.split(x[:-1]) for x in re.findall(r'\/home.*?[\,\}]', text.replace('\n',''))]
    paths = [(path.split('/')[-1], name) for (path, name) in paths]
    paths = {event_name: run_name for (run_name, event_name) in paths}
    return paths

def load_tb_data(file_path, fields, iter=-1, orig_offset=10000):
    # infer the iteration of the files.
    if iter == -1:
        mean_iter = get_mean_iter(file_path, 'cum_reward')
        #print(f'Got mean_iter {mean_iter}')
        if mean_iter > 10000:
            iter = 10000
        else:
            iter = 1000
    data = {f: [] for f in fields}
    fields_set = set(fields)
    for i, s in enumerate(tf.train.summary_iterator(file_path)):
        #print(i, s.step, )
        for v in s.summary.value:
            if v.tag in fields_set:
                #if v.tag == 'cum_reward':
                #    print(i, s.step)
                data[v.tag].append((s.step * iter + orig_offset, v.simple_value))
    return data

def get_mean_iter(file_path, field):
    num_to_track = 10
    i_list = []
    for i, s in enumerate(tf.train.summary_iterator(file_path)):
        for v in s.summary.value:
            #print(v.tag)
            if v.tag == field:
                i_list.append(i)
        if len(i_list) >= num_to_track:
            break
    i_list = np.array(i_list)
    mean_diff = np.mean(i_list[1:] - i_list[:-1])
    return mean_diff


def cut_down_meta_single_file(file_path, cut_down_path, dir):
    data = load_tb_data(file_path, ['cum_reward', 'time'])
    with open(os.path.join(cut_down_path, dir), 'wb') as f:
        pickle.dump(data, f)
    print(f'Finished {file_path} {dir}!')

def cut_down_meta(source_dir, dest_dir, regex):
    dirs = os.listdir(source_dir)
    print(f'Found {cpu_count()} devices... Creating pool.')
    pool = Pool(processes=cpu_count())
    for dir in dirs:
        if os.path.isfile(os.path.join(dest_dir, dir)):
            print(f'Found existing file {os.path.join(dest_dir, dir)}. Skipping.')
            continue
        #match = re.match(r'^meta\_(.+?)\_(\d)reward\_10mult\_(\d)$', dir)
        match = re.match(regex, dir)
        if not match:
            continue
        #(game, reward_partitions, run_num) = match.groups()
        #print(game, reward_partitions, run_num)
        print(dir)

        # get the events file
        event_files = [x for x in os.listdir(os.path.join(source_dir, dir)) if 'events' in x]
        if len(event_files) != 1:
            error = f'Found {len(event_files)} event files in {dir}. Skipping.'
            print(error)
            continue
        event_file = event_files[0]
        file_path = os.path.join(source_dir, dir, event_file)
        #args_list.append((file_path, cut_down_path, dir))
        pool.apply_async(cut_down_meta_single_file, args=(file_path, dest_dir, dir))
        #data = load_tb_data(file_path, ['cum_reward', 'time'])
        #with open(os.path.join(cut_down_path, dir), 'wb') as f:
        #    pickle.dump(data, f)
    pool.close()
    pool.join()


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

def cut_down_data_new(fields, base_path, dest_path, regex, dont_repeat_work=True):
    pool = Pool(processes=cpu_count())
    files = [x for x in os.listdir(base_path) 
             if re.match(regex, x) and os.path.isdir(os.path.join(base_path, x))]
    for f in files:
        run_path = os.path.join(base_path, f)
        event_files = [x for x in os.listdir(run_path) if x.startswith('events')]
        if len(event_files) != 1:
            raise Exception(f'Found {len(event_files)} event files in {run_path}.')
        event_file = event_files[0]
        if os.path.isdir(os.path.join(dest_path, f)):
            print(f'skipping {f}') 
            continue
        pool.apply_async(load_data_thread, args=(run_path, event_file, f, fields, dest_path))
    pool.close()
    pool.join()


def load_data_thread(run_path, event_file, name, fields, dest_path):
    print(f'Starting {name}!')
    data = load_tb_data(os.path.join(run_path, event_file), fields)
    with open(os.path.join(dest_path, name), 'wb') as f:
        pickle.dump(data, f)
    print(f'Finished {name}!')

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



def load_cutdown_data(cut_down_path, filter_regex=None):
    names = os.listdir(cut_down_path)
    all_data = {}
    for name in tqdm.tqdm(names):
        print(name)
        if filter_regex is None or re.match(filter_regex, name):
            with open(os.path.join(cut_down_path, name), 'rb') as f:
                data = replace_step_with_time(pickle.load(f))
            all_data[name] = data
    return all_data

def load_metacontroller_and_baseline_data(baselines, baseline_regex, meta, meta_regex, baseline_name, meta_name):
    meta_runs = dict()
    baseline_runs = dict()
    for name in tqdm.tqdm(os.listdir(baselines)):
        match = re.match(baseline_regex, name)
        if not match:
            print(f'{name} didnt match. Skipping...')
            continue
        (game, run_num) = match.groups()
        if baseline_name != game:
            print(f'{game} did not match baseline_name {baseline_name}. Skipping...')
            continue
        with open(os.path.join(baselines, name), 'rb') as f:
            data = pickle.load(f)
            print(name, len(data['cum_reward']))
            #data = replace_step_with_time(data)
            print(data.keys())
        if game in baseline_runs:
            baseline_runs[game].append(data)
        else:
            baseline_runs[game] = [data]
    for name in tqdm.tqdm(os.listdir(meta)):
        match = re.match(meta_regex, name)
        if not match:
            print(f'{name} didnt match. Skipping...')
            continue
        (game, num_rewards, run_num) = match.groups()
        if meta_name != game:
            print(f'{game} did not match meta_name {meta_name}. Skipping...')
            continue
        with open(os.path.join(meta, name), 'rb') as f:
            data = pickle.load(f)
            #data = replace_step_with_time(pickle.load(f))
            print(data.keys())

        if game in meta_runs:
            if num_rewards in meta_runs[game]:
                meta_runs[game][num_rewards].append(data)
            else:
                meta_runs[game][num_rewards] = [data]
        else:
            meta_runs[game] = {num_rewards: [data]}
    return meta_runs, baseline_runs

def load_no_top_data(single_game=None):
    dir = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/cut_down_no_top'
    meta_runs = dict()
    baseline_runs = dict()
    for name in tqdm.tqdm(os.listdir(dir)):
        match = re.match(r'^baseline\_(.+?)\_.+?(\d)$', name)
        if not match:
            print(f'{name} didnt match. Skipping...')
            continue
        (game, run_num) = match.groups()
        if single_game is not None and single_game != game:
            continue
        with open(os.path.join(dir, name), 'rb') as f:
            data = replace_step_with_time(pickle.load(f))
            print(data.keys())
        if game in baseline_runs:
            baseline_runs[game].append(data)
        else:
            baseline_runs[game] = [data]
    for name in tqdm.tqdm(os.listdir(dir)):
        match = re.match(r'^(.+?)\_(\d)reward.+?(\d)$', name)
        if not match or name.startswith('baseline'):
            print(f'{name} didnt match. Skipping...')
            continue
        (game, num_rewards, run_num) = match.groups()
        if single_game is not None and single_game != game:
            continue
        with open(os.path.join(dir, name), 'rb') as f:
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



def load_and_merge_data(cut_down_path, filter_regex=None):
    data = load_cutdown_data(cut_down_path, filter_regex=filter_regex)
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


def make_J_disentangled_plots(path, merged_data, n=2000):
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
        plt.legend()
        #plt.title('Disentanglement Score')
        plt.xlabel('Timesteps')
        plt.ylabel('Disentanglement Score')
        plt.hold(False)
        plt.savefig(os.path.join(path, f'{name}_{num_rewards}reward.pdf'))

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def make_plot(curve_sets, colors, names, output_path):
    plt.clf()
    resolution = 10000
    for curve_set, color, name in zip(curve_sets, colors, names):
        all_ys = []
        for curve in curve_set:
            x = [time for time, J in curve['cum_reward'] if time % resolution == 0][1:]
            y = smooth([J for time, J in curve['cum_reward'] if time % resolution == 0][1:], weight=0.95)
            print(len(x), len(y))
            all_ys.append(y)
        min_len = min([len(y) for y in all_ys])
        all_ys = [y[:min_len] for y in all_ys]
        mean = np.mean(all_ys)
        plt.plot(x[:min_len], mean, color=color, label=name)
    plt.legend()
    plt.savefig(output_path)

def make_repeat_plots(run_dir, dest_dir):
    games = ['assault', 'pacman', 'seaquest']
    reward_numbers = [2, 3, 5, 8]
    repeat_numbers = [1,2,4,8]
    trials = [1,2,3,4]
    for mode in ['', 'icf_']:
        for game in games:
            for reward_number in reward_numbers:
                plot_name = f'{mode}{game}_{reward_number}reward_plot.pdf'
                print(f'making {plot_name}...')
                curve_sets = []
                names = []
                colors = ['red', 'blue', 'green', 'orange']
                for repeat_number in repeat_numbers:
                    curve_set = []
                    rd_run_name = f'{mode}{game}_{reward_number}reward_{repeat_number}repeat'
                    for trial in trials:
                        full_name = rd_run_name+f'_{trial}'
                        with open(os.path.join(run_dir, full_name), 'rb') as f:
                            data = pickle.load(f)
                        curve_set.append(data)
                    curve_sets.append(curve_set)
                    names.append(f'{repeat_number} Repeats')
                make_plot(curve_sets, colors, names, os.path.join(dest_dir, plot_name))



def make_meta_controller_plots(path, meta_runs, baseline_runs, n=1, y_range=None, meta_to_baseline_mapping=None):
    all_keys = set(list(meta_runs.keys()))
    resolution = 10000
    for game in all_keys:
        plt.clf()
        plt.hold(True)
        # baselines
        all_ys = []
        bl_game = game if meta_to_baseline_mapping is None else meta_to_baseline_mapping[game]
        for data in baseline_runs[bl_game]:
            print(len(data))
            print(len(data['cum_reward']))
            x = [time for time, J in data['cum_reward'] if time % resolution == 0][1:]
            y = smooth([J for time, J in data['cum_reward'] if time % resolution == 0][1:], weight=0.95)
            print('shapes', len(data['cum_reward']), np.shape(y))
            all_ys.append(y)

        min_len = min([len(y) for y in all_ys])
        all_ys = [y[:min_len] for y in all_ys]
        
        mean = np.mean(all_ys, axis=0)
        err = np.std(all_ys, axis=0)
        plt.plot(x[:min_len], mean, color='blue', label=f'baseline')
        if y_range is not None:
            plt.ylim(y_range[0], y_range[1])
        plt.ylim()
        plt.fill_between(x[:min_len], mean-err, mean+err, color='blue', alpha=0.5)

        color_mapping = {'2': 'red', '3': 'green', '4': 'orange', '5': 'purple', '8': 'orange'}
        for reward, data_runs in meta_runs[game].items():
            all_ys = []
            for i, data in enumerate(data_runs):
                x = [time for time, J in data['cum_reward'] if time % resolution == 0][1:]
                y = smooth([J for time, J in data['cum_reward'] if time % resolution == 0][1:], weight=0.95)
                all_ys.append(y)
                print('meta_run', i, reward, np.shape(y))
                #plt.plot(x, y, color=color_mapping[reward], label=f'{reward} rewards')
            print(np.shape(all_ys))
            min_len = min([len(y) for y in all_ys])
            x = x[:min_len]
            all_ys = [y[:min_len] for y in all_ys]
            err = np.std(all_ys, axis=0)
            mean = np.mean(all_ys, axis=0)
            plt.plot(x, mean, color=color_mapping[reward], label=f'{reward} rewards')
            plt.xlabel('Timesteps')
            plt.ylabel('Score')
            plt.fill_between(x, mean-err, mean+err, color=color_mapping[reward], alpha=0.5)
        plt.legend()
        plt.hold(False)
        plt.savefig(os.path.join(path, f'{game}_meta.pdf'))




if __name__ == '__main__':
    #path_assault = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/new_baselines2/baseline_assault_2/events.out.tfevents.1542562453.rldl11'
    #path_breakout = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/new_baselines2/baseline_breakout_1/events.out.tfevents.1542677889.rldl11'
    #path_pacman = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/new_baselines2/baseline_pacman_1/events.out.tfevents.1542562393.rldl11'
    #path_seaquest = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/new_baselines2/baseline_seaquest_1/events.out.tfevents.1542562406.rldl11'
    #load_tb_data(path_pacman, ['time', 'cum_reward'])
#def cut_down_data_new(fields, base_path, dest_path, regex, dont_repeat_work=True):
    #cut_down_data_new(['J_disentangled', 'J_indep', 'J_nontrivial', 'max_value_constraint', 'time'],
    #                  '/home/crgrimm/minimal_q_learning/ALL_DATA/reward_learning_runs/',
    #                  '/home/crgrimm/minimal_q_learning/ALL_DATA/cut_down',
    #                  '^.+?[58]reward.+?$',
    #                  dont_repeat_work=False)
    #data = cut_down_data(['J_disentangled', 'J_indep', 'J_nontrivial', 'max_value_constraint', 'time'], dont_repeat_work=False)
    #data = cut_down_meta_data()
    #cut_down_meta(
    #  '/home/crgrimm/minimal_q_learning/ALL_DATA/meta_runs', 
    #  '/home/crgrimm/minimal_q_learning/ALL_DATA/cut_down_meta', 
    #  '^.+?[58]reward.+?')
    meta_runs, baseline_runs = load_metacontroller_and_baseline_data(
      '/home/crgrimm/minimal_q_learning/ALL_DATA/cut_down_baselines',
       r'^baseline\_(.+)\_(\d+)$',
      '/home/crgrimm/minimal_q_learning/ALL_DATA/cut_down_repeat',
       r'^meta\_(.+)\_(\d+)reward\_10mult\_(\d+)$',
      meta_name='seaquest',
      baseline_name='seaquest')
    make_meta_controller_plots(
      '/home/crgrimm/minimal_q_learning/ALL_DATA/plots/meta_plots',
      meta_runs, 
      baseline_runs)
      #meta_to_baseline_mapping={'assault_restricted_with_base': 'assault_restricted',
      #                          'pacman_restricted_with_base': 'pacman_restricted',
      #                          'seaquest_restricted_with_base': 'seaquest_restricted'})
    #merged_data = load_and_merge_data(filter_regex=r'^.*?sokoban\_4reward.*?\_[234]$')
    #merged_data = load_and_merge_data('/home/crgrimm/minimal_q_learning/ALL_DATA/cut_down')
    #make_J_disentangled_plots('/home/crgrimm/minimal_q_learning/ALL_DATA/disentangled_plots', merged_data)
