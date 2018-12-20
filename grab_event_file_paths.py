import os, numpy as np, re, sys, subprocess


def get_event_files(base_dir, regex):
    run_dirs = [(x, os.path.join(base_dir, x)) for x in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, x))]
    event_paths = []
    for name, dir_path in run_dirs:
        if re.match(regex, name) is not None:
            event_files = [x for x in os.listdir(dir_path) if x.startswith('events')]
        assert len(event_files) == 1
        event_path = os.path.join(dir_path, event_files[0])
        event_paths.append(event_path)
    command = f'rsync -r crgrimm@rldl11.eecs.umich.edu:{{{",".join(event_paths)}}} .'
    print(command)

def get_event_files_rsync(base_dir, regex):
    hostname = subprocess.check_output('hostname').decode('utf-8').strip()
    ssh_path = f'crgrimm@{hostname}.eecs.umich.edu:'
    run_dirs = [x for x in os.listdir(base_dir)
                if re.match(regex, x)]
    command_list = []
    for run_dir in run_dirs:
        event_file = [x for x in os.listdir(os.path.join(base_dir, run_dir))
                      if 'event' in x]
        if len(event_file) != 1:
            raise Exception(f'Path: {os.path.join(base_dir, run_dir)} contains {len(event_file)} event files.')
        event_file = event_file[0]
        rel_path = ssh_path+os.path.join(base_dir, '.', run_dir, event_file)
        command = f'rsync -r --relative {rel_path} .'
        command_list.append(command)
    command = '; '.join(command_list)
    print(command)

def get_weight_files_rsync(base_dir, regex):
    hostname = subprocess.check_output('hostname').decode('utf-8').strip()
    ssh_path = f'crgrimm@{hostname}.eecs.umich.edu:'
    run_dirs = [x for x in os.listdir(base_dir)
                if re.match(regex, x)]
    command_list = []
    for run_dir in run_dirs:
        weights_path = os.path.join(base_dir, '.', run_dir, 'best_weights', '*')
        rel_path = ssh_path + weights_path
        command = f'rsync -r --relative {rel_path} .'
        command_list.append(command)
    command = '; '.join(command_list)
    print(command)

mode, base_dir, regex = sys.argv[1], sys.argv[2], sys.argv[3]
if mode == 'events':
    get_event_files_rsync(base_dir, regex)
elif mode == 'weights':
    get_weight_files_rsync(base_dir, regex)
else:
    raise Exception(f'Mode {mode} not in supported modes: "events", "weights".')
