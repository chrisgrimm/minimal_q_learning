import os, numpy as np, re, sys


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
    run_dirs = [x for x in os.listdir(base_dir)
                if re.match(regex, x)]
    command_list = []
    for run_dir in run_dirs:
        event_file = [x for x in os.listdir(os.path.join(base_dir, run_dir))
                      if 'event' in x]
        if len(event_file) != 1:
            raise Exception(f'Path: {os.path.join(base_dir, run_dir)} contains {len(event_file)} event files.')
        event_file = event_file[0]
        rel_path = os.path.join(base_dir, '.', run_dir, event_file)
        command = f'rsync -r --relative {rel_path} .'
        command_list.append(command)
    command = '; '.join(command_list)
    print(command)


base_dir, regex = sys.argv[1], sys.argv[2]
get_event_files_rsync(base_dir, regex)