import numpy as np
import sys
import os, re
import subprocess

path_to_runs = sys.argv[1]
match_regex = sys.argv[2]
run_folders = [x for x in os.listdir(path_to_runs) if re.match(match_regex, x)]
# build the data-structure to fold restored runs onto their originals.
command_list = []
hostname = subprocess.check_output('hostname').decode('utf-8').strip()
num_to_take = 5
for run_folder in run_folders:
    destination_path = '.'
    ssh_path = f'crgrimm@{hostname}.eecs.umich.edu:'
    source_path = os.path.join(path_to_runs, '.', run_folder)
    source_path = ssh_path+source_path
    files = [x for x in os.listdir(os.path.join(path_to_runs, run_folder, 'images'))]
    stat_files = []
    matrix_files = []
    behavior_files = []
    image_files = []
    for x in files:
        print(x)
        match = re.match(r'^policy\_vis\_(\d+)(.*?)\..+?$', x)
        if not match:
            continue
        (iteration_num, file_type) = match.groups()
        print(iteration_num, file_type)
        print('file_type', file_type)
        iteration_num = int(iteration_num)
        if file_type == '_statistics':
            stat_files.append((iteration_num, x))
        elif file_type == '_value_matrix':
            matrix_files.append((iteration_num, x))
        elif file_type == '_behavior_file':
            behavior_files.append((iteration_num, x))
        elif file_type == '':
            image_files.append((iteration_num, x))
        else:
            raise Exception(f'Unrecognized filetype {file_type} on iteration {iteration_num}')
    grabber = lambda file_list: [y for (x,y) in sorted(file_list, key=lambda x: x[0])[-num_to_take:]]
    files_to_grab = []
    files_to_grab += grabber(stat_files)
    files_to_grab += grabber(matrix_files)
    files_to_grab += grabber(behavior_files)
    files_to_grab += grabber(image_files)
    relative_paths = [os.path.join(path_to_runs, '.', run_folder, 'images', x) for x in files_to_grab]

    for source_path in relative_paths:
        command = f'rsync -r --relative {ssh_path+source_path} {destination_path}'
        command_list.append(command)
command = '; '.join(command_list)
print(command)