import os
import re
import sys
import subprocess
# ~/minimal_q_learning/new_runs/
path_to_runs = sys.argv[1]
match_regex = sys.argv[2]
run_folders = [x for x in os.listdir(path_to_runs) if re.match(match_regex, x)]
# build the data-structure to fold restored runs onto their originals.
command_list = []
hostname = subprocess.check_output('hostname').decode('utf-8').strip()
for run_folder in run_folders:
    destination_path = '.'
    ssh_path = f'crgrimm@{hostname}.eecs.umich.edu:'
    source_path = os.path.join(path_to_runs, '.', run_folder)
    source_path = ssh_path+source_path

    command = f'rsync -r --relative {source_path} {destination_path}'
    command_list.append(command)
command = '; '.join(command_list)
print(command)


