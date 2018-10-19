import os
import re
import sys
import subprocess
# ~/minimal_q_learning/new_runs/
path_to_runs = sys.argv[1]
run_folders = os.listdir(path_to_runs)
bins = {}
# build the data-structure to fold restored runs onto their originals.
for run in run_folders:
    match = re.match(r'^(restore\_)?(.+?)\_(\d)reward\_10mult(\_\d)?$', run)
    if not match:
        continue
    (restore, game, reward, run_num) = match.groups()
    if (game, reward, run_num) in bins:
        bins[(game, reward, run_num)].append(((restore, game, reward, run_num),run))
    else:
        bins[(game, reward,run_num)] = [((restore, game, reward, run_num), run)]

# select the appropriate run from each bin
bin_choices = {}
for (game, reward, run_num), bin in bins.items():
    assert len(bin) in [1,2]
    if len(bin) == 2:
        filter = [x for x in bin if x[0][0] is not None]
        assert len(filter) == 1
        bin_choice = filter[0][1]
    else:
        bin_choice = bin[0][1]
    bin_choices[(game, reward, run_num)] = bin_choice

# compute the rsync command for each choice.
hostname = subprocess.check_output('hostname').decode('utf-8').strip()
command_list = []
for (game, reward, run_num), run_name in bin_choices.items():
    destination_path = '.'
    ssh_path = f'crgrimm@{hostname}:'
    source_path = os.path.join(path_to_runs, '.', run_name, 'best_weights')
    source_path = ssh_path+source_path

    command = f'rsync -r --relative {source_path} {destination_path}'
    command_list.append(command)
command = '; '.join(command_list)
print(command)

