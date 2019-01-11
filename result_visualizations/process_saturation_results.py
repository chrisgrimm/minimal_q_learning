import numpy as np
import re
with open('saturation_results.txt', 'r') as f:
    data = f.readlines()

run_type_store = dict()

for line in data:
    [name, percent_sat] = line.strip('\n').split(' ')
    percent_sat = float(percent_sat)
    run_type = re.match(r'^(.+?)\_\d$', name).groups()[0]
    if run_type in run_type_store:
        run_type_store[run_type].append(percent_sat)
    else:
        run_type_store[run_type] = [percent_sat]

for run_type, percent_sat_list in run_type_store.items():
    assert len(percent_sat_list) == 4
    print(f'{run_type}: {np.mean(percent_sat_list)}')