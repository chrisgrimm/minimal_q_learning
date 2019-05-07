import numpy as np
import os
import sys
from itertools import product

gridsearch = {
    'corners-world-ideal-repr': ['[(1,0,1,0),(0,1,0,1)]',
                                 '[(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]'],
    'corners-world-size': ['15','50','100'],
    'nonvisual-num-hiddne-layers': ['0', '1'],
    'dqn-learning-rate': ['10**-4', '10**-3', '10**-2'],
    'dqn-target-update-frequency': ['1000', '100', '10', '1'],

    'run-dir': ['perfect_q_repr'],
    'corners-world-ideal': [''],
    'mode': ['CORNERS_WORLD'],
    'corners-world-reset-mode': ['deterministic'],
    'corners-world-task': ['1000'],
}

generated_parameters = {
    'gpu-num': [lambda i,name: (i % 3) + 1],
    'name': [lambda i,name: f'perfect_q::{name}']
}

preamble = lambda i,name: f'CUDA_VISIBLE_DEVICES={(i % 3) + 1} PYTHONPATH=~/ICF_copy/DL/ICF_simple:~/baselines:. python main_dqn.py'

# process gridsearch dictionary into argument list
all_args = []
for key, values in gridsearch.items():
    args = []
    for value in values:
        arg = (key, value)
        args.append(arg)
    all_args.append(args) # of lists of all possible arguments
    # [(pairs of all possible key values adn

def parse_value(value):
    return value.replace('(', '\(').replace(')', '\)')

for i, arg_list in enumerate(product(*all_args)):
    name = ','.join([f'{key}:{parse_value(value)}' for key, value in arg_list])
    args = [f'--{key}={parse_value(value)}' if len(value) > 0 else f'--{key}' for key, value in arg_list]
    generated_args = [f'--{key}={generated_parameters[key][0](i,name)}' for key in generated_parameters]
    arg_string = ' '.join(args + generated_args)
    command = preamble(i,name) + ' ' + arg_string
    print(command)