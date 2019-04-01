import subprocess, os
import re
import argparse


def get_screen_names():
    process = subprocess.Popen(['screen', '-ls'], stdout=subprocess.PIPE)
    out, err = process.communicate()
    out = out.decode('utf-8')
    #err = err.decode('utf-8')
    regex = r'(\d+)\.(\S+).+?(\(Detached\)|\(Attached\))'
    groups = re.findall(regex, out)
    return [g[1] for g in groups]

def make_tensorboard_string(regex=None):
    base_dir = os.getcwd()
    if regex is None:
        names = get_screen_names()
        name_dir_pairs = [name+":"+os.path.join(base_dir, name) for name in names
                          if os.path.isdir(os.path.join(base_dir, name))]
    else:
        matched_names = [x for x in os.listdir(base_dir) if re.match(regex, x)]
        name_dir_pairs = [f'{name}:{os.path.join(base_dir,name)}' for name in matched_names]


    command = f'tensorboard --logdir={",".join(name_dir_pairs)}'
    print(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regex', type=str, default=None)
    args = parser.parse_args()
    make_tensorboard_string(args.regex)
