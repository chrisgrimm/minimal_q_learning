import subprocess, os
import re


def get_screen_names():
    process = subprocess.Popen(['screen', '-ls'], stdout=subprocess.PIPE)
    out, err = process.communicate()
    out = out.decode('utf-8')
    #err = err.decode('utf-8')
    print('out', out)
    print('err', err)
    regex = r'(\d+)\.(\S+).+?(\(Detached\)|\(Attached\))'
    groups = re.findall(regex, out)
    return [g[1] for g in groups]

def make_tensorboard_string():
    base_dir = os.getcwd()
    names = get_screen_names()
    name_dir_pairs = [name+":"+os.path.join(base_dir, name) for name in names
                      if os.path.isdir(os.path.join(base_dir, name))]
    command = f'tensorboard --logdir={",".join(name_dir_pairs)}'
    print(command)

if __name__ == '__main__':
    make_tensorboard_string()
