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
  

base_dir, regex = sys.argv[1], sys.argv[2]
get_event_files(base_dir, regex)