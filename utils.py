import os
import tensorflow as tf

class TBDataWriter:

    def __init__(self):
        self.global_steps = dict()
        self.summary_writer = None


    def setup(self, logdir):
        self.logdir = logdir


    def add_line(self, name, value):
        if self.summary_writer is None:
            self.summary_writer = tf.summary.FileWriter(self.logdir)
        global_step = self.global_steps.get(name, 0)
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        self.summary_writer.add_summary(summary, global_step=global_step)
        self.global_steps[name] = global_step + 1
        self.summary_writer.flush()


    def purge(self):
        for name in os.listdir(self.logdir):
            path = os.path.join(self.logdir, name)
            if os.path.isfile(path):
                os.remove(path)

LOG = TBDataWriter()


def build_directory_structure(base_dir, dir_structure):
    current_path = base_dir
    for target_key in dir_structure.keys():
        target_path = os.path.join(current_path, target_key)
        # make the dir if it doesnt exist.
        if not os.path.isdir(target_path):
            os.mkdir(target_path)
        # build downwards
        build_directory_structure(target_path, dir_structure[target_key])