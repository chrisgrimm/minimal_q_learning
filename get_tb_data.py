import tensorflow as tf
import numpy as np



for summary in tf.train.summary_iterator('/Users/chris/projects/q_learning/new_runs/sokoban_2reward_10mult/events.out.tfevents.1538756539.rldl11'):
    print(summary)