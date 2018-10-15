import tensorflow as tf
import numpy as np



for summary in tf.train.summary_iterator('/Users/chris/projects/q_learning/events.out.tfevents.1538757069.rldl11'):
    print(summary)