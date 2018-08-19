import gym
import numpy as np
import os
import pickle
from random import choice

BASE_DIR = os.path.split(os.path.realpath(__file__))[0]

class SimpleAssault(object):

    def __init__(self):
        self.env = gym.make('Assault-v0')
        self.episode_length = 30
        self.step_num = 0
        with open(os.path.join(BASE_DIR, 'stored_states.pickle'), 'rb') as f:
            self.stored_states = pickle.load(f)

    def get_current_state(self):
        state = self.env.env.clone_full_state()
        return {'state': state, 'step_num': self.step_num}


    def restore_state(self, state):
        self.env.env.restore_full_state(state['state'])
        self.step_num = state['step_num']
        return self.env.env._get_obs()

    def step(self, a):
        self.step_num += 1
        a_processed = np.argmax(a, axis=0)
        sp, r, t, _ = self.env.step(a_processed)
        if r > 0:
            r = 1
            t = True
        else:
            r = 0
        if self.step_num >= self.episode_length:
            t = True
        return sp, r, t, {}

    def reset(self):
        self.step_num = 0
        self.env.reset()
        state = choice(self.stored_states)
        return self.restore_state({'state': state, 'step_num': 0})

    def render(self):
        self.env.render()


if __name__ == '__main__':
    state = None
    def save(env):
        global state
        state = env.get_current_state()

    def restore(env):
        global state
        env.restore_state(state)

    def reset(env):
        env.reset()

    def onehot(i, n):
        a = np.zeros(shape=[n], dtype=np.uint8)
        a[i] = 1
        return a

    env = SimpleAssault()
    n = env.env.action_space.n

    action_mapping = {'w': onehot(2, n),
                      'a': onehot(4, n),
                      'd': onehot(3, n),
                      '': onehot(0, n),
                      'save': save,
                      'restore': restore,
                      'reset': reset}

    s = env.reset()
    while True:
        try:
            action = action_mapping[input()]
            if callable(action):
                action(env)
                continue
        except KeyError:
            continue
        s, r, t, _ = env.step(action)
        print(r)
        if t:
            env.reset()
        env.render()






