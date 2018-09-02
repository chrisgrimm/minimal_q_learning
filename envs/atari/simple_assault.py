import gym
import numpy as np
import os
import pickle
import cv2
from random import choice
from gym.spaces import Box

BASE_DIR = os.path.split(os.path.realpath(__file__))[0]

class SimpleAssault(object):

    def __init__(self, initial_states_file='stored_states.pickle'):
        self.env = gym.make('Assault-v0')
        self.episode_length = 100
        self.frame_buffer_len = 3
        self.step_num = 0
        self.use_initial_states = initial_states_file is not None
        if self.use_initial_states:
            with open(os.path.join(BASE_DIR, initial_states_file), 'rb') as f:
                self.stored_states = pickle.load(f)
        self.action_space = self.env.action_space
        self.image_size = 64
        self.observation_space = Box(0, 255, shape=[self.image_size, self.image_size, 3*self.frame_buffer_len])

    def get_current_state(self):
        state = self.env.env.clone_full_state()
        copied_framebuffer = [np.copy(x) for x in self.frame_buffer]
        return {'state': state, 'step_num': self.step_num, 'frame_buffer': copied_framebuffer}


    def restore_state(self, state):
        self.env.env.restore_full_state(state['state'])
        self.step_num = state['step_num']
        self.frame_buffer = [np.copy(x) for x in state['frame_buffer']]
        return self.get_obs()

    def get_raw_obs(self):
        obs = self.env.env._get_obs()
        return cv2.resize(obs, (self.image_size, self.image_size))

    def get_obs(self):
        return np.concatenate([np.copy(x) for x in self.frame_buffer], axis=2) # [32 x 32 x 3*frame_buffer_len]

    def step(self, a):
        self.step_num += 1
        _, r, t, _ = self.env.step(a)
        sp = self.get_raw_obs()
        ship_status = self.determine_ship_states()
        self.frame_buffer = self.frame_buffer[1:] + [sp]
        if r > 0:
            # dont give reward for the last ship.
            if ship_status[2]:
                r = 1
                t = True
            else:
                r = 0
                t = False
        else:
            r = 0
        if self.step_num >= self.episode_length:
            t = True
        obs = self.get_obs()
        info = {
            'ship_status': self.determine_ship_states(),
            'internal_terminal': t
        }
        if t:
            self.reset()

        return obs, r, False, info

    def reset(self):
        if not self.use_initial_states:
            self.frame_buffer = [np.zeros(shape=[self.image_size, self.image_size, 3]) for _ in range(self.frame_buffer_len)]
            self.step_num = 0
            self.env.reset()
            self.frame_buffer = self.frame_buffer[1:] + [self.get_raw_obs()]
            return self.get_obs()
        else:
            self.env.reset()
            state = choice(self.stored_states)
            state['step_num'] = 0
            return self.restore_state(state)

        #state = choice(self.stored_states)
        #s = self.restore_state({'state': state, 'step_num': 0})
        #self.frame_buffer


    def render(self):
        self.env.render()

    def determine_ship_states(self):
        ram = self.env.env.ale.getRAM()
        is_ship_alive_array = [x == 192 for x in ram[54:57]]
        return is_ship_alive_array



class RAMTracker(object):

    def __init__(self, ram_size):
        self.ram_size = ram_size
        self.ram_no_change = []
        self.ram_change = []

    def add_no_change(self, pre_ram, ram):
        self.ram_no_change.append( (pre_ram.tolist(), ram.tolist()) )

    def add_change(self, pre_ram, ram):
        self.ram_change.append( (pre_ram.tolist(), ram.tolist()) )

    def evaluate(self):
        # want the set of things that only change when a space-ship blows up.
        candidate_array = [True for _ in range(self.ram_size)]
        for i in range(self.ram_size):

            for pre_ram, ram in self.ram_no_change:
                if pre_ram[i] != ram[i]:
                    candidate_array[i] = False
            for pre_ram, ram in self.ram_change:
                if pre_ram[i] == ram[i]:
                    candidate_array[i] = False

            #print(list(zip(self.ram_no_change)))
            # check how many values ram state i takes on in the no_change categories.
            #ram_i_values_no_change = set(list(zip(*self.ram_no_change))[i])

            #ram_i_values_change = set(list(zip(*self.ram_change))[i])
            #print(len(ram_i_values_change), len(ram_i_values_no_change))
            #print(ram_i_values_no_change)
            #if len(ram_i_values_no_change) == 1 and len(ram_i_values_change) > 1:
            #    candidate_array[i] = True
        print('Candidate indices:', [i for i, bool in enumerate(candidate_array) if bool])








if __name__ == '__main__':
    all_states = []
    state = None

    def save(env):
        global state
        state = env.get_current_state()

    def restore(env):
        global state
        env.restore_state(state)

    def reset(env):
        env.reset()

    def store_state(env):
        global all_states
        new_state = env.get_current_state()
        all_states.append(new_state)
        with open('stored_states_64.pickle', 'wb') as f:
            pickle.dump(all_states, f)

    def onehot(i, n):
        a = np.zeros(shape=[n], dtype=np.uint8)
        a[i] = 1
        return a

    env = SimpleAssault(initial_states_file='stored_states_64.pickle')
    ram_tracker = RAMTracker(env.env.env.ale.getRAMSize())
    n = env.env.action_space.n

    action_mapping = {'w': 2,
                      'a': 4,
                      'd': 3,
                      '': 0,
                      'save': save,
                      'restore': restore,
                      'store': store_state,
                      'reset': reset}

    s = env.reset()
    while True:
        pre_ram = env.env.env.ale.getRAM()
        try:
            action = action_mapping[input()]
            if callable(action):
                action(env)
                continue
        except KeyError:
            continue
        s, r, t, info = env.step(action)
        cv2.imshow('game', cv2.resize(s[:, :, 6:9], (400,400)))
        cv2.waitKey(1)

        ram = env.env.env.ale.getRAM()
        print(f'R({r}) T({t}) RAM({info["ship_status"]})')

        if r == 0:
            pass
            #ram_tracker.add_no_change(pre_ram, ram)
        else:
            pass
            #ram_tracker.add_change(pre_ram, ram)
            #ram_tracker.evaluate()
            #print(ram[72:75])
        #print(r, env.determine_ship_states())

        if t:
            print(t)
            env.reset()
        #env.render()






