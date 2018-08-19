import gym
import numpy as np
import pickle

state = None
all_states = []
def save(env):
    global state
    state = env.env.ale.clone_full_state()

def restore(env):
    global state
    env.env.restore_full_state(state)

def store_state(env):
    global all_states
    new_state = env.env.clone_full_state()
    all_states.append(new_state)
    with open('stored_states.pickle', 'wb') as f:
        pickle.dump(all_states, f)

def run_game():
    action_mapping = {'w': 2, 'a': 4, 'd': 3, '': 0,
                      'save': save, 'restore': restore, 'store': store_state}
    env = gym.make('Assault-v0')
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
        env.render()



with open('stored_states.pickle', 'rb') as f:
    state_list = pickle.load(f)
print(len(state_list))