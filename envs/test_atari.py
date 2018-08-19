import gym
import numpy as np

state = None

def save(env):
    global state
    state = env.env.ale.cloneSystemState()

def restore(env):
    global state
    env.env.ale.restoreSystemState(state)

action_mapping = {'w': 2, 'a': 4, 'd': 3, '': 0,
                  'save': save, 'restore': restore}
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

