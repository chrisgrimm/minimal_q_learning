from reward_network2 import ReparameterizedRewardNetwork
from envs.atari.atari_wrapper import PacmanWrapper
import numpy as np
env = PacmanWrapper()

reward_net = ReparameterizedRewardNetwork(env, 10, 0.0005, None, env.action_space.n, 'reward_net', num_channels=4)
reward_net.restore('reparam_runs/pacman_10reward_100x/weights', 'reward_net.ckpt')

def reset():
    s = env.reset()
    for _ in range(50):
        s, r, t, info = env.step(0)
    return s



policy_num = 0
s = reset()

while True:
    a = reward_net.get_state_actions([s])[policy_num][0]
    s, r, t, info = env.step(a)
    env.render()
    command = input('Command')
    if command == '':
        continue
    elif command == 'reset':
        s = reset()
        continue
    elif command.isnumeric():
        s = reset()
        try:
            policy_num = int(command)
        except ValueError:
            print('Policy number must be integer.')
        continue
    else:
        print(f'Invalid command: {command}.')


