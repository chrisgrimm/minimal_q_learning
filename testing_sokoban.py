from envs.block_world.block_pushing_domain import BlockPushingDomain
from reward_network2 import ReparameterizedRewardNetwork

env = BlockPushingDomain(observation_mode='image', configuration='standard')

reward_net = ReparameterizedRewardNetwork(env, 2, 0.0005, None, env.action_space.n, 'reward_net')
reward_net.restore('reparam_runs/policy_DQN_2/weights', 'reward_net.ckpt')

action_map = {'w': 1,
              'a': 3,
              's': 0,
              'd': 2,
              '': 4}

policy = 0
s = env.reset()

while True:
    a = reward_net.get_state_actions([s])[policy][0]
    #a = input('Action: ')
    #if a not in action_map:
    #    print('not in action map!')
    #    continue
    #a = action_map[a]
    print(a)
    sp, r, t, info = env.step(a)
    print(reward_net.get_reward(s,a,sp))
    s = sp
    input('...')
    env.render()



