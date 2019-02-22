from gym.envs.atari import AtariEnv
import numpy as np
from gym.spaces import Box
import cv2

#env = gym.make('MsPacman-v0')
class AtariWrapper():

    def __init__(self, game, remove_reward_mode=False):
        self.env = AtariEnv(game=game, obs_type='image', frameskip=5)
        self.action_space = self.env.action_space
        self.image_size = 64
        self.frame_buffer_len = 4
        self.num_channels = 1
        self.frame_buffer = [np.zeros(shape=(self.image_size, self.image_size, self.num_channels), dtype=np.uint8) for _ in range(self.frame_buffer_len)]
        self.observation_space = Box(0, 255, shape=(self.image_size,self.image_size,self.num_channels*self.frame_buffer_len), dtype=np.uint8)
        self.reward_range = (0, 1)
        self.metadata = self.env.metadata
        self.spec = self.env.spec
        self.max_steps = 10000
        self.step_counter = 0
        self.remove_reward_mode = remove_reward_mode

    def get_current_state(self):
        state = self.env.clone_full_state()
        #copied_framebuffer = [np.copy(x) for x in self.frame_buffer]
        return {'state': state}


    def restore_state(self, state):
        self.env.restore_full_state(state['state'])
        self.frame_buffer = [np.zeros(shape=(self.image_size, self.image_size, self.num_channels), dtype=np.uint8) for _ in range(self.frame_buffer_len)]
        # generate the framebuffer from the new state instead of storing it in a state-buffer.
        for _ in range(self.frame_buffer_len):
            self.step(np.random.randint(0, self.action_space.n))
        #self.frame_buffer = [np.copy(x) for x in state['frame_buffer']]
        return self.get_obs()

    def get_obs(self):
        return np.concatenate(self.frame_buffer, axis=2)

    def get_unprocessed_obs(self):
        return self.env._get_image()

    def process_obs(self, obs):
        obs = cv2.resize(obs, (self.image_size,self.image_size), interpolation=cv2.INTER_AREA)
        obs = np.reshape(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (self.image_size, self.image_size, self.num_channels))
        return obs

    def step(self, a):
        sp, r, t, info = self.env.step(a)
        if self.remove_reward_mode and self.remove_reward():
            r = 0
            info['r_env'] = 0
        else:
            info['r_env'] = r
            r = 1 if r > 0 else 0

        self.frame_buffer = self.frame_buffer[1:] + [self.process_obs(sp)]
        obs = self.get_obs()
        if t:
            self.env.reset()
        info['internal_terminal'] = t
        self.step_counter += 1
        if self.step_counter >= self.max_steps:
            info['internal_terminal'] = True
        return obs, r, False, info

    def reset(self):
        self.frame_buffer = [np.zeros(shape=(self.image_size, self.image_size, self.num_channels), dtype=np.uint8) for _ in range(self.frame_buffer_len)]
        s = self.env.reset()
        self.frame_buffer = self.frame_buffer[1:] + [self.process_obs(s)]
        obs = self.get_obs()
        self.step_counter = 0
        return obs

    def remove_reward(self):
        return False

    def render(self):
        return self.env.render()

class PacmanWrapper(AtariWrapper):
    def __init__(self, remove_reward_mode=False):
        super().__init__('ms_pacman', remove_reward_mode=remove_reward_mode)

    def remove_reward(self):
        ram = self.env.ale.getRAM()
        y_coord = ram[16]
        x_coord = ram[10]
        #print(x_coord, y_coord)
        return y_coord >= 98
        #y_coord = ram[0x]
        #print(x_coord, y_coord)

    def get_agent_position(self):
        ram = self.env.ale.getRAM()
        y_coord = ram[16]
        x_coord = ram[10]
        return x_coord, y_coord



class QBertWrapper(AtariWrapper):
    def __init__(self):
        super().__init__('qbert')

class AssaultWrapper(AtariWrapper):
    def __init__(self, remove_reward_mode=False):
        super().__init__('assault', remove_reward_mode=remove_reward_mode)
        self.last_known_position = 100
        self.last_known_shot_position = 100
        self.last_shot_status = 127

    def remove_reward(self):
        ram = self.env.ale.getRAM()
        # the position idx only appears to hold the position when the agent is moving
        position_idx = 5*18+4 - 1
        position_opt = ram[position_idx]
        # 60 means no new position data.
        # 0 means you just shot something.
        if position_opt not in [60, 0]:
            self.last_known_position = position_opt
        # no reward if last_known_position > 100
        #print(self.last_known_position)
        return self.last_known_position > 100

    def get_agent_position(self):
        ram = self.env.ale.getRAM()
        position_idx = 5*18+4 - 1
        position_opt = ram[position_idx]
        shot_height = ram[67]
        #print('RAM', ram)
        #print('shot height?', ram[67])
        if position_opt not in [60, 0, 1, 2]:
            self.last_known_position = position_opt
        if self.last_shot_status == 127 and shot_height != 127:
            #print('Fired Shot!')
            self.last_known_shot_position = self.last_known_position
        self.last_shot_status = shot_height
        return self.last_known_shot_position, 0



class BreakoutWrapper(AtariWrapper):
    def __init__(self):
        super().__init__('breakout')

class AlienWrapper(AtariWrapper):
    def __init__(self):
        super().__init__('alien')

class SeaquestWrapper(AtariWrapper):
    def __init__(self, remove_reward_mode=False):
        super().__init__('seaquest', remove_reward_mode=remove_reward_mode)

    def remove_reward(self):
        ram = self.env.ale.getRAM()
        y_idx = 5*18+8-1
        x_idx = 3*18 + 17 - 1
        #print('y', ram[y_idx], 'x', ram[x_idx])
        #print(ram)
        # only reward when in the bottom half of the env.
        return ram[y_idx] <= 65

    def get_agent_position(self):
        ram = self.env.ale.getRAM()
        y_idx = 5 * 18 + 8 - 1
        x_idx = 3 * 18 + 17 - 1
        return ram[x_idx], ram[y_idx]

if __name__ == '__main__':
    env = AssaultWrapper(remove_reward_mode=True)
    print(env.action_space.n)
    print(env.env.get_action_meanings())
    #action_mapping = {'w': 0,}
    action_set = {'w': 2, 'd': 3, 'a': 4, 's': 5, '': 0, ' ': 1}
    print('n', env.action_space.n)
    s = env.reset()
    i = 0
    while True:
        a = input()
        if a not in action_set:
            continue
        a = action_set[a]
        print('a', a)
        sp, r, t, info = env.step(a)
        if r == 1:
            print('Got reward!')
        #print(env.remove_reward())
        cv2.imshow('pacman', cv2.resize(sp[:, :, 6:9], (400, 400)))
        cv2.waitKey(1)
        env.render()
        image = env.get_unprocessed_obs()
        image = np.tile(np.reshape(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), list(image.shape[:2]) + [1]), [1,1,3])
        cv2.imwrite('./pacman_sample.png', image)
        #env.env.ale.saveScreenPNG(b'test_image2.png')

        print(i, r, t, info)
        input()
        s = sp
        if t:
            s = env.reset()
        i += 1