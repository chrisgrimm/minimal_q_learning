from gym.envs.atari import AtariEnv
import numpy as np
from gym.spaces import Box
import cv2

#env = gym.make('MsPacman-v0')
class AtariWrapper():

    def __init__(self, game):
        self.env = AtariEnv(game=game, obs_type='image', frameskip=5)
        self.action_space = self.env.action_space
        self.image_size = 64
        self.frame_buffer_len = 3
        self.frame_buffer = [np.zeros(shape=(64, 64, 3), dtype=np.uint8) for _ in range(self.frame_buffer_len)]
        self.observation_space = Box(0, 255, shape=(64,64,3*self.frame_buffer_len), dtype=np.uint8)

    def get_current_state(self):
        state = self.env.clone_full_state()
        copied_framebuffer = [np.copy(x) for x in self.frame_buffer]
        return {'state': state, 'frame_buffer': copied_framebuffer}


    def restore_state(self, state):
        self.env.restore_full_state(state['state'])
        self.frame_buffer = [np.copy(x) for x in state['frame_buffer']]
        return self.get_obs()

    def get_obs(self):
        return np.concatenate(self.frame_buffer, axis=2)

    def process_obs(self, obs):
        return cv2.resize(obs, (64,64), interpolation=cv2.INTER_AREA)

    def step(self, a):
        sp, r, t, info = self.env.step(a)
        r = 1 if r > 0 else 0
        self.frame_buffer = self.frame_buffer[1:] + [self.process_obs(sp)]
        obs = self.get_obs()
        if t:
            self.env.reset()
        return obs, r, False, info

    def reset(self):
        self.frame_buffer = [np.zeros(shape=(64, 64, 3), dtype=np.uint8) for _ in range(self.frame_buffer_len)]
        s = self.env.reset()
        self.frame_buffer = self.frame_buffer[1:] + [self.process_obs(s)]
        obs = self.get_obs()
        return obs

    def render(self):
        return self.env.render()

class PacmanWrapper(AtariWrapper):
    def __init__(self):
        super().__init__('ms_pacman')

class QBertWrapper(AtariWrapper):
    def __init__(self):
        super().__init__('qbert')

if __name__ == '__main__':
    env = QBertWrapper()
    print(env.action_space.n)
    action_mapping = {'w': 0,}

    s = env.reset()
    i = 0
    while True:
        sp, r, t, info = env.step(np.random.randint(0, env.action_space.n))
        cv2.imshow('pacman', cv2.resize(sp[:, :, 6:9], (400, 400)))
        cv2.waitKey(1)
        env.render()
        print(i, r, t, info)
        input()
        s = sp
        if t:
            s = env.reset()
        i += 1