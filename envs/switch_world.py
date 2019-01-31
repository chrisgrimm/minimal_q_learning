import numpy as np
import cv2
from gym.spaces import Discrete, Box

class SwitchWorld(object):

    def __init__(self):
        self.w, self.h = 5, 5
        self.switch_positions = [(0,0), (4,4), (0,4), (4,0)]
        self.switch_states = [False, False, False, False]
        self.switch_color_off = (0, 255, 0)
        self.switch_color_on = (255, 0, 0)
        self.background_color = (255, 255, 255)
        self.agent_color = (0, 0, 0)

        self.agent_position = self.get_agent_position()
        self.reset()
        self.timer = 0
        self.observation_space = Box(0, 255, shape=[64, 64, 3], dtype=np.uint8)
        self.action_space = Discrete(4)
        self.action_mapping = {0: (0,1), 1: (1,0), 2: (0,-1), 3: (-1,0)}
        self.human_mapping = {'a': 3, 'd': 1, 'w': 2, 's': 0}

    def step(self, a):
        delta_x, delta_y = self.action_mapping[a]
        new_x_pos = np.clip(self.agent_position[0] + delta_x, 0, self.w-1)
        new_y_pos = np.clip(self.agent_position[1] + delta_y, 0, self.h-1)
        for i, switch_pos in enumerate(self.switch_positions):
            if (new_x_pos, new_y_pos) == switch_pos:
                self.switch_states[i] = True
        self.agent_position = new_x_pos, new_y_pos
        reward = 1 if all(self.switch_states) else 0
        self.timer += 1
        internal_terminal = False
        if self.timer >= 1000:
            internal_terminal = True
            self.reset()
        obs = self.get_observation()
        return obs, reward, False, {'internal_terminal': internal_terminal}


    def reset(self):
        self.switch_states = [False, False, False, False]
        self.agent_position = self.get_agent_position()
        self.timer = 0
        return self.get_observation()

    def get_current_state(self):
        state = (self.switch_states[:], self.agent_position, self.timer)
        #copied_framebuffer = [np.copy(x) for x in self.frame_buffer]
        return state


    def restore_state(self, state):
        switch_states, agent_position, timer = state
        self.switch_states = switch_states
        self.agent_position = agent_position
        self.timer = timer
        return self.get_observation()

    def get_observation(self):
        canvas = np.ones([self.h, self.w, 3], dtype=np.uint8) * np.reshape((255,255,255), [1,1,3])
        for pos, is_on in zip(self.switch_positions, self.switch_states):
            canvas[pos[1], pos[0], :] = self.switch_color_on if is_on else self.switch_color_off
        x, y = self.agent_position[0], self.agent_position[1]
        canvas[y, x, :] = canvas[y, x, :] // 2 + np.array(self.agent_color) // 2
        img = cv2.resize(canvas, (64,64), interpolation=cv2.INTER_NEAREST)
        return img

    def get_obs(self):
        return self.get_observation()

    def get_agent_position(self):
        while True:
            pos_x, pos_y = np.random.randint(0, self.w), np.random.randint(0, self.h)
            if (pos_x, pos_y) not in self.switch_positions:
                break
        return (pos_x, pos_y)

if __name__ == '__main__':
    env = SwitchWorld()
    while True:
        a = np.random.randint(0, 4)
        s, r, t, info = env.step(a)
        input('...')
