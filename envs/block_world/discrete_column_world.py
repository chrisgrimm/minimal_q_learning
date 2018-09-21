import numpy as np
import gym
from gym.spaces import Discrete, Box
import cv2
import itertools


class ColumnGame(object):

    def __init__(self, num_columns, num_steps):
        self.column_positions = np.random.randint(-num_steps, num_steps+1, size=[num_columns], dtype=np.int32)
        self.goal_positions = np.array([-num_steps if i % 2 ==0 else num_steps for i in range(num_columns)])
        self.block_width = 4
        self.spacing = 1
        self.num_columns = num_columns
        self.num_steps = num_steps
        self.action_space = Discrete(self.num_columns*3)
        self.observation_space = Box(low=0, high=255, shape=[64,64,3], dtype=np.uint8)
        self.max_episode_steps = 100
        self.step_num = 0

    def get_current_state(self):
        return {'column_positions': np.copy(self.column_positions),
                'step_num': self.step_num}

    def restore_state(self, state):
        self.column_positions = np.copy(state['column_positions'])
        self.step_num = state['step_num']
        return self.get_observation(self.column_positions)


    def step(self, action):
        # actions are in [0, num_columns * 3]
        column_target = action // 3
        column_action = action % 3
        diff = [1, 0, -1][column_action]
        self.column_positions[column_target] = np.clip(self.column_positions[column_target] + diff, -self.num_steps, self.num_steps)
        r = self.get_reward(self.column_positions)
        t = self.get_terminal()
        s = self.get_observation(self.column_positions)
        internal_s = np.copy(self.column_positions)
        if t:
            self.reset()
        return s, r, False, {'internal_terminal': t, 'vector_state': internal_s}



    def get_observation(self, column_positions):
        scale = self.spacing*(self.num_columns) + self.block_width*self.num_columns
        canvas = np.zeros([scale, scale], np.uint8)
        for column_num, height in enumerate(column_positions):
            norm_height = int(scale * (height + self.num_steps) / (2*self.num_steps + 1))
            pixel_x_pos = column_num * self.block_width + (column_num+1) * self.spacing
            canvas[scale-norm_height:scale, pixel_x_pos:pixel_x_pos+self.block_width] = 255
        return cv2.resize(np.tile(np.reshape(canvas, [scale, scale, 1]), [1,1,3]), (64, 64), interpolation=cv2.INTER_NEAREST)

    def get_reward(self, column_positions):
        return 1.0 if np.all(column_positions == self.goal_positions) else 0.0

    def get_terminal(self):
        return np.all(self.column_positions == self.goal_positions) or self.step_num >= self.max_episode_steps

    def produce_all_states(self):
        assert self.num_columns == 2
        mapping = {}
        for y in range(-self.num_steps, self.num_steps+1):
            for x in range(-self.num_steps, self.num_steps+1):
                column_positions = np.array([x, y])
                obs = self.get_observation(column_positions)
                r = self.get_reward(column_positions)
                mapping[(x,y)] = (obs, r)
        return mapping


    def reset(self):
        self.column_positions = np.random.randint(-self.num_steps, self.num_steps+1, size=[self.num_columns], dtype=np.int32)
        self.step_num = 0
        return self.get_observation(self.column_positions)


if __name__ == '__main__':
    env = ColumnGame(2, 2)
    s = env.reset()
    while True:
        a = np.random.randint(0, env.action_space.n)
        s, r, t, info = env.step(a)
        print(r, info['internal_terminal'], info['vector_state'])
        cv2.imshow('game', s)
        cv2.waitKey(1)