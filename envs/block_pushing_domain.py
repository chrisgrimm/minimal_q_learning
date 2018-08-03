import numpy as np
import cv2
from gym.spaces import Discrete, Box


class BlockPushingDomain(object):

    def __init__(self):
        self.grid_size = 5
        self.block_size = 4

        self.agent_color = (255, 0, 0)
        self.block_color = (0, 0, 0)
        self.bg_color = (255, 255, 255)

        self.observation_mode = 'vector'
        assert self.observation_mode in ['vector', 'image', 'encoding']

        self.action_mapping = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0), 4: (0, 0)}

        self.goal_configuration = []
        # object positions are top-left positions.
        self.block_colors = [self.agent_color, self.block_color]
        self.num_objects = len(self.block_colors)
        self.top_right_position = (self.grid_size-1, 0)
        self.agent_index = 0
        self.obs_size = 2*self.num_objects
        self.max_timesteps = 1000
        self.timestep = 0
        self.action_space = Discrete(5)
        self.observation_space = Box(-1, 1, shape=[2*self.obs_size])
        self.reset()

    def get_all_states(self):
        states = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (x, y) == self.top_right_position:
                    states.append(self.get_observation([(0,0), self.top_right_position]))
                else:
                    states.append(self.get_observation([(x, y), self.top_right_position]))
        return states


    def get_current_state(self):
        return {'object_positions': self.object_positions[:],
                'timestep': self.timestep}

    def restore_state(self, state):
        self.object_positions = state['object_positions'][:]
        self.timestep = state['timestep']
        return self.get_observation()

    def get_nonintersecting_positions(self, num_positions, existing_positions=set()):
        positions = set()
        while len(positions) < num_positions:
            candidate = tuple(np.random.randint(0, self.grid_size, size=2))
            if candidate not in existing_positions:
                positions.add(candidate)
        return list(positions)

    def produce_image(self, object_positions):
        image_size = self.grid_size * self.block_size
        canvas = np.zeros(shape=[image_size, image_size, 3])
        canvas[:, :] = self.bg_color
        for pos, color in zip(object_positions, self.block_colors):
            x_pos, y_pos = self.block_size * pos[0], self.block_size * pos[1]
            canvas[y_pos:y_pos+self.block_size, x_pos:x_pos+self.block_size] = color
        return canvas

    def produce_vector(self, object_positions):
        return np.concatenate(object_positions, axis=0)


    def get_observation(self, object_positions=None):
        object_positions = self.object_positions if object_positions is None else object_positions

        if self.observation_mode == 'image':
            image = self.produce_image(object_positions)
            goal_image = self.produce_image(self.goal_configuration)
            return np.concatenate([image, goal_image], axis=2)
        elif self.observation_mode == 'encoding':
            raise NotImplemented
        elif self.observation_mode == 'vector':
            vector = self.produce_vector(object_positions)
            #goal_vector = self.produce_vector(self.goal_configuration)
            return np.concatenate([vector, np.zeros_like(vector)], axis=0) / self.grid_size
            #return vector

    def action_converter(self, raw_action):
        return np.argmax(raw_action)

    def out_of_bounds(self, pos):
        pos_x, pos_y = pos
        return not ((0 <= pos_x) and (pos_x < self.grid_size) and (0 <= pos_y) and (pos_y < self.grid_size))

    def can_move(self, pos, delta, index, all_objects, update_set):
        new_pos = tuple(np.array(pos) + np.array(delta))
        # return the index
        if self.out_of_bounds(new_pos):
            return False
        can_move = True
        for obj_index, obj_pos in enumerate(all_objects):
            if index == obj_index:
                continue
            # recursively handle collisions, assumes only 1 collision is possible for each considered object.
            #print(new_pos, obj_pos)
            if new_pos == obj_pos:
                updated_all_objects = all_objects[:]
                updated_all_objects[index] = new_pos
                can_move = self.can_move(obj_pos, delta, obj_index, updated_all_objects, update_set)
                if not can_move:
                    break
        if can_move:
            update_set.add((index, new_pos))
        return can_move


    def perform_action(self, action):
        self.timestep += 1
        delta = self.action_mapping[action]
        agent_pos = self.object_positions[self.agent_index]
        update_set = set()
        can_move = self.can_move(agent_pos, delta, self.agent_index, self.object_positions, update_set)
        if can_move:
            for index, pos in update_set:
                self.object_positions[index] = pos


    def get_reward(self, obs, goal=None):

        [agent_x, agent_y] = obs[:2]
        agent_x = int(agent_x * self.grid_size)
        agent_y = int(agent_y * self.grid_size)
        reward = 0.0 if (agent_x == (self.grid_size-1)) or (agent_y == 0) else -1.0
        #R1 = 0.0  if agent_x == (self.grid_size-1) else
        #R2 = 0.0 if agent_y == 0 else -1.0
        return reward
        #assert self.observation_mode in ['vector', 'encoding']
        #return 10.0 if self.get_terminal(obs, goal) else -0.1
        #raise NotImplemented


    def get_terminal(self, obs, goal=None):
        return self.timestep >= self.max_timesteps
        # assert self.observation_mode in ['vector', 'encoding']
        # obs_part = obs[:self.obs_size]
        # goal_part = obs[self.obs_size:] if goal is None else goal
        # at_goal = np.max(np.abs(obs_part - goal_part)) < 0.001
        # return at_goal
        #raise NotImplemented

    def step(self, raw_action):
        old_obs = self.get_observation()
        #action = self.action_converter(raw_action)
        action = raw_action
        self.perform_action(action)
        new_obs = self.get_observation()
        reward = self.get_reward(new_obs)
        terminal = self.get_terminal(new_obs)
        # TODO use old_obs for hindsight.
        return new_obs, reward, terminal, {}

    def render(self):
        image = self.produce_image(self.object_positions)
        cv2.imshow('game', image)
        cv2.waitKey(1)


    def reset(self):
        self.object_positions = self.get_nonintersecting_positions(self.num_objects - 1, existing_positions=set([self.top_right_position])) + [self.top_right_position]
        self.timestep = 0
        return self.get_observation()
        #self.goal_configuration = self.get_nonintersecting_positions(self.num_objects)


def onehot(i, n):
    vec = np.zeros(shape=[n])
    vec[i] = 1
    return vec


action_mapping = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0), 4: (0, 0)}

if __name__ == '__main__':
    env = BlockPushingDomain()
    s = env.reset()
    env.render()
    action_map = {'w': onehot(1, 5),
                  'a': onehot(3, 5),
                  's': onehot(0, 5),
                  'd': onehot(2, 5),
                  ' ': onehot(4, 5)}
    while True:
        action_key = input('Action: ')
        if action_key == 'r':
            env.reset()
            env.render()
            continue
        if action_key not in action_map:
            continue
        action = action_map[action_key]
        print(action)
        s, reward, terminal, _ = env.step(np.argmax(action))
        env.render()
        print(f'reward {reward}, terminal {terminal}')
        if terminal:
            env.reset()
            env.render()

