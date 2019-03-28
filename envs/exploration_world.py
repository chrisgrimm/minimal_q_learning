from gym import Env
from gym.spaces import Box, Discrete
import numpy as np
from itertools import count
import cv2


class ExplorationWorld(Env):

    def __init__(self, world_size=100, reward_mode='EXPLORE'):
        self.agent = (0,0)
        self.inner_walls = [
            (2,1), (2,2), (2,-1), (2,-2),
            (1,2), (1,-2),
            (-1,2), (-1,-2),
            (-2,-2), (-2,-1), (-2,1), (-2,2)
        ]
        self.inner_walls = [
            (-1,3), (0,3), (1,3),
            (-1,-3), (0,-3), (1,-3),
            (3,1), (3,0), (3,-1),
            (-3,1), (-3,0), (-3,-1)
        ]
        self.world_size = world_size # the size of the world in any direction outside of the inner walls.

        self.specified_walls = set(self.inner_walls)
        self.generated_wall_rules = [
            lambda p: ((p[0] == 0 and np.abs(p[1]) > 3) or (p[1] == 0 and np.abs(p[0]) > 3)),
            #lambda p: ((np.abs(p[0]) > 2) and (np.abs(p[0]) == np.abs(p[1]))), # builds the diagonal walls.
            #lambda p: ((np.abs(p[0]) > self.world_size + 2) or (np.abs(p[1]) > self.world_size + 2)), # builds the outer walls
        ]

        self.action_mapping = {
            0: (1,0),
            1: (-1,0),
            2: (0,1),
            3: (0,-1)
        }

        self.human_action_mapping = {
            0: 'd',
            1: 'a',
            2: 's',
            3: 'w'
        }

        self.observation_space = Box(-1, 1, shape=[2], dtype=np.float32)
        self.action_space = Discrete(4)

        self.reward_modes = ['EXPLORE', 'COLLECT', 'ONE']
        self.reward_mode = reward_mode
        assert self.reward_mode in self.reward_modes

        self.exploration_counts = dict()
        self.collected_states = {(0,0)}
        self.beta = 1.0
        self.max_episode_steps = 1000
        self.step_num = 0
        self.cached_wall_image = None


    def is_wall(self, pos):
        if np.abs(pos[0]) >= 2*self.world_size + 3 or np.abs(pos[1]) >= 2*self.world_size + 3:
            return True
        if pos in self.specified_walls:
            return True
        else:
            for wall_rule in self.generated_wall_rules:
                if wall_rule(pos):
                    return True
        return False

    def to_image_pos(self, p):
        return (p[0] + self.world_size + 2, p[1] + self.world_size + 2)

    def get_cached_wall_image(self):
        if self.cached_wall_image is not None:
            return np.copy(self.cached_wall_image)
        else:
            wall_color = (0, 0, 0)
            canvas = 255 * np.ones([2 * self.world_size + 8, 2 * self.world_size + 8, 3], dtype=np.uint8)
            for x in range(-self.world_size - 2, self.world_size + 2 + 1):
                for y in range(-self.world_size - 2, self.world_size + 2 + 1):
                    img_x, img_y = self.to_image_pos((x, y))
                    if self.is_wall((x, y)):
                        canvas[img_y, img_x, :] = wall_color
            self.cached_wall_image = canvas
            return np.copy(self.cached_wall_image)



    def generate_image(self):
        canvas = self.get_cached_wall_image()
        agent_color = (255,0,0)
        x, y = self.to_image_pos(self.agent)
        canvas[y, x, :] = agent_color
        return canvas

    def get_exploration_reward(self, pos):
        if pos in self.exploration_counts:
            return self.beta * self.exploration_counts[pos]**-0.5
        else:
            return self.beta * 1

    def get_reward(self, pos):
        if self.reward_mode == 'ONE':
            return 1
        elif self.reward_mode == 'EXPLORE':
            return self.get_exploration_reward(pos)
        elif self.reward_mode == 'COLLECT':
            return 0 if pos in self.collected_states else 1
        else:
            raise Exception(f'Unrecognized reward mode: {self.reward_mode}')

    def update_exploration_reward(self, pos):
        if pos in self.exploration_counts:
            self.exploration_counts[pos] += 1
        else:
            self.exploration_counts[pos] = 1

    def get_observation(self):
        return np.array(self.agent) / (self.world_size + 2 + 1)

    def to_pos(self, obs):
        return tuple((obs * (self.world_size + 2 + 1)).astype(np.int32))

    def step(self, action):
        delta_x, delta_y = self.action_mapping[action]
        new_x, new_y = self.agent[0] + delta_x, self.agent[1] + delta_y
        # check for collisions and update agent position accordingly
        if not self.is_wall((new_x, new_y)):
            self.agent = (new_x, new_y)
        if not self.is_wall((new_x, new_y)):
            self.agent = (new_x, new_y)
        # update the exploration counts
        self.update_exploration_reward(self.agent)
        r = self.get_reward(self.agent)
        # update the collection set AFTER getting the reward.
        self.collected_states.add(self.agent)

        s = self.get_observation()
        info = {'internal_terminal': False,
                'r_env': r,
                'agent_position': self.agent}
        self.step_num += 1
        if self.step_num == self.max_episode_steps:
            s = self.reset()
            info['internal_terminal'] = True
        return s, r, False, info

    def reset(self):
        self.agent = (0,0)
        self.step_num = 0
        self.collected_states = {(0,0)}
        return self.get_observation()

    def visualize_trajectory(self, canvas, color, trajectory):
        for (x,y) in trajectory:
            img_x, img_y = self.to_image_pos((x,y))
            canvas[img_y, img_x,:] = color
        return canvas


if __name__ == '__main__':
    env = ExplorationWorld()
    for time in count():
        a = np.random.randint(0, 4)
        s, r, t, info = env.step(a)
        obs = env.generate_image()
        obs = cv2.resize(obs, (400, 400))
        #print(time, s)
        cv2.imshow('game', obs)
        cv2.waitKey(1)





