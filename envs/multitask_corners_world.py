from gym import Env
from gym.spaces import Box, Discrete
import numpy as np
from itertools import count
import cv2
from reward_network2 import ReparameterizedRewardNetwork


class CornersTaskWorld(Env):

    def __init__(self, world_size=5, visual=False, task=(1,1,1,1), reset_mode='deterministic'):
        assert reset_mode in ['deterministic', 'random']
        self.reset_mode = reset_mode
        self.agent = (0,0)
        self.image_mode = visual
        self.image_size = 64
        self.world_size = world_size
        self.corner_walls = [(1,0), (1,world_size-1), (world_size-2,0), (world_size-2,world_size-1)]
        self.corners = [(0,0), (0, world_size-1), (world_size-1, 0), (world_size-1,world_size-1)]
        self.specified_walls = set(self.corner_walls)
        self.corner_set = set(self.corners)

        self.set_task(task)

        self.generated_wall_rules = [
            #lambda p: ((np.abs(p[0]) > 2) and (np.abs(p[0]) == np.abs(p[1]))), # builds the diagonal walls.
            #lambda p: ((np.abs(p[0]) > self.world_size + 2) or (np.abs(p[1]) > self.world_size + 2)), # builds the outer walls
            lambda p: (p[0] < 0 or p[1] < 0 or p[0] >= self.world_size or p[1] >= self.world_size),
        ]


        self.action_mapping = {
            0: (1,0),
            1: (-1,0),
            2: (0,1),
            3: (0,-1)
        }

        self.human_action_mapping = {
            'd': 0,
            'a': 1,
            's': 2,
            'w': 3
        }

        if self.image_mode:
            self.observation_space = Box(0, 255, shape=[64,64,3], dtype=np.uint8)
        else:
            self.observation_space = Box(-1, 1, shape=[2], dtype=np.float32)
        self.action_space = Discrete(4)

        self.max_episode_steps = 1000
        self.step_num = 0
        self.cached_wall_image = None
        self.cached_collection_image = None


    def is_wall(self, pos):
        if np.abs(pos[0]) >= self.world_size + 3 or np.abs(pos[1]) >= self.world_size + 3:
            return True
        if pos in self.specified_walls:
            return True
        else:
            for wall_rule in self.generated_wall_rules:
                if wall_rule(pos):
                    return True
        return False

    def to_image_pos(self, p):
        return p

    def position_iterator(self):
        for x in range(0, self.world_size):
            for y in range(0, self.world_size):
                yield (x, y)


    def get_cached_wall_image(self):
        if self.cached_wall_image is not None:
            return np.copy(self.cached_wall_image)
        else:
            wall_color = (0, 0, 0)
            reward_color = (255, 0, 0)
            canvas = 255 * np.ones([self.world_size, self.world_size, 3], dtype=np.uint8)
            for (x, y) in self.position_iterator():
                img_x, img_y = self.to_image_pos((x, y))
                if self.is_wall((x, y)):
                    canvas[img_y, img_x, :] = wall_color
                if (x, y) in self.corner_set:
                    canvas[img_y, img_x, :] = reward_color
            self.cached_wall_image = canvas
            return np.copy(self.cached_wall_image)





    def generate_image(self, position=None):
        position = self.agent if position is None else position
        canvas = self.get_cached_wall_image()
        agent_color = (255,0,0)
        x, y = self.to_image_pos(position)
        canvas[y, x, :] = agent_color
        return canvas


    def get_reward(self, pos):
        return 1 if pos in self.rewarding_pos else 0


    def get_observation(self, position=None):
        position = self.agent if position is None else position
        if self.image_mode:
            img_x, img_y = self.to_image_pos(position)
            obs = self.get_cached_wall_image()
            #print(obs)
            obs[img_y, img_x, :] = (0,0,255)
            obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_NEAREST)
            return obs
        else:
            return np.array(position) / (self.world_size + 2 + 1)

    def get_action_update(self, pos, a):
        delta_x, delta_y = self.action_mapping[a]
        new_x, new_y = pos[0] + delta_x, pos[1] + delta_y
        if (new_x, new_y) in self.corner_set:
            return (self.world_size // 2, self.world_size // 2)
        if not self.is_wall((new_x, new_y)):
            return new_x, new_y
        return pos

    def step(self, action, update_internal=True):
        delta_x, delta_y = self.action_mapping[action]
        new_x, new_y = self.agent[0] + delta_x, self.agent[1] + delta_y
        # check for collisions and update agent position accordingly
        if not self.is_wall((new_x, new_y)) and update_internal:
            self.agent = (new_x, new_y)
        # update the exploration counts
        r = self.get_reward(self.agent)
        s = self.get_observation()
        info = {'internal_terminal': False,
                'r_env': r,
                'agent_position': self.agent}
        if update_internal:
            self.step_num += 1
        if self.step_num == self.max_episode_steps or r == 1:
            s = self.reset(update_internal=update_internal)
            info['internal_terminal'] = True
        return s, r, False, info

    def set_task(self, task):
        self.task = task
        self.rewarding_pos = set([pos for active, pos in zip(self.task, self.corners) if active == 1])
        self.cached_wall_image = None


    def sample_random_position(self):
        while True:
            pos = tuple(np.random.randint(0, self.world_size, size=[2]))
            if pos in self.corner_set or self.is_wall(pos):
                continue
            return pos


    def reset(self, update_internal=True):
        if self.reset_mode == 'deterministic':
            agent = (self.world_size // 2, self.world_size // 2)
        else:
            agent = tuple(np.random.randint(0, self.world_size, size=2))
        if update_internal:
            self.agent = agent
            self.step_num = 0
        return self.get_observation(agent)

    def visualize_trajectory(self, canvas, color, trajectory):
        for (x,y) in trajectory:
            img_x, img_y = self.to_image_pos((x,y))
            # theres a weird indexing thing, but i dont think it really matters for this application.
            try:
                canvas[img_y, img_x,:] = color
            except IndexError:
                pass
        return canvas

    def visualize_reward_bonuses(self):
        canvas = np.zeros_like(self.get_cached_wall_image()[:, :, 0], dtype=np.float32)
        return canvas

    def visualize_rewards(self, net: ReparameterizedRewardNetwork):
        make_canvas = lambda: np.zeros_like(self.get_cached_wall_image()[:, :, 0], dtype=np.float32)
        canvases = [make_canvas() for _ in range(net.num_rewards)]
        for (x, y) in self.position_iterator():
            if self.is_wall((x, y)) or (x, y) in self.corner_set:
                continue
            S, A, SP = [], [], []
            for a in range(self.action_space.n):
                (new_x, new_y) = self.get_action_update((x, y), a)
                S.append(self.get_observation((x, y)))
                A.append(a)
                SP.append(self.get_observation((new_x, new_y)))
            if self.is_wall((x, y)):
                continue
            rewards = np.max(net.get_partitioned_reward(S, A, SP), axis=0) # [num_partitions]
            for reward_num, reward in enumerate(rewards):
                canvases[reward_num][y, x] = reward
        # process canvases into heatmaps
        heatmaps = []
        for canvas in canvases:
            canvas = np.clip(canvas, 0, np.inf)  # strip negative values
            canvas = canvas / np.max(canvas)
            canvas = (255 * canvas).astype(np.uint8)
            canvas = cv2.applyColorMap(canvas, cv2.COLORMAP_HOT)
            heatmaps.append(canvas)
        return heatmaps


    def visualize_reward_values(self, net: ReparameterizedRewardNetwork):
        make_canvas = lambda: np.zeros_like(self.get_cached_wall_image()[:, :, 0], dtype=np.float32)
        canvases = [make_canvas() for _ in range(net.num_rewards)]
        for (x,y) in self.position_iterator():
            img_x, img_y = self.to_image_pos((x,y))
            obs = self.get_observation((x,y))
            values = net.get_state_values([obs])[:, 0]
            for reward_num, value in enumerate(values):
                canvases[reward_num][img_y, img_x] = value
        # process canvases into heatmaps
        heatmaps = []
        for canvas in canvases:
            canvas = np.clip(canvas, 0, np.inf) # strip negative values
            canvas = canvas / np.max(canvas)
            canvas = (255 * canvas).astype(np.uint8)
            canvas = cv2.applyColorMap(canvas, cv2.COLORMAP_HOT)
            heatmaps.append(canvas)
        return heatmaps







if __name__ == '__main__':
    import os
    from state_representation_wrapper import StateRepresentationWrapper
    world_size = 15
    env = CornersTaskWorld(visual=True, task=(1,1,1,1), world_size=world_size)

    #path = '/Users/chris/projects/q_learning/reparam_runs/'
    #names = ['top_5', 'bottom_5']  # , 'left_5', 'right_5']
    #paths = [os.path.join(path, name, 'weights') for name in names]
    #repr_wrapper = StateRepresentationWrapper(env, 2, paths)
    #net = repr_wrapper.networks[0]
    #net = ReparameterizedRewardNetwork(env, 2, 0.001, None, 4, 'reward_net', True, gpu_num=-1)

    use_network = True
    if use_network:
        net = ReparameterizedRewardNetwork(env, 2, 0.001, None, 4, 'reward_net', True, gpu_num=-1)

        path = f'/Users/chris/projects/q_learning/reparam_runs/corners_right_{world_size}_2/weights'
        net.restore(path, 'reward_net.ckpt')
        # restore state rep too.
        path = '/Users/chris/projects/q_learning/reparam_runs/'
        # names = ['top_5', 'bottom_5']#, 'left_5', 'right_5']
        names = ['undecomp_bottom_5', 'undecomp_top_5']
        paths = [os.path.join(path, name, 'weights') for name in names]
        repr_wrapper = StateRepresentationWrapper(env, 1, paths, mode='dqn')

    policy_num = 1
    s = env.reset()
    for time in count():
        if not use_network:
            try:
                a = input('Action:')
                if a == 'r':
                    env.reset()
                    continue
                a = env.human_action_mapping[a]
            except KeyError:
                print('invalid action!')
                continue
        else:
            input('...')
            a = net.get_state_actions([s])[policy_num][0]
        s, r, t, info = env.step(a)
        print(repr_wrapper.get_state_repr([s])[0])
        print(r)

        obs = env.get_observation()
        obs = cv2.resize(obs, (400, 400))
        #print(time, s)
        cv2.imshow('game', obs)
        cv2.waitKey(1)





