import os

import cv2
import numpy as np
from envs.block_world.initialization_types import AgentInitialization, ConstantInitialization, RandomInitialization
from gym.spaces import Discrete, Box

from envs.block_world.blocks import ConstantGoalBlock, AgentBlock, BackgroundBlock, RandomGoalBlock

BASE_DIR = os.path.split(os.path.realpath(__file__))[0]


class BlockPushingDomain(object):

    def __init__(self, observation_mode='vector'):
        self.grid_size = 5
        self.block_size = 8
        self.visual_mode_image_size = 64
        self.render_mode_image_size = self.grid_size * self.block_size


        self.agent_color = (0, 0, 0)
        self.block_color = (0, 0, 0)
        self.bg_color = (255, 255, 255)
        self.goal_color1 = (0, 255, 0)
        self.goal_color2 = (0, 0, 255)

        self.observation_mode = observation_mode
        assert self.observation_mode in ['vector', 'image', 'encoding']
        self.DOWN, self.UP, self.RIGHT, self.LEFT, self.NOOP = 0, 1, 2, 3, 4
        self.action_mapping = {self.DOWN: (0, 1), self.UP: (0, -1),
                               self.RIGHT: (1, 0), self.LEFT: (-1, 0), self.NOOP: (0, 0)}

        self.texture_cache = dict()

        self.goal_configuration = []
        # object positions are top-left positions.


        #self.blocks = (
        #    [AgentBlock(self.agent_color), ConstantImmoveableBlock((self.grid_size-1, 0), self.block_color)] +
        #    [ConstantGoalBlock((0, y), self.goal_color) for y in range(0, self.grid_size)] +
        #    [ConstantGoalBlock((self.grid_size-1, y), self.goal_color) for y in range(0, self.grid_size)]
        #)



        RGB_ordering = [0,1,2]
        #print(cv2.imread(os.path.join(BASE_DIR, 'blue_wall.png')))
        background_texture = cv2.imread(os.path.join(BASE_DIR, 'textures', 'blue_wall.png'))[:, :, RGB_ordering]
        agent_texture = cv2.imread(os.path.join(BASE_DIR, 'textures', 'agent_sprite.png'))[:, :, RGB_ordering]
        goal1_texture = cv2.imread(os.path.join(BASE_DIR, 'textures', 'reward_square_red.png'))[:, :, RGB_ordering]
        goal2_texture = cv2.imread(os.path.join(BASE_DIR, 'textures', 'reward_square_green.png'))[:, :, RGB_ordering]
        #print('bg_shape', background_texture.shape)
        background_color = (255, 0, 0)
        background_blocks = [BackgroundBlock((x,y), background_color, background_texture)
                             for x in range(self.grid_size) for y in range(self.grid_size)]

        # self.blocks = (
        #     [AgentBlock(self.agent_color, texture=agent_texture),
        #      ConstantGoalBlock((0,0), self.goal_color1, reward=1.0, texture=goal1_texture),
        #      ConstantGoalBlock((self.grid_size-1, self.grid_size-1), self.goal_color2, reward=1.0, texture=goal2_texture),
        #      ConstantGoalBlock((0, self.grid_size-1), self.goal_color2, reward=1.0, texture=goal2_texture)] +
        #     background_blocks
        # )

        self.blocks = (
            [AgentBlock(self.agent_color, texture=agent_texture),
             RandomGoalBlock(self.goal_color1, texture=goal1_texture, reward=1.0),
             RandomGoalBlock(self.goal_color2, texture=goal2_texture, reward=1.0)] +
            background_blocks
        )

        # any time a change is made to blocks, there needs to be a corresponding call to update the block indices.
        self.obs_blocks, self.obs_block_indices, \
        self.goal_blocks, self.goal_block_indices, \
        self.agent_index, self.agent_block = self.update_block_indices()

        self.num_objects = len(self.blocks)
        self.top_right_position = (self.grid_size-1, 0)
        self.obs_size = 2*len(self.obs_blocks)
        self.goal_size = 2*len(self.goal_blocks)
        self.max_timesteps = 30
        self.timestep = 0
        self.action_space = Discrete(5)
        if self.observation_mode == 'image':
            self.observation_space = Box(0, 255, shape=[32, 32, 3], dtype=np.uint8)
        else:
            self.observation_space = Box(-1, 1, shape=[self.obs_size + self.goal_size])

        self.reset()


    def produce_object_positions_from_blocks(self, blocks=None):
        blocks = self.blocks if blocks is None else blocks
        object_positions = []
        for i in self.obs_block_indices + self.goal_block_indices:
            pair = list(blocks[i].get_position())
            object_positions += pair
        return np.array(object_positions)


    def clone_blocks_from_object_positions(self, object_positions):
        cloned_blocks = [block.copy() for block in self.blocks]
        for index, block_index in enumerate(self.obs_block_indices + self.goal_block_indices):
            position = object_positions[2*index:2*index+2]
            cloned_blocks[block_index].position = tuple(position)
        return cloned_blocks


    def is_obs_block(self, block):
        return (block.is_moveable() or block.get_initialization_type().get_unique_name() == 'random') and (not block.is_goal())

    def is_goal_block(self, block):
        return block.is_goal()

    def is_agent_block(self, block):
        return block.get_initialization_type().get_unique_name() == 'agent'


    def update_block_indices(self):
        obs_blocks = [block for block in self.blocks if self.is_obs_block(block)]
        obs_block_indices = [i for i, block in enumerate(self.blocks) if self.is_obs_block(block)]

        goal_blocks = [block for block in self.blocks if self.is_goal_block(block)]
        goal_block_indices = [i for i, block in enumerate(self.blocks) if self.is_goal_block(block)]

        agent_blocks = [(i, block) for i, block in enumerate(self.blocks) if self.is_agent_block(block)]
        assert len(agent_blocks) == 1
        (agent_index, agent_block) = agent_blocks[0]
        return obs_blocks, obs_block_indices, goal_blocks, goal_block_indices, agent_index, agent_block


    def get_all_agent_positions(self):
        # get all the blocks that could be in the way of an agent.
        non_agent_physical_blocks = [block for block in self.blocks
                                     if (not self.is_agent_block(block)) and block.is_physical()]
        state_pairs = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                valid_position = not any([(x,y) == block.get_position() for block in non_agent_physical_blocks])
                if valid_position:
                    cloned_blocks = self.clone_blocks_with_new_agent_position((x, y))
                    positions = self.produce_object_positions_from_blocks(blocks=cloned_blocks)
                    state = self.get_observation(self.observation_mode, object_positions=positions)
                    state_pairs.append(((x,y), state))
        return state_pairs



    def get_all_states(self):
        states = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (x, y) == self.top_right_position:
                    states.append(None)
                else:
                    cloned_blocks = self.clone_blocks_with_new_agent_position((x,y))
                    positions = self.produce_object_positions_from_blocks(blocks=cloned_blocks)
                    states.append(self.get_observation(self.observation_mode, object_positions=positions))
        return states

    def clone_blocks_with_new_agent_position(self, new_agent_position):
        new_blocks = []
        for block in self.blocks:
            new_block = block.copy()
            if new_block.get_initialization_type().get_unique_name() == 'agent':
                new_block.position = tuple(new_agent_position)
            elif new_block.get_position() == new_agent_position and new_block.is_physical():
                raise Exception('Cannot put agent in occupied position')
            new_blocks.append(new_block)
        return new_blocks


    def get_current_state(self):
        return {
            'blocks': [block.copy() for block in self.blocks],
            'timestep': self.timestep
        }


    def restore_state(self, state):
        self.blocks = [block.copy() for block in state['blocks']]
        self.timestep = state['timestep']
        self.obs_blocks, self.obs_block_indices, \
        self.goal_blocks, self.goal_block_indices, \
        self.agent_index, self.agent_block = self.update_block_indices()
        return self.get_observation(self.observation_mode)


    def get_nonintersecting_positions(self, num_positions, existing_positions=set()):
        positions = set()
        while len(positions) < num_positions:
            candidate = tuple(np.random.randint(0, self.grid_size, size=2))
            if candidate not in existing_positions:
                positions.add(candidate)
        return list(positions)

    def produce_image(self, object_positions, image_size):
        blocks = self.clone_blocks_from_object_positions(object_positions)
        canvas = np.zeros(shape=[self.grid_size*self.block_size, self.grid_size*self.block_size, 3], dtype=np.uint8)
        canvas[:, :] = self.bg_color
        for block in sorted(blocks, key=lambda x: x.get_draw_priority()):
            pos = block.get_position()
            if block.is_textured():
                color_data = self.get_texture_using_cache(block)
            else:
                color_data = block.get_color()
            x_pos, y_pos = self.block_size * pos[0], self.block_size * pos[1]
            canvas[y_pos:y_pos+self.block_size, x_pos:x_pos+self.block_size] = color_data
        canvas = cv2.resize(canvas, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        return canvas

    def get_texture_using_cache(self, block):
        if block.id in self.texture_cache:
            return self.texture_cache[block.id]
        else:
            color_data = block.get_texture()
            color_data = cv2.resize(color_data, (self.block_size, self.block_size), interpolation=cv2.INTER_NEAREST)
            self.texture_cache[block.id] = color_data
            return color_data



    def extract_obs_part_vector(self, obs):
        return obs[:2*len(self.obs_blocks)]


    def extract_goal_part_vector(self, obs):
        return obs[-2*len(self.goal_blocks):]


    def extract_agent_vector(self, obs):
        return obs[2*self.agent_index:2*self.agent_index+2]

    # expects object_positions as tuples.
    def produce_vector(self, object_positions):
        return np.concatenate(object_positions, axis=0)

    def get_observation(self, observation_mode, object_positions=None):
        if object_positions is None:
            object_positions = self.produce_object_positions_from_blocks()
        if observation_mode == 'image':
            # TODO there might be something that I need to do in terms of goals in the future. like having goal images
            # as seperate images that get concatenated.
            image = self.produce_image(object_positions, self.visual_mode_image_size)
            return image
        elif observation_mode == 'encoding':
            raise NotImplemented
        elif observation_mode == 'vector':
            vector = np.copy(object_positions)
            return vector / self.grid_size

    def action_converter(self, raw_action):
        return np.argmax(raw_action)

    def out_of_bounds(self, pos):
        pos_x, pos_y = pos
        return not ((0 <= pos_x) and (pos_x < self.grid_size) and (0 <= pos_y) and (pos_y < self.grid_size))

    def can_move(self, block, delta, block_index, all_blocks, update_set):
        # should not ever call can_move on a non-physical block.
        assert block.is_physical()

        pos = block.get_position()
        new_pos = tuple(np.array(pos) + np.array(delta))
        if self.out_of_bounds(new_pos) or (not block.is_moveable()):
            return False
        can_move = True
        for other_block_index, other_block in enumerate(all_blocks):
            if (block_index == other_block_index) or (not other_block.is_physical()):
                continue
            if new_pos == other_block.get_position():
                updated_all_blocks = [b.copy() for b in all_blocks]
                updated_all_blocks[block_index].set_position(new_pos)
                can_move = self.can_move(other_block, delta, other_block_index, updated_all_blocks, update_set)
                if not can_move:
                    break
        if can_move:
            update_set.add((block_index, new_pos))
        return can_move


    def perform_action(self, action):
        self.timestep += 1
        delta = self.action_mapping[action]
        update_set = set()
        can_move = self.can_move(self.agent_block, delta, self.agent_index, self.blocks, update_set)
        if can_move:
            for index, new_pos in update_set:
                self.blocks[index].set_position(new_pos)


    def get_reward(self, obs):
        goal_part = self.extract_goal_part_vector(obs)
        assert len(goal_part) % 2 == 0
        reward_zone = [tuple(goal_part[2*i:2*i+2]) for i in range(len(goal_part)//2)]
        [agent_x, agent_y] = self.extract_agent_vector(obs)
        reward = 0
        for i, (x, y) in enumerate(reward_zone):
            if (agent_x, agent_y) == (x, y):
                reward = self.goal_blocks[i].get_reward()
                break
        return reward


    def get_terminal(self, obs):
        at_goal_state = (self.get_reward(obs) == 1.0)
        game_over = (self.timestep >= self.max_timesteps)
        return game_over or at_goal_state
        #return game_over

    def step(self, raw_action):
        #old_obs = self.get_observation()
        action = raw_action
        self.perform_action(action)
        # TODO : find a better way to handle rewards when we are dealing with images. Right now we just pass the
        # vectorized observation to the get_reward function.
        new_obs_vec = self.get_observation('vector')
        new_obs = self.get_observation(self.observation_mode)
        reward = self.get_reward(new_obs_vec)
        terminal = self.get_terminal(new_obs_vec)
        info = {'internal_terminal': terminal}
        if terminal:
            self.reset()
        # TODO use old_obs for hindsight.
        return new_obs, reward, False, info

    def render(self):
        object_positions = self.produce_object_positions_from_blocks()
        image = self.produce_image(object_positions, self.render_mode_image_size)
        cv2.imshow('game', image)
        cv2.waitKey(1)


    def reset(self):
        initialized_positions = set()
        initialized_positions = initialized_positions.union(
            ConstantInitialization().initialize(self.blocks, {'grid_size': self.grid_size,
                                                              'initialized_positions': initialized_positions}))
        initialized_positions = initialized_positions.union(
            RandomInitialization().initialize(self.blocks, {'grid_size': self.grid_size,
                                                            'initialized_positions': initialized_positions}))
        initialized_positions = initialized_positions.union(
            AgentInitialization().initialize(self.blocks, {'grid_size': self.grid_size,
                                                           'initialized_positions': initialized_positions}))

        self.timestep = 0
        return self.get_observation(self.observation_mode)


def onehot(i, n):
    vec = np.zeros(shape=[n])
    vec[i] = 1
    return vec


action_mapping = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0), 4: (0, 0)}

def block_info(block):
    return (block.get_position(), block.get_initialization_type().get_unique_name())

if __name__ == '__main__':
    env = BlockPushingDomain(observation_mode='image')
    s = env.reset()
    env.render()
    action_map = {'w': onehot(1, 5),
                  'a': onehot(3, 5),
                  's': onehot(0, 5),
                  'd': onehot(2, 5),
                  ' ': onehot(4, 5)}
    state = None
    while True:
        action_key = input('Action: ')
        if action_key == 'r':
            env.reset()
            env.render()
            continue
        if action_key == 'save':
            state = env.get_current_state()
            print([block_info(block) for block in state['blocks']])
            continue
        if action_key == 'restore':
            env.restore_state(state)
            print([block_info(block) for block in env.blocks])
            env.render()
            continue
        if action_key not in action_map:
            continue
        action = action_map[action_key]
        print(action)
        s, reward, terminal, _ = env.step(np.argmax(action))
        cv2.imwrite('./test.png', s)
        print(s.shape)
        env.render()
        print(f'reward {reward}, terminal {terminal}')
        if terminal:
            env.reset()
            env.render()

