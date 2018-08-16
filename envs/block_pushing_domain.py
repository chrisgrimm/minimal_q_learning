import numpy as np
import cv2
from gym.spaces import Discrete, Box
from envs.blocks import ConstantImmoveableBlock, ConstantMoveableBlock, ConstantGoalBlock, RandomMoveableBlock, \
    RandomGoalBlock, RandomImmoveableBlock, AgentBlock
from envs.initialization_types import AgentInitialization, ConstantInitialization, RandomInitialization



class BlockPushingDomain(object):

    def __init__(self):
        self.grid_size = 5
        self.block_size = 4

        self.agent_color = (0, 0, 0)
        self.block_color = (0, 0, 0)
        self.bg_color = (255, 255, 255)
        self.goal_color1 = (0, 255, 0)
        self.goal_color2 = (0, 0, 255)

        self.observation_mode = 'vector'
        assert self.observation_mode in ['vector', 'image', 'encoding']
        self.DOWN, self.UP, self.RIGHT, self.LEFT, self.NOOP = 0, 1, 2, 3, 4
        self.action_mapping = {self.DOWN: (0, 1), self.UP: (0, -1),
                               self.RIGHT: (1, 0), self.LEFT: (-1, 0), self.NOOP: (0, 0)}

        self.goal_configuration = []
        # object positions are top-left positions.


        #self.blocks = (
        #    [AgentBlock(self.agent_color), ConstantImmoveableBlock((self.grid_size-1, 0), self.block_color)] +
        #    [ConstantGoalBlock((0, y), self.goal_color) for y in range(0, self.grid_size)] +
        #    [ConstantGoalBlock((self.grid_size-1, y), self.goal_color) for y in range(0, self.grid_size)]
        #)

        # self.blocks = (
        #     [AgentBlock(self.agent_color),
        #      RandomGoalBlock(self.goal_color1, reward=1.0),
        #      RandomGoalBlock(self.goal_color2, reward=1.0)]
        # )

        self.blocks = (
            [AgentBlock(self.agent_color),
             ConstantGoalBlock((0,0), self.goal_color1, reward=1.0),
             ConstantGoalBlock((self.grid_size-1, self.grid_size-1), self.goal_color2, reward=1.0)]
        )

        # any time a change is made to blocks, there needs to be a corresponding call to update the block indices.
        self.obs_blocks, self.goal_blocks, self.agent_index, self.agent_block = self.update_block_indices()

        self.num_objects = len(self.blocks)
        self.top_right_position = (self.grid_size-1, 0)
        self.obs_size = 2*len(self.obs_blocks)
        self.goal_size = 2*len(self.goal_blocks)
        self.max_timesteps = 30
        self.timestep = 0
        self.action_space = Discrete(5)
        self.observation_space = Box(-1, 1, shape=[self.obs_size + self.goal_size])
        self.reset()

    def is_obs_block(self, block):
        return (block.is_moveable() or block.get_initialization_type().get_unique_name() == 'random') and (not block.is_goal())

    def is_goal_block(self, block):
        return block.is_goal()

    def is_agent_block(self, block):
        return block.get_initialization_type().get_unique_name() == 'agent'


    def update_block_indices(self):
        obs_blocks = [block for block in self.blocks if self.is_obs_block(block)]
        goal_blocks = [block for block in self.blocks if self.is_goal_block(block)]
        agent_blocks = [(i, block) for i, block in enumerate(self.blocks) if self.is_agent_block(block)]
        assert len(agent_blocks) == 1
        (agent_index, agent_block) = agent_blocks[0]
        return obs_blocks, goal_blocks, agent_index, agent_block


    def get_all_agent_positions(self):
        non_agent_non_goal_blocks = [block for block in self.blocks
                                     if not (self.is_agent_block(block) or self.is_goal_block(block))]
        state_pairs = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                valid_position = not any([(x,y) == block.get_position() for block in non_agent_non_goal_blocks])
                if valid_position:
                    cloned_blocks = self.clone_blocks_with_new_agent_position((x, y))
                    positions_obs = [block.get_position() for block in cloned_blocks if self.is_obs_block(block)]
                    positions_goal = [block.get_position() for block in cloned_blocks if self.is_goal_block(block)]
                    state = self.get_observation(object_positions=positions_obs + positions_goal)
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
                    positions_obs = [block.get_position() for block in cloned_blocks if self.is_obs_block(block)]
                    positions_goal = [block.get_position() for block in cloned_blocks if self.is_goal_block(block)]
                    states.append(self.get_observation(object_positions=positions_obs + positions_goal))
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
        self.obs_blocks, self.goal_blocks, self.agent_index, self.agent_block = self.update_block_indices()
        return self.get_observation()


    def get_nonintersecting_positions(self, num_positions, existing_positions=set()):
        positions = set()
        while len(positions) < num_positions:
            candidate = tuple(np.random.randint(0, self.grid_size, size=2))
            if candidate not in existing_positions:
                positions.add(candidate)
        return list(positions)

    def produce_image(self, blocks):
        image_size = self.grid_size * self.block_size
        canvas = np.zeros(shape=[image_size, image_size, 3])
        canvas[:, :] = self.bg_color
        for block in sorted(blocks, key=lambda x: x.get_draw_priority()):
            pos = block.get_position()
            color = block.get_color()
            x_pos, y_pos = self.block_size * pos[0], self.block_size * pos[1]
            canvas[y_pos:y_pos+self.block_size, x_pos:x_pos+self.block_size] = color
        return canvas


    def extract_obs_part_vector(self, obs):
        return obs[:2*len(self.obs_blocks)]


    def extract_goal_part_vector(self, obs):
        return obs[-2*len(self.goal_blocks):]


    def extract_agent_vector(self, obs):
        return obs[2*self.agent_index:2*self.agent_index+2]


    def produce_vector(self, object_positions):
        return np.concatenate(object_positions, axis=0)

    def get_observation(self, object_positions=None):
        if object_positions is None:
            object_positions = (
                [block.get_position() for block in self.obs_blocks] +
                [block.get_position() for block in self.goal_blocks]
            )
        if self.observation_mode == 'image':
            raise Exception('This is not properly implemented yet.')
            image = self.produce_image()
            goal_image = self.produce_image(self.goal_configuration)
            return np.concatenate([image, goal_image], axis=2)
        elif self.observation_mode == 'encoding':
            raise NotImplemented
        elif self.observation_mode == 'vector':
            vector = self.produce_vector(object_positions)
            #goal_vector = self.produce_vector(self.goal_configuration)
            return vector / self.grid_size
            #return vector

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

        #in_reward_zone = (agent_x, agent_y) in reward_zone
        #reward = 1.0 if in_reward_zone else 0.0
        #return reward
        #assert self.observation_mode in ['vector', 'encoding']
        #return 10.0 if self.get_terminal(obs, goal) else -0.1
        #raise NotImplemented


    def get_terminal(self, obs, goal=None):
        return self.timestep >= self.max_timesteps

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
        image = self.produce_image(self.blocks)
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
        return self.get_observation()


def onehot(i, n):
    vec = np.zeros(shape=[n])
    vec[i] = 1
    return vec


action_mapping = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0), 4: (0, 0)}

def block_info(block):
    return (block.get_position(), block.get_initialization_type().get_unique_name())

if __name__ == '__main__':
    env = BlockPushingDomain()
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
        env.render()
        print(f'reward {reward}, terminal {terminal}')
        if terminal:
            env.reset()
            env.render()

