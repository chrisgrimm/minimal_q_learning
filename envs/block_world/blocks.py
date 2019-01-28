import numpy as np

from envs.block_world.initialization_types import ConstantInitialization, RandomInitialization, AgentInitialization


def block_id_generator():
    id = 0
    while True:
        yield id
        id += 1

BLOCK_ID_GEN = block_id_generator()

class Block(object):

    def get_color(self):
        raise NotImplementedError

    def is_textured(self):
        raise NotImplementedError

    def get_texture(self):
        raise NotImplementedError

    def get_position(self):
        raise NotImplementedError

    def set_position(self, pos):
        raise NotImplementedError

    def is_physical(self):
        raise NotImplementedError

    def is_moveable(self):
        raise NotImplementedError

    def is_goal(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def get_initialization_type(self):
        raise NotImplementedError

    # low numbers are drawn first.
    def get_draw_priority(self):
        raise NotImplementedError


class ImmoveableBlock(Block):

    def is_physical(self):
        return True

    def is_moveable(self):
        return False

    def is_goal(self):
        return False

    def get_draw_priority(self):
        return 1


class MoveableBlock(Block):

    def is_physical(self):
        return True

    def is_moveable(self):
        return True

    def is_goal(self):
        return False

    def get_draw_priority(self):
        return 1


class GoalBlock(Block):

    def is_physical(self):
        return False

    def is_moveable(self):
        return False

    def is_goal(self):
        return True

    def get_draw_priority(self):
        return 0

    def get_reward(self):
        raise NotImplementedError




class ConstantMoveableBlock(MoveableBlock, Block):

    def __init__(self, position, color, texture=None, id=None):
        self.position = tuple(position)
        self.color = tuple(color)
        self.texture = texture
        self.id = next(BLOCK_ID_GEN) if id is None else id

    def get_color(self):
        return self.color

    def is_textured(self):
        return self.texture is not None

    def get_texture(self):
        if self.texture is None:
            raise Exception('Cannot get texture of block without specified texture.')
        else:
            return self.texture

    def get_position(self):
        return self.position

    def set_position(self, pos):
        self.position = tuple(pos)

    def copy(self):
        texture = None if self.texture is None else np.copy(self.texture)
        return ConstantMoveableBlock(self.position, self.color, texture=texture, id=self.id)

    def get_initialization_type(self):
        return ConstantInitialization()



class RandomMoveableBlock(MoveableBlock, Block):

    def __init__(self, color, texture=None, id=None):
        self.position = (0,0)
        self.color = tuple(color)
        self.texture = texture
        self.id = next(BLOCK_ID_GEN) if id is None else id


    def get_color(self):
        return self.color

    def is_textured(self):
        return self.texture is not None

    def get_texture(self):
        if self.texture is None:
            raise Exception('Cannot get texture of block without specified texture.')
        else:
            return self.texture

    def get_position(self):
        return self.position

    def set_position(self, pos):
        self.position = tuple(pos)

    def copy(self):
        texture = None if self.texture is None else np.copy(self.texture)
        block = RandomMoveableBlock(self.color, texture=texture, id=self.id)
        block.position = self.position
        return block

    def get_initialization_type(self):
        return RandomInitialization()

class ConstantImmoveableBlock(ImmoveableBlock, Block):

    def __init__(self, position, color, texture=None, id=None):
        self.position = tuple(position)
        self.color = tuple(color)
        self.texture = texture
        self.id = next(BLOCK_ID_GEN) if id is None else id


    def get_color(self):
        return self.color

    def is_textured(self):
        return self.texture is not None

    def get_texture(self):
        if self.texture is None:
            raise Exception('Cannot get texture of block without specified texture.')
        else:
            return self.texture

    def get_position(self):
        return self.position

    def set_position(self, pos):
        raise Exception('Cannot set position of Constant Block')

    def copy(self):
        texture = None if self.texture is None else np.copy(self.texture)
        return ConstantImmoveableBlock(self.position, self.color, texture=texture, id=self.id)

    def get_initialization_type(self):
        return ConstantInitialization()

class RandomImmoveableBlock(ImmoveableBlock, Block):

    def __init__(self, color, texture=None, id=None):
        self.position = (0,0)
        self.color = tuple(color)
        self.texture = texture
        self.id = next(BLOCK_ID_GEN) if id is None else id


    def get_color(self):
        return self.color

    def is_textured(self):
        return self.texture is not None

    def get_texture(self):
        if self.texture is None:
            raise Exception('Cannot get texture of block without specified texture.')
        else:
            return self.texture

    def get_position(self):
        return self.position

    def set_position(self, pos):
        self.position = tuple(pos)

    def copy(self):
        texture = None if self.texture is None else np.copy(self.texture)
        block = RandomImmoveableBlock(self.color, texture=texture, id=self.id)
        block.position = self.position
        return block

    def get_initialization_type(self):
        return RandomInitialization()

class AgentBlock(MoveableBlock, Block):
    def __init__(self, color, texture=None, id=None):
        self.position = (0,0)
        self.color = tuple(color)
        self.texture = texture
        self.id = next(BLOCK_ID_GEN) if id is None else id


    def get_color(self):
        return self.color

    def is_textured(self):
        return self.texture is not None

    def get_texture(self):
        if self.texture is None:
            raise Exception('Cannot get texture of block without specified texture.')
        else:
            return self.texture

    def get_position(self):
        return self.position

    def set_position(self, pos):
        self.position = tuple(pos)

    def copy(self):
        texture = None if self.texture is None else np.copy(self.texture)
        block = AgentBlock(self.color, texture=texture, id=self.id)
        block.position = self.position
        return block

    def get_initialization_type(self):
        return AgentInitialization()

class ConstantGoalBlock(GoalBlock, Block):

    def __init__(self, position, color, reward=1.0, texture=None, id=None):
        self.position = tuple(position)
        self.color = tuple(color)
        self.reward = reward
        self.texture = texture
        self.id = next(BLOCK_ID_GEN) if id is None else id


    def get_color(self):
        return self.color

    def is_textured(self):
        return self.texture is not None

    def get_texture(self):
        if self.texture is None:
            raise Exception('Cannot get texture of block without specified texture.')
        else:
            return self.texture

    def get_position(self):
        return self.position

    def set_position(self, pos):
        raise Exception('Cannot set position of constant block')

    def copy(self):
        texture = None if self.texture is None else np.copy(self.texture)
        return ConstantGoalBlock(self.position, self.color, texture=texture, id=self.id)

    def get_initialization_type(self):
        return ConstantInitialization()

    def get_reward(self):
        return self.reward

class ConstantInteractiveBlock(Block):

    def __init__(self, position, color, reward=0.0, texture=None, id=None):
        self.position = tuple(position)
        self.color = tuple(color)
        self.reward = reward
        self.texture = texture
        self.id = next(BLOCK_ID_GEN) if id is None else id



class RandomGoalBlock(GoalBlock, Block):

    def __init__(self, color, reward=1.0, texture=None, id=None):
        self.position = (0,0)
        self.color = tuple(color)
        self.reward = reward
        self.texture = texture
        self.id = next(BLOCK_ID_GEN) if id is None else id


    def get_color(self):
        return self.color

    def is_textured(self):
        return self.texture is not None

    def get_texture(self):
        if self.texture is None:
            raise Exception('Cannot get texture of block without specified texture.')
        else:
            return self.texture

    def get_position(self):
        return self.position

    def set_position(self, pos):
        self.position = tuple(pos)

    def copy(self):
        texture = None if self.texture is None else np.copy(self.texture)
        block = RandomGoalBlock(self.color, texture=texture, id=self.id)
        block.position = self.position
        return block

    def get_initialization_type(self):
        return RandomInitialization()

    def get_reward(self):
        return self.reward


class BackgroundBlock(Block):

    def __init__(self, position, color, texture=None, id=None):
        self.position = tuple(position)
        self.color = tuple(color)
        self.texture = texture
        self.id = next(BLOCK_ID_GEN) if id is None else id


    def get_color(self):
        return self.color

    def is_textured(self):
        return self.texture is not None

    def get_texture(self):
        if self.texture is None:
            raise Exception('Cannot get texture of block without specified texture.')
        else:
            return self.texture

    def get_position(self):
        return self.position

    def set_position(self, pos):
        raise Exception('Cannot set position of constant block')

    def copy(self):
        texture = None if self.texture is None else np.copy(self.texture)
        block = BackgroundBlock(self.position, self.color, texture=texture, id=self.id)
        return block

    def get_initialization_type(self):
        return ConstantInitialization()

    def is_physical(self):
        return False

    def is_moveable(self):
        return False

    def is_goal(self):
        return False

    def get_draw_priority(self):
        return -1



