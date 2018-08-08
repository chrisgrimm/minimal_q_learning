from envs.initialization_types import ConstantInitialization, RandomInitialization, AgentInitialization

class Block(object):

    def get_color(self):
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




class ConstantMoveableBlock(MoveableBlock, Block):

    def __init__(self, position, color):
        self.position = tuple(position)
        self.color = tuple(color)

    def get_color(self):
        return self.color

    def get_position(self):
        return self.position

    def set_position(self, pos):
        raise Exception('Cannot set position of Constant Block')

    def copy(self):
        return ConstantMoveableBlock(self.position, self.color)

    def get_initialization_type(self):
        return ConstantInitialization()



class RandomMoveableBlock(MoveableBlock, Block):

    def __init__(self, color):
        self.position = (0,0)
        self.color = tuple(color)

    def get_color(self):
        return self.color

    def get_position(self):
        return self.position

    def set_position(self, pos):
        self.position = tuple(pos)

    def copy(self):
        return RandomMoveableBlock(self.color)

    def get_initialization_type(self):
        return RandomInitialization()

class ConstantImmoveableBlock(ImmoveableBlock, Block):

    def __init__(self, position, color):
        self.position = tuple(position)
        self.color = tuple(color)

    def get_color(self):
        return self.color

    def get_position(self):
        return self.position

    def set_position(self, pos):
        raise Exception('Cannot set position of Constant Block')

    def copy(self):
        return ConstantImmoveableBlock(self.position, self.color)

    def get_initialization_type(self):
        return ConstantInitialization()

class RandomImmoveableBlock(ImmoveableBlock, Block):

    def __init__(self, color):
        self.position = (0,0)
        self.color = tuple(color)

    def get_color(self):
        return self.color

    def get_position(self):
        return self.position

    def set_position(self, pos):
        raise Exception('Cannot set position of Constant Block')

    def copy(self):
        return RandomImmoveableBlock(self.color)

    def get_initialization_type(self):
        return RandomInitialization()

class AgentBlock(MoveableBlock, Block):
    def __init__(self, color):
        self.position = (0,0)
        self.color = tuple(color)

    def get_color(self):
        return self.color

    def get_position(self):
        return self.position

    def set_position(self, pos):
        self.position = tuple(pos)

    def copy(self):
        return AgentBlock(self.color)

    def get_initialization_type(self):
        return AgentInitialization()

class ConstantGoalBlock(GoalBlock, Block):

    def __init__(self, position, color):
        self.position = tuple(position)
        self.color = tuple(color)

    def get_color(self):
        return self.color

    def get_position(self):
        return self.position

    def set_position(self, pos):
        self.position = pos

    def copy(self):
        return ConstantGoalBlock(self.position, self.color)

    def get_initialization_type(self):
        return ConstantInitialization()

class RandomGoalBlock(GoalBlock, Block):

    def __init__(self, color):
        self.position = (0,0)
        self.color = tuple(color)

    def get_color(self):
        return self.color

    def get_position(self):
        return self.position

    def set_position(self, pos):
        raise Exception('Cannot set position of Goal Block')

    def copy(self):
        return RandomGoalBlock(self.position, self.color)

    def get_initialization_type(self):
        return ConstantInitialization()



