from envs.utils import get_nonintersecting_positions

class InitializationType(object):

    def get_unique_name(self):
        raise NotImplementedError

    def initialize(self, all_blocks, env_parameters):
        all_blocks_of_type = [block for block in all_blocks
                              if block.get_initialization_type().get_unique_name() == self.get_unique_name()]
        self._initialize(all_blocks_of_type, env_parameters)
        initialized_positions = set([block.get_position() for block in all_blocks_of_type])
        return initialized_positions

    def _initialize(self, all_blocks_of_type, env_parameters):
        raise NotImplementedError

class RandomInitialization(InitializationType):

    def get_unique_name(self):
        return 'random'

    def _initialize(self, all_blocks_of_type, env_parameters):

        positions = get_nonintersecting_positions(env_parameters['grid_size'], len(all_blocks_of_type), env_parameters['initialized_positions'])

        for pos, block in zip(positions, all_blocks_of_type):
            block.set_position(pos)

class ConstantInitialization(InitializationType):

    def get_unique_name(self):
        return 'constant'

    def _initialize(self, all_blocks_of_type, env_parameters):
        pass


class AgentInitialization(InitializationType):

    def get_unique_name(self):
        return 'agent'

    def _initialize(self, all_blocks_of_type, env_parameters):
        assert len(all_blocks_of_type) == 1
        positions = get_nonintersecting_positions(env_parameters['grid_size'], len(all_blocks_of_type), env_parameters['initialized_positions'])

        all_blocks_of_type[0].set_position(positions[0])





