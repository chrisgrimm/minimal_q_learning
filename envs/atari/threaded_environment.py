import multiprocessing
from multiprocessing import Pool, Pipe, Process
import numpy as np
import gym
from envs.atari.simple_assault import SimpleAssault



class ThreadedEnvironment(object):


    def __init__(self, num_instances, env_constructor, env_class):
        self.num_instances = num_instances
        self.env_constructor = env_constructor
        self.env_class = env_class
        self.env_array_pipes = [Pipe() for _ in range(num_instances)]
        self.process_array = [self.build_environment_thread(i, env_constructor, child_connection)
                              for i, (parent_connection, child_connection) in enumerate(self.env_array_pipes)]

        self.start_processes()


    def start_processes(self):
        [process.start() for process in self.process_array]


    def stop_processes(self):
        [parent_connection.send(('__kill__', [])) for (parent_connection, child_connection) in self.env_array_pipes]



    def __call__(self, attribute, args=None, sharded_args=None):
        if (args is None) and (sharded_args is None):
            raise Exception('args and sharded_args cannot both be None.')
        if (args is not None) and (sharded_args is not None):
            raise Exception('One of args or sharded_args must be None.')
        if (sharded_args is not None) and (len(sharded_args) != self.num_instances):
            raise Exception(f'List of argument lists passed to sharded_args must be of length {self.num_instances}, received list of length len(sharded_args)')

        class_items = dir(self.env_class)
        if attribute not in class_items:
            raise Exception(f'Attribute {attribute} is not a member of class {self.env_class}')
        # dispatch tasks
        if sharded_args is None:
            [parent_connection.send((attribute, args))
             for (parent_connection, child_connection) in self.env_array_pipes]
        else:
            [parent_connection.send((attribute, _args))
             for (_args, (parent_connection, child_connection)) in zip(sharded_args, self.env_array_pipes)]

        # receive responses
        response = [parent_connection.recv() for (parent_connection, child_connection) in self.env_array_pipes]
        for r in response:
            if isinstance(r, Exception):
                raise r
        return response



    def build_environment_thread(self, i, env_constructor, connection):
        def process_function(connection):
            env = env_constructor(i)
            while True:
                attribute, args = connection.recv()
                if attribute == '__kill__':
                    connection.close()
                    break
                else:
                    selected_attribute = env.__getattribute__(attribute)
                    try:
                        if callable(selected_attribute):
                            result = selected_attribute(*args)
                        else:
                            result = selected_attribute
                        connection.send(result)
                    except Exception as e:
                        connection.send(e)


        return Process(target=process_function, args=(connection,))


if __name__ == '__main__':
    #sandwich = ThreadedEnvironment(2, lambda i: SimpleAssault(initial_states_file='stored_states_64.pickle'), SimpleAssault)
    #out1 = sandwich('reset', args=[])
    #out2 = sandwich('step', sharded_args=[[0], [1]])
    #out3 = sandwich('determine_ship_states', args=[])
    #print(out3)
    #sandwich.stop_processes()
    print(multiprocessing.cpu_count())





