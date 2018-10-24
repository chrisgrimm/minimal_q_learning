import subprocess

class SafeSubprocess(object):

    def __init__(self):
        self.process = subprocess.Popen(['/bin/bash'], stdin=subprocess.PIPE, bufsize=0, stdout=subprocess.PIPE, shell=True)

    def push_command(self, command_list, finish_string='COMMAND IS FINISHED'):
        all_output = ''
        current_output = None
        command = ' '.join(command_list)
        full_string = f'{command} & wait; echo "{finish_string}"\n'
        self.process.stdin.write(full_string.encode())
        while current_output != finish_string:
            current_output = self.process.stdout.readline().decode('utf-8').strip()
            if current_output != finish_string:
                all_output += (current_output + '\n')
            else:
                return all_output





if __name__ == '__main__':
    safe = SafeSubprocess()
    print(safe.push_command(['echo', '"dog"']))
    print(safe.push_command(['echo', '"cat"']))
    #safe.push_command(['touch', 'zz'])