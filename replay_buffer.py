import numpy as np

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.S = [None for _ in range(capacity)]
        self.A = [None for _ in range(capacity)]
        self.R = [None for _ in range(capacity)]
        self.SP = [None for _ in range(capacity)]
        self.T = [None for _ in range(capacity)]
        self.index = 0
        self.is_full = False

    def append(self, s, a, r, sp ,t):
        self.S[self.index] = s
        self.A[self.index] = a
        self.R[self.index] = r
        self.SP[self.index] = sp
        self.T[self.index] = t
        self.index += 1
        if self.index == self.capacity:
            self.index = 0
            self.is_full = True

    def sample(self, batch_size):
        indices = np.random.randint(0, self.length(), size=batch_size)
        S, A, R, SP, T = [], [], [], [], []
        for idx in indices:
            S.append(self.S[idx])
            A.append(self.A[idx])
            R.append(self.R[idx])
            SP.append(self.SP[idx])
            T.append(self.T[idx])
        return S, A, R, SP, T

    def length(self):
        if self.is_full:
            return self.capacity
        else:
            return self.index


class ReplayBuffer2(object):

    def __init__(self, capacity, num_frames, num_color_channels):
        self.color_channels = num_color_channels
        # things break in the append function if there is only 1 color channel.
        assert self.color_channels > 1
        self.capacity = capacity
        self.num_frames = num_frames
        self.S = [None for _ in range(self.capacity)]
        self.A = [None for _ in range(self.capacity)]
        self.R = [None for _ in range(self.capacity)]
        self.T = [None for _ in range(self.capacity)]
        self.idx = 0
        self.is_full = False
        self.test_mode = False

    def append(self, s, a, r, sp, t):
        if self.test_mode:
            self.S[self.idx] = sp
        else:
            self.S[self.idx] = sp[:, :, -self.color_channels:]
        self.A[self.idx] = a
        self.R[self.idx] = r
        self.T[self.idx] = t
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.is_full = True

    def rejection_sample_indices(self, num_indices):
        indices = []
        max_sample_val = self.capacity if self.is_full else self.idx
        min_sample_val = 0 if self.is_full else self.num_frames + 1
        num_back = self.num_frames + 1
        while len(indices) < num_indices:
            idx = np.random.randint(min_sample_val, max_sample_val)
            # checks if the self.idx is going to be crossed by the sample.
            # "addition" handles the edge case when idx and idx - num_frames crosses 0 by sliding everything forward.
            # this makes checking for "between-ness" on a ring easier.
            addition = num_back if idx - num_back < 0 else 0
            upper_bound = (idx + addition) % self.capacity
            lower_bound = (idx + addition - num_back) % self.capacity
            boundary = (self.idx + addition) % self.capacity
            if upper_bound > boundary > lower_bound:
                #print(f'rejected {idx}!')
                continue
            indices.append(idx)
        return indices

    def get_S_slice(self, i0, i1):

        if i0 < 0 and i1 >= 0:
            slice = self.S[i0:] + self.S[:i1]
        else:
            slice = self.S[i0:i1]
        # if len(slice) != self.num_frames:
        #     print(f'failed!, {i0}, {i1}')
        #     input('...')
        if self.test_mode:
            return slice
        else:
            print(i0, i1)
            #print(slice)

            return np.concatenate(slice, axis=2)



    def sample(self, batch_size):
        S = []
        A = []
        R = []
        SP = []
        T = []
        for idx in self.rejection_sample_indices(batch_size):
            S.append(self.get_S_slice(idx-1-self.num_frames, idx-1))
            SP.append(self.get_S_slice(idx-self.num_frames, idx))
            A.append(self.A[idx-1])
            R.append(self.R[idx-1])
            T.append(self.T[idx-1])
        return S, A, R, SP, T

    def length(self):
        if self.is_full:
            return self.capacity
        else:
            return self.idx











class StateReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.state = [None for _ in range(capacity)]
        self.index = 0
        self.is_full = False

    def append(self, state):
        self.state[self.index] = state
        self.index += 1
        if self.index == self.capacity:
            self.index = 0
            self.is_full = True

    def sample(self, batch_size):
        indices = np.random.randint(0, self.length(), size=batch_size)
        STATE = []
        for idx in indices:
            STATE.append(self.state[idx])
        return STATE

    def length(self):
        if self.is_full:
            return self.capacity
        else:
            return self.index


if __name__ == '__main__':
    buff = ReplayBuffer2(100, 4)
    buff.test_mode = True

    for i in range(100):
        buff.append(f'S{i}', f'A{i}', f'R{i+1}', f'SP{i+1}', f'T{i+1}')
    #buff.is_full = True
    while True:
        print(buff.sample(1))