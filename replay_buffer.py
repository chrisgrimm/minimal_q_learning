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
