import numpy as np
n = 8
P = np.zeros([n, n], dtype=np.float32)
for i in range(n):
    P[i, (i-1) % n] = 1/3
    P[i, (i+1) % n] = 1/3
    P[i, i] = 1/3


Pbar = np.zeros([n+1,n+1], dtype=np.float32)
Pbar[:n, :n] = P
Pbar[0, 0] = 1/4
Pbar[0, -1 % n] = 1/4
Pbar[0, 1 % n] = 1/4
Pbar[0, n] = 1/4
Pbar[n, :n] = 1/n


def compute_stationary(P):
    S = P
    for i in range(8000):
        S = S @ P
    return S


print(compute_stationary(P))
print(compute_stationary(Pbar))