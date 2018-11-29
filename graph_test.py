import numpy as np
n = 1000
P = np.zeros([n, n], dtype=np.float32)
for i in range(n):
    P[i, (i-1) % n] = 1/2
    P[i, (i+1) % n] = 1/2

Pbar = np.zeros([n+1,n+1], dtype=np.float32)
Pbar[:n, :n] = P
Pbar[0, -1 % n] = 1/3
Pbar[0, 1 % n] = 1/3
Pbar[0, n] = 1/3
Pbar[n, :n] = 1/n
print(Pbar)


def compute_stationary(P):
    S = P
    for i in range(1000):
        print(i)
        S = S @ P
    return S



print(compute_stationary(Pbar))