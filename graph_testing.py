import numpy as np
import tqdm

nodes = [1,2,3,4]
edges = {
    1: [2,3,4],
    2: [1,3],
    3: [1,2]
}

def do_walk(start_at, stop_at):
    node = start_at
    steps = 0
    while node != stop_at:
        node = np.random.choice(edges[node])
        steps += 1
    return steps

all_steps = []
for i in tqdm.tqdm(range(10000)):
    steps = do_walk(np.random.choice([1,2,3]), 4)
    all_steps.append(steps)

print('steps', np.mean(all_steps))

