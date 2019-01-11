import numpy as np
from reward_network import RewardPartitionNetwork
from envs.atari.atari_wrapper import PacmanWrapper, AssaultWrapper, SeaquestWrapper, AtariWrapper
import re, os, tqdm, sys
from pyemd import emd

def build_reward_net(env, run_dir, run_name):
    num_rewards = int(re.match(r'^.+?(\d+)reward.+?$', run_name).groups()[0])
    weights_path = os.path.join(run_dir, run_name, 'best_weights')
    reward_net = RewardPartitionNetwork(env, None, None, num_rewards, env.observation_space.shape[0],
                                        env.action_space.n, 'reward_net', traj_len=10, gpu_num=-1,
                                        use_gpu=False, num_visual_channels=9, visual=True)
    reward_net.restore(weights_path, 'reward_net.ckpt')
    return reward_net

def earth_movers(p, q):
    # computes the earth mover's distance
    assert p.shape == q.shape
    n = p.shape[0]
    X = np.array(list(range(n))).astype(np.float64)
    Y = np.array(list(range(n))).astype(np.float64)
    XX, YY = np.meshgrid(X, Y)
    dist_mat = np.abs(XX - YY)
    #print(dist_mat)
    return emd(p, q, dist_mat)

def distance_from_saturated(p):
    n = p.shape[0]
    dist = np.inf
    for i in range(n):
        q = np.zeros([n], dtype=np.float64)
        q[i] = 1.0
        dist = min(earth_movers(p, q), dist)
    return dist

def distance_from_saturated2(p):
    p = p
    n = p.shape[0]
    return (np.max(p) - 1/n) / (1 - 1/n)



def collect_reward_stats_for_trajectory(env: AtariWrapper, reward_net: RewardPartitionNetwork):
    num_rewarding = 0
    required_rewarding = 1000
    all_saturations = []
    s = env.reset()
    pbar = tqdm.tqdm(total=required_rewarding)
    while num_rewarding < required_rewarding:
        a = np.random.randint(0, env.action_space.n)
        s, r, t, info = env.step(a)
        if r != 0:
            partitioned_reward = reward_net.get_reward(s, r)
            distr = (partitioned_reward / np.sum(partitioned_reward)).astype(np.float64)
            saturation = distance_from_saturated2(distr)
            all_saturations.append(saturation)
            num_rewarding += 1
            pbar.update(1)
    # down here we need to collect the statistics.
    return np.mean(all_saturations)

game_mapping = {
    'assault': AssaultWrapper,
    'pacman': PacmanWrapper,
    'seaquest': SeaquestWrapper
}

def collect_reward_stats_for_single(run_dir, run_name):
    try:
        game = re.match(r'^(.+?)\_\d.+?$', run_name).groups()[0]
    except AttributeError:
        print(f'{run_name} didnt match regex.')
        return
    env = game_mapping[game]()
    reward_net = build_reward_net(env, run_dir, run_name)
    saturation_score = collect_reward_stats_for_trajectory(env, reward_net)
    with open('./saturation_results.txt', 'a') as f:
        print(run_name, saturation_score, file=f)
    print(run_name, saturation_score)

def make_command(run_dir, regex):
    matched_runs = [x for x in os.listdir(run_dir) if re.match(regex, x)]

    path_variables = [
        '~/q_learning',
        '~/baselines'
    ]
    preamble = f'PYTHONPATH={":".join(path_variables)} '
    all_commands = []
    for run_name in matched_runs:
        file_path = os.path.abspath(__file__)
        command = preamble + f'python {file_path} {run_dir} {regex} {run_name}'
        all_commands.append(command)
    print('; '.join(all_commands))



if __name__ == '__main__':
    run_dir, regex, run_name = sys.argv[1], sys.argv[2], sys.argv[3]
    if run_name == 'make_command':
        make_command(run_dir, regex)
    else:
        collect_reward_stats_for_single(run_dir, run_name)

