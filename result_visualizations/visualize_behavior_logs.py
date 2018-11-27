from examine_behavior import produce_all_videos

import os, numpy as np, pickle, re
import cv2

def visualize_value_matrix(base_path, number):
    matrix_path = os.path.join(base_path, f'policy_vis_{number}_value_matrix.pickle')
    with open(matrix_path, 'rb') as f:
        data = pickle.load(f)
    image = cv2.resize(data, (400, 400), interpolation=cv2.INTER_NEAREST)
    image = (255*image).astype(np.uint8)
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    save_path = os.path.join(base_path, f'viz_value_matrix_{number}.png')
    cv2.imwrite(save_path, image)


def process_run(base_path):
    files = os.listdir(base_path)
    all_numbers = set()
    for f in files:
        match = re.match(r'^policy\_vis\_(\d+).+?$', f)
        if not match:
            continue
        (number,) = match.groups()
        all_numbers.add(number)
    for number in all_numbers:
        visualize_value_matrix(base_path, number)
        produce_all_videos(base_path, number)

def process_all_runs():
    behavior_log_path = '/Users/chris/projects/q_learning/new_dqn_results/completed_runs/behavior_logs'
    rldls = [11]
    rldls = [f'rldl{x}' for x in rldls]
    all_base_paths = []
    for rldl in rldls:
        base_paths = [x for x in os.listdir(os.path.join(behavior_log_path, rldl))
                      if re.match(r'^.+?reward.+?$', x)]
        base_paths = [os.path.join(behavior_log_path, rldl, x, 'images') for x in base_paths]
        all_base_paths += base_paths
    for base_path in all_base_paths:
        process_run(base_path)

if __name__ == '__main__':
    process_all_runs()


#visualize_value_matrix('/Users/chris/projects/q_learning/new_dqn_results/completed_runs/behavior_logs/rldl11/assault_2reward_10mult_4/images', 9950000)
