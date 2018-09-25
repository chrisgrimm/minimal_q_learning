import numpy as np
import cv2
import os
import re
import matplotlib.pyplot as plt

def produce_maximum_image():

    def max_over_set(images):
        canvas = np.zeros(shape=[400,400], dtype=np.uint8)
        for image in images:
            one_chan = np.max(image, axis=2)
            canvas = np.maximum(canvas, one_chan)
        cv2.imwrite('max_image2.png', np.tile(np.reshape(canvas, [400,400,1]), [1,1,3]))

    path = './policy_data/pi_2'

    all_files = [os.path.join(path, f'{i}_last.png') for i in range(10)]
    all_images = [cv2.imread(f) for f in all_files]
    max_over_set(all_images)



path = './runs/assault_statistics'
def extract_number(file_name):
    match_obj = re.match(r'^policy\_vis\_(\d+)\_statistics.txt$', file_name)
    if match_obj:
        (number,) = match_obj.groups()
        return int(number)
    else:
        raise Exception('No match')



def extract_all_number_like(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    text = text.replace('\n','')

    raw_numbers = [float(x) for x in re.findall(r'([\de\+\-\.]+)', text)]
    print(file_path, raw_numbers)
    policy1, policy2, policy3 = raw_numbers[:4], raw_numbers[4:8], raw_numbers[8:12]
    assert len(policy1) == len(policy2) == len(policy3)
    return policy1[1:], policy2[1:], policy3[1:]

def compute_difference(x, policy_num):
    onehot = np.zeros([3])
    onehot[policy_num] = 1
    return 0.5*np.sum(np.abs(x - onehot))


sorted_files = [os.path.join(path, x) for x in sorted(os.listdir(path), key=extract_number)]
numbers0, numbers1, numbers2 = zip(*[extract_all_number_like(path) for path in sorted_files])
differences0 = [compute_difference(n, 0) for n in numbers0]
differences1 = [compute_difference(n, 1) for n in numbers1]
differences2 = [compute_difference(n, 2) for n in numbers2]

n = 6
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

x =[i*100 for i in range(len(differences0))]
plt.hold(True)
plt.plot(x[n-1:], moving_average(differences0, n=n), 'r', label='Policy 0')
plt.plot(x[n-1:], moving_average(differences1, n=n), 'g', label='Policy 1')
plt.plot(x[n-1:], moving_average(differences2, n=n), 'b', label='Policy 2')
plt.legend(loc='upper right')
plt.savefig('differences.pdf')