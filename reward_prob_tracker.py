import numpy as np
from typing import List
from envs.atari.simple_assault import SimpleAssault
import cv2
import os

class PixelHistogram(object):

    def __init__(self):
        self.histogram = np.zeros([256], dtype=np.int32)
        self.histogram_denom = 0

    def prob_hist(self) -> float:
        return self.histogram / float(self.histogram_denom)

    def add(self, value: int):
        self.histogram[value] += 1
        self.histogram_denom += 1

class RewardProbTracker(object):

    def __init__(self, image_height, image_width, image_depth):
        self.h, self.w, self.d = image_height, image_width, image_depth
        self.positive_reward = self.build_pixel_histogram_array(image_height, image_width, image_depth)
        self.zero_reward = self.build_pixel_histogram_array(image_height, image_width, image_depth)

    def build_pixel_histogram_array(self, image_height, image_width, image_depth) -> List[List[List[PixelHistogram]]]:
        h_array = []
        for h in range(image_height):
            w_array = []
            for w in range(image_width):
                d_array = []
                for d in range(image_depth):
                    d_array.append(PixelHistogram())
                w_array.append(d_array)
            h_array.append(w_array)
        return h_array

    def add(self, image: np.ndarray, reward: int):
        assert reward in [0, 1]
        hist_array = self.positive_reward if reward == 1 else self.zero_reward
        for h in range(self.h):
            for w in range(self.w):
                for d in range(self.d):
                    hist_array[h][w][d].add(image[h][w][d])

    def prob_hist(self, h: int, w: int, d: int, reward: int) -> float:
        assert reward in [0, 1]
        if reward == 0:
            return self.zero_reward[h][w][d].prob_hist()
        else:
            return self.positive_reward[h][w][d].prob_hist()

    def compute_threshold_image(self, threshold: float) -> np.ndarray:
        threshold_image = np.zeros(shape=[self.h, self.w, self.d], dtype=np.uint8)
        #diff_image = np.zeros(shape=[self.h, self.w, self.d], dtype=np.float32)
        for h in range(self.h):
            for w in range(self.w):
                for d in range(self.d):
                    td = np.sum(np.abs(self.prob_hist(h,w,d,0) - self.prob_hist(h,w,d,1)))
                    #diff_image[h,w,d] = td
                    if td >= threshold:
                        threshold_image[h, w, d] = 1
        return threshold_image

def compute_threshold_image(tracker, threshold):
    image = tracker.compute_threshold_image(threshold)
    image = np.tile(np.max(image, axis=2, keepdims=True), [1, 1, 3])
    return image

if __name__ == '__main__':
    env = SimpleAssault(initial_states_file='stored_states_64.pickle')
    env.reset()
    print('building...')
    tracker = RewardProbTracker(64, 64, 3)
    print('finished building!')
    min_each = 500
    for desired_reward in [0, 1]:
        num_reward = 0
        while num_reward < min_each:
            sp, r, t, _ = env.step(np.random.randint(env.action_space.n))
            if r == desired_reward:
                tracker.add(sp[:, :, :], r)
                num_reward += 1
                print(f'{num_reward}/{min_each}')
            if t:
                env.reset()

    for threshold in np.linspace(0.05, 0.2, num=20):
        image = compute_threshold_image(tracker, threshold)
        cv2.imwrite(os.path.join('./prob_tracker_images', f'{threshold}.png'), image)




