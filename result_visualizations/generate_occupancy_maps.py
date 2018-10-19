import cv2
import numpy as np
import os
import re
import tqdm
from utils import horz_stack_images
from envs.atari.pacman import PacmanWrapper, AssaultWrapper, QBertWrapper, SeaquestWrapper, BreakoutWrapper, AtariWrapper
from baselines.deepq.experiments.training_wrapper import QNetworkTrainingWrapper


def match_by_color(frame: np.ndarray, colors, threshold):
    last_frame = np.copy(frame[:, :, -3:])
    result = np.zeros(last_frame.shape, dtype=np.uint8)
    for color in colors:
        result[np.where((np.square(last_frame/255. - np.array(color)/255.) < threshold).all(axis=2))] = 255
    return result



pacman_colors = [
    (186, 149, 81),
    (210, 164, 74),
    (166, 136, 87),
    (199, 158, 94),
    (193, 155, 95),
    (165, 136, 96),
    (153, 129, 97),
    (186, 150, 95),
    (128, 112, 100),
]
pacman_threshold = 0.003

assault_colors = [
    #(65, 89, 69),
    (146, 122, 95),
    (112, 82, 49),
    (115, 101, 84),
    (141, 103, 60),
]
assault_threshold = 0.003

qbert_colors = [
    (175, 86, 44),
]
qbert_threshold = 0.005
def compute_policy_histogram(agent: QNetworkTrainingWrapper, env: AtariWrapper, target_colors, target_threshold):
    s = env.reset()
    histogram = np.zeros(shape=env.observation_space.shape[:2], dtype=np.float32)
    n = 0
    before_start = 0
    num_steps = 100000
    for i in tqdm.tqdm(range(num_steps + before_start)):
        a = agent.get_action([s])[0]
        s, r, t, info = env.step(a)
        if i < before_start:
            continue
        match = match_by_color(s, target_colors, target_threshold).astype(np.float32)[:,:,0] / 255.
        histogram += match
        n += 1
    z_score = (histogram - np.mean(histogram)) / np.std(histogram)
    normed = z_score - np.min(z_score) / (np.max(z_score) - np.min(z_score))
    histogram = 255*np.tile(np.reshape(normed, list(histogram.shape[:2]) + [1]), [1,1,3])
    cv2.imwrite('histogram_test.png', histogram)

class RandomAgent():
    def __init__(self, num_actions):
        self.n = num_actions

    def get_action(self, s):
        return [np.random.randint(0, self.n)]

def test_color_getter(env, colors, threshold):
    #threshold, colors = (pacman_threshold, pacman_colors)
    #env = PacmanWrapper()
    s = env.reset()
    while True:
        s, r, t, info = env.step(np.random.randint(0, env.action_space.n))
        # cv2.imwrite('sample_image.png', s[:, :, -3:])
        match = match_by_color(s, colors, threshold)
        match = cv2.resize(match, (64, 64))
        s = cv2.resize(s[:, :, -3:], (64, 64))
        both = horz_stack_images(match, s)
        cv2.imwrite('match.png', both)
        input('...')
        #cv2.imshow('match.png', both)

        # stacked = horz_stack_images(match, s[:, :, 6:9], background_color=(255,0,0))
        # cv2.imshow('game', match)
        #cv2.waitKey(1)


if __name__ == '__main__':
    env = QBertWrapper()
    colors = qbert_colors
    threshold = qbert_threshold
    compute_policy_histogram(RandomAgent(env.action_space.n), env, colors, threshold)
    #test_color_getter(env, colors, threshold)
