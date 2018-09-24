import cv2, numpy as np, pickle
import tempfile, os, shutil, subprocess
from utils import build_directory_structure

file_name = 'policy_vis_39700_behavior_file.pickle'
with open(file_name, 'rb') as f:
    data = pickle.load(f)

def produce_video_from_traj(video_name, ext, traj):

    directory = tempfile.mkdtemp()
    print(directory)
    try:
        for idx, image in enumerate(traj):
            print(image[:,:,6:9].shape)
            cv2.imwrite(os.path.join(directory, f'{idx}.png'), cv2.resize(image[:, :, 6:9], (400, 400), interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(video_name+'_last.png', cv2.resize(traj[-1][:,:,6:9], (400,400), interpolation=cv2.INTER_NEAREST))
        subprocess.call(['ffmpeg', '-y', '-i', os.path.join(directory, '%d.png'), video_name+'.'+ext])

    finally:
        shutil.rmtree(directory)

    cv2.destroyAllWindows()

def produce_all_videos(data):
    build_directory_structure('.', {'policy_data':
                                        {'pi_0': {},
                                         'pi_1': {},
                                         'pi_2': {}}})
    for policy_num, policy_trajectories in enumerate(data):
        for traj_num, trajectory in enumerate(policy_trajectories):
            name = f'policy_data/pi_{policy_num}/{traj_num}'
            produce_video_from_traj(name, 'avi', trajectory)

produce_all_videos(data)
#produce_video_from_traj('./video.avi', data[0][0])