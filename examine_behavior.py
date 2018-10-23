import cv2, numpy as np, pickle
import tempfile, os, shutil
from utils import build_directory_structure
from safe_subprocess import SafeSubprocess

#file_name = 'policy_vis_55000_behavior_file.pickle' # assault
#file_name = './runs/statistics_dir/policy_vis_6200_behavior_file.pickle' # pacman
#file_name = './runs/statistics_dir/policy_vis_10000_behavior_file.pickle' # qbert
#file_name = './runs/statistics_dir/policy_vis_5100_behavior_file.pickle' # pacman traj20 3part
#file_name = './runs/statistics_dir/policy_vis_20600_behavior_file.pickle' #good qbert part2_traj10
#file_name = './runs/statistics_dir/policy_vis_24000_behavior_file.pickle' #decent pacman part2_traj10
#file_name = './qbert_final/part2/policy_vis_21600_behavior_file.pickle'
#file_name = './pacman_final/part2/policy_vis_29200_behavior_file.pickle'
#file_name = './alien_final/part2/policy_vis_8800_behavior_file.pickle'
#file_name = './assault_final/part2/policy_vis_17600_behavior_file.pickle'


# takes in a video_name, an extension, and the trajectory that it is going to make the video from.
ffmpeg_subprocess = SafeSubprocess()

def produce_video_from_traj(video_name, ext, traj):

    directory = tempfile.mkdtemp()
    try:
        for idx, image in enumerate(traj):
            print(image[:,:,6:9].shape)
            cv2.imwrite(os.path.join(directory, f'{idx}.png'), cv2.resize(image[:, :, 6:9], (400, 400), interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(video_name+'_last.png', cv2.resize(traj[-1][:,:,6:9], (400,400), interpolation=cv2.INTER_NEAREST))
        #subprocess.call(['ffmpeg', '-y', '-i', os.path.join(directory, '%d.png'), video_name+'.'+ext])
        ffmpeg_subprocess.push_command(['ffmpeg', '-y', '-i', os.path.join(directory, '%d.png'), video_name+'.'+ext])

    finally:
        shutil.rmtree(directory)
    cv2.destroyAllWindows()

def produce_all_videos(root_dir, file_number):
    file_name = os.path.join(root_dir, f'policy_vis_{file_number}_behavior_file.pickle')
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    build_directory_structure(root_dir, {f'{file_number}':
                                        {f'pi_{i}': {} for i in range(len(data))}})
    for policy_num, policy_trajectories in enumerate(data):
        for traj_num, trajectory in enumerate(policy_trajectories):
            name = os.path.join(root_dir, f'{file_number}/pi_{policy_num}/{traj_num}')
            produce_video_from_traj(name, 'avi', trajectory)



if __name__ == '__main__':
    root_dir = './pacman_new_dqn_5x_freq/part2/'
    file_number = '6867000'
    produce_all_videos(root_dir, file_number)
