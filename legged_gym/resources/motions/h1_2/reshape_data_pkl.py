import pickle
import dill
import numpy as np

from tqdm import tqdm

from scipy.spatial.transform import Rotation as R

def joint_pos_to_pose(joint_pos_traj, root_rot_traj):
    len_traj = joint_pos_traj.shape[0]
    pose_aa = np.zeros((len_traj, 28, 3), dtype=np.float32)

    pose_aa[:, 0, :] = R.from_quat(root_rot_traj).as_rotvec()
    # left leg
    pose_aa[:, 1, 2] = joint_pos_traj[:, 0]
    pose_aa[:, 2, 1] = joint_pos_traj[:, 1]
    pose_aa[:, 3, 0] = joint_pos_traj[:, 2]
    pose_aa[:, 4, 1] = joint_pos_traj[:, 3]
    pose_aa[:, 5, 1] = joint_pos_traj[:, 4]
    pose_aa[:, 6, 0] = joint_pos_traj[:, 5]
    # right leg
    pose_aa[:, 7, 2] = joint_pos_traj[:, 6]
    pose_aa[:, 8, 1] = joint_pos_traj[:, 7]
    pose_aa[:, 9, 0] = joint_pos_traj[:, 8]
    pose_aa[:, 10, 1] = joint_pos_traj[:, 9]
    pose_aa[:, 11, 1] = joint_pos_traj[:, 10]
    pose_aa[:, 12, 0] = joint_pos_traj[:, 11]
    # torso
    pose_aa[:, 13, 2] = joint_pos_traj[:, 12]
    # left arm
    pose_aa[:, 14, 1] = joint_pos_traj[:, 13]
    pose_aa[:, 15, 0] = joint_pos_traj[:, 14]
    pose_aa[:, 16, 2] = joint_pos_traj[:, 15]
    pose_aa[:, 17, 1] = joint_pos_traj[:, 16]
    pose_aa[:, 18, 0] = joint_pos_traj[:, 17]
    pose_aa[:, 19, 1] = joint_pos_traj[:, 18]
    pose_aa[:, 20, 2] = joint_pos_traj[:, 19]
    # right arm
    pose_aa[:, 21, 1] = joint_pos_traj[:, 20]
    pose_aa[:, 22, 0] = joint_pos_traj[:, 21]
    pose_aa[:, 23, 2] = joint_pos_traj[:, 22]
    pose_aa[:, 24, 1] = joint_pos_traj[:, 23]
    pose_aa[:, 25, 0] = joint_pos_traj[:, 24]
    pose_aa[:, 26, 1] = joint_pos_traj[:, 25]
    pose_aa[:, 27, 2] = joint_pos_traj[:, 26]

    return pose_aa

with open('motion_cmu_retarget_h1m_new.pkl', 'rb') as input:
    data = pickle.load(input)
    # Now the input 'data' is a list of data for teleoperation.

print(len(data))
for i in range(len(data)):
    print(len(data[i]))
    length = len(data[i])

output_data = dict()

for num in tqdm(range(length)):
    # print(data[0][num].keys())
    # print(data[1][num])
    # print(data[2][num])
    # print(data[3][num])
    # print(data[4][num])

    # for key in data[0][num].keys():
    #     try:
    #         print(key, data[0][num][key].shape)
    #     except:
    #         print(key, data[0][num][key])
    
    key_slice = 'CMU_' + data[0][num]['file_name']

    data_slice = dict()
    data_slice['root_trans_offset'] = data[0][num]['root_pos']
    data_slice['dof'] = data[0][num]['dof_pos']
    data_slice['root_rot'] = data[0][num]['root_rot']
    data_slice['fps'] = data[0][num]['fps']
    data_slice['smpl_joints'] = None

    pose_aa = joint_pos_to_pose(data[0][num]['dof_pos'], data[0][num]['root_rot'])
    data_slice['pose_aa'] = pose_aa

    print("###########################################")
    print(key_slice)
    print(data_slice.keys())
    for key in data_slice.keys():
        try:
            print(key, data_slice[key].shape)
        except:
            print(key, data_slice[key])

    output_data[key_slice] = data_slice

import ipdb; ipdb.set_trace()
with open('cmu_retarget_test.pkl', 'wb') as output:
    pickle.dump(output_data, output, pickle.HIGHEST_PROTOCOL)