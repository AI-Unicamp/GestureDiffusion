import numpy as np
import utils.rotation_conversions as geometry
import bvhsdk
from scipy.signal import savgol_filter


def get_indexes(dataset):
    n_joints = 83
    if dataset == 'genea2023':
        idx_positions = np.asarray([ [i*6+3, i*6+4, i*6+5] for i in range(n_joints) ]).flatten()
        idx_rotations = np.asarray([ [i*6, i*6+1, i*6+2] for i in range(n_joints) ]).flatten()
    elif dataset == 'genea2023+':
        idx_positions = np.asarray([ [i*9+6, i*9+7, i*9+8] for i in range(n_joints) ]).flatten()
        idx_rotations = np.asarray([ [i*9, i*9+1, i*9+2, i*9+3, i*9+4, i*9+5] for i in range(n_joints) ]).flatten()  
    else:
        raise NotImplementedError("This dataset is not implemented.")
    return idx_positions, idx_rotations

def split_pos_rot(dataset, data):
    # Split the data into positions and rotations
    # Shape expected [num_samples(bs), 1, chunk_len, 1245 or 498]
    # Output shape [num_samples(bs), 1, chunk_len, 498 or 249]
    idx_positions, idx_rotations = get_indexes(dataset)
    return data[..., idx_positions], data[..., idx_rotations]

def rot6d_to_euler(data):
    # Convert numpy array to euler angles
    # Shape expected [num_samples(bs), 1, chunk_len, 498]
    # Output shape [num_samples(bs) * chunk_len, n_joints, 3]
    n_joints = 83
    assert data.shape[-1] == n_joints*6
    sample_rot = sample_rot.view(sample_rot.shape[:-1] + (-1, 6))                   # [num_samples(bs), 1, chunk_len, n_joints, 6]
    sample_rot = geometry.rotation_6d_to_matrix(sample_rot)                         # [num_samples(bs), 1, chunk_len, n_joints, 3, 3]
    sample_rot = geometry.matrix_to_euler_angles(sample_rot, "ZXY")[..., [1, 2, 0] ]*180/np.pi # [num_samples(bs), 1, chunk_len, n_joints, 3]
    sample_rot = sample_rot.view(-1, *sample_rot.shape[2:]).permute(0, 2, 3, 1)     # [num_samples(bs), n_joints, 3, chunk_len]

def tobvh(bvhreference, rotation, position=None):
    # Converts to bvh format
    # Shape expected  [njoints, 3, frames]
    # returns a bvh object
    rotation = rotation.transpose(2, 0, 1) # [frames, njoints, 3]
    bvhreference.frames = rotation.shape[0]
    for j, joint in enumerate(bvhreference.getlistofjoints()):
        joint.rotation = rotation[:, j, :]
        joint.translation = np.tile(joint.offset, (bvhreference.frames, 1))
    if position:
        position = position.transpose(2, 0, 1) # [frames, njoints, 3]
        bvhreference.root.translation = position[:, 0, :]
    return bvhreference

def posfrombvh(bvh):
    # Extracts positions from bvh
    # returns a numpy array shaped [frames, njoints, 3]
    position = np.zeros((bvh.frames, bvh.njoints * 3))
    # This way takes advantage of the implementarion of getPosition (16.9 seconds ~4000 frames)
    for frame in range(bvh.frames):
        for i, joint in enumerate(bvh.getlistofjoints()):
            position[frame, i*3:i*3+3] = joint.getPosition(frame)
    return position


def filter_and_interp(rotation, position, num_frames=120, chunks=None):
    # Smooth chunk transitions
    n_chunks = chunks if chunks else int(rotation.shape[-1]/num_frames)
    inter_range = 10 #interpolation range in frames
    for transition in np.arange(1, n_chunks-1)*num_frames:
        all_motions[..., transition:transition+2] = np.tile(np.expand_dims(all_motions[..., transition]/2 + all_motions[..., transition-1]/2,-1),2)
        all_motions_rot[..., transition:transition+2] = np.tile(np.expand_dims(all_motions_rot[..., transition]/2 + all_motions_rot[..., transition-1]/2,-1),2)
        for i, s in enumerate(np.linspace(0, 1, inter_range-1)):
            forward = transition-inter_range+i
            backward = transition+inter_range-i
            all_motions[..., forward] = all_motions[..., forward]*(1-s) + all_motions[:, :, :, transition-1]*s  
            all_motions[..., backward] = all_motions[..., backward]*(1-s) + all_motions[:, :, :, transition]*s
            all_motions_rot[..., forward] = all_motions_rot[..., forward]*(1-s) + all_motions_rot[:, :, :, transition-1]*s
            all_motions_rot[..., backward] = all_motions_rot[..., backward]*(1-s) + all_motions_rot[:, :, :, transition]*s
            
    all_motions = savgol_filter(all_motions, 9, 3, axis=-1)
    all_motions_rot = savgol_filter(all_motions_rot, 9, 3, axis=-1)

    return all_motions, all_motions_rot