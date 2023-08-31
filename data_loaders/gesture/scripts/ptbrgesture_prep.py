from argparse import ArgumentParser
import os
from data_loaders.gesture.scripts.motion_process import bvh2representations2 as bvh2npy
import librosa

def main(args):
    bvhpath, wavpath, rot6dpath, rot3dpath, pospath, npy16k = paths_get_and_check(args.data_dir)
    takes = takes_get_and_check(bvhpath, wavpath)
    print(takes)
    

def paths_get_and_check(data_dir):
    assert os.path.exists(data_dir), 'Data directory does not exist'
    motionpath = os.path.join(data_dir, 'motion')
    audiopath = os.path.join(data_dir, 'audio')
    bvhpath = os.path.join(motionpath, 'bvh')
    wavpath = os.path.join(audiopath, 'wav')
    assert os.path.exists(motionpath), 'Motion directory does not exist'
    assert os.path.exists(audiopath), 'Audio directory does not exist'
    assert os.path.exists(bvhpath), 'BVH directory does not exist'
    assert os.path.exists(wavpath), 'WAV directory does not exist'
    rot6dpath = os.path.join(motionpath, 'rot6d')
    rot3dpath = os.path.join(motionpath, 'rot3d')
    pospath = os.path.join(motionpath, 'pos')
    npy16k = os.path.join(audiopath, 'npy16k')
    assert not os.path.exists(rot6dpath), 'rot6d directory already exists'
    assert not os.path.exists(rot3dpath), 'rot3d directory already exists'
    assert not os.path.exists(pospath), 'pos directory already exists'
    assert not os.path.exists(npy16k), 'npy16k directory already exists'
    return bvhpath, wavpath, rot6dpath, rot3dpath, pospath, npy16k
    
def takes_get_and_check(bvhpath, wavpath):
    takes = []
    assert len(os.listdir(bvhpath)) == len(os.listdir(wavpath)), 'Number of BVH files does not match number of WAV files'
    for take in os.listdir(bvhpath):
        takes.append(take[:-4])
    for take in os.listdir(wavpath):
        assert take[:-4] in takes, 'WAV file {} does not have a corresponding BVH file'.format(take[:-4])
    return takes

def wav2npy(take, sr=16000):
    signal, _ = librosa.load(take, mono=True, sr=sr)
    return signal
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset/PTBRGestures', help='path to the dataset directory')
    args = parser.parse_args()
    main(args)