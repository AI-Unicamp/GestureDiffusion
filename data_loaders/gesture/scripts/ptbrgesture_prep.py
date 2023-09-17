from argparse import ArgumentParser
import os
import numpy as np
import librosa
import torch
from wavlm.WavLM import WavLM, WavLMConfig
import torch.nn.functional as F
from data_loaders.gesture.scripts.motion_process import bvh2representations1
import bvhsdk
from tqdm import tqdm

def main(args):
    bvhpath, wavpath, rot6dpath, rot3dpath, pospath, npy16k, wavlmpath = paths_get_and_check(args.data_dir)
    takes = takes_get_and_check(bvhpath, wavpath)
    assert args.step in ['all', 'bvh', 'wav', 'wavlm'], f"Step {args.step} not recognized. Options: \'all\', \'bvh\', \'wav\', \'wavlm\'" # Check if user is trying to process a step that does not exist
    steps = [args.step] if args.step != 'all' else ['bvh', 'wav', 'wavlm']
    if 'bvh' in steps:
        print('Processing bvh')
        process_bvh(bvhpath, rot6dpath, rot3dpath, pospath, takes)
        print('Computing mean and std')
        compute_meanstd(rot6dpath, os.path.join(args.data_dir, 'rot6d'), npstep=1)
        compute_meanstd(rot3dpath, os.path.join(args.data_dir, 'rot3d'), npstep=1)
        compute_meanstd(pospath, os.path.join(args.data_dir, 'pos'), npstep=1)
        compute_meanstd(rot3dpath, os.path.join(args.data_dir, 'velrot'), npstep=1, vel=True)
        compute_meanstd(pospath, os.path.join(args.data_dir, 'velpos'), npstep=1, vel=True)
    if 'wav' in steps:
        print('Processing wav')
        process_wav(wavpath, npy16k)
    if 'wavlm' in steps:
        print('Processing wavlm')
        process_wavlm(npy16k, wavlmpath)

def process_wavlm(sourcepath, savepath, split):
    wavlm_layer = 11 
    fps=30
    sr=16000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert os.path.exists(sourcepath), f"audio16k_npy not found in {sourcepath}. Required to process wavlm representations, make sure wav files were processed first."
    assert os.path.exists(savepath), f"wavlm model directory not found in current directory."
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    checkpoint = torch.load('./wavlm/WavLM-Base+.pt')
    wavlm_cfg = WavLMConfig(checkpoint['cfg'])
    wavlm = WavLM(wavlm_cfg)
    #wavlm.to(device)
    wavlm.load_state_dict(checkpoint['model'])
    wavlm.eval()
    for file in tqdm(os.listdir(sourcepath)):
        if not os.path.exists(os.path.join(savepath, file)):
            audio_path = os.path.join(sourcepath, file)
            # Load with Numpy
            signal = np.load(audio_path)
            # Set to model innput format
            signal = torch.tensor(signal).unsqueeze(0)#.to(device)
            # Normalize
            if wavlm_cfg.normalize:
                signal_norm = torch.nn.functional.layer_norm(signal , signal.shape)
            else:
                signal_norm = signal
            # Run Model (rep=Desired Layer, layer_results=all layers)
            rep, layer_results = wavlm.extract_features(signal_norm, output_layer=wavlm_layer, ret_layer_results=True)[0]
            layer_reps = [x.transpose(0, 1) for x, _ in layer_results] # fix shape
            # Get Number of Seconds of Audio File
            n_secs = signal.shape[1] / sr
            # Get Number of poses equivalent to audio file duration, given fps (alignment len)
            n_pose = n_secs * fps
            # Interpolate number of representations to match number of poses corresponding to audio file
            interp_reps = F.interpolate(rep.transpose(1, 2), size=int(n_pose), align_corners=True, mode='linear')
            # Prepare to save
            interp_reps = interp_reps.squeeze(0).transpose(0,1).cpu().detach().data.cpu().numpy()
            # Double check dimension
            assert (interp_reps.shape[0] == int(np.ceil(n_pose)) or interp_reps.shape[0] == int(np.floor(n_pose)))
            np.save(os.path.join(savepath, file), interp_reps)

def process_wav(sourcepath, savepath, sr=16000):
    assert not os.path.exists(savepath), f"audio_16k_npy already exists in {savepath}. Delete it to process again."
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    for file in tqdm(os.listdir(sourcepath)):
        #if not os.path.exists(os.path.join(savepath, file[:-4] + '.npy')):
        signal, _sr = librosa.load(os.path.join(sourcepath, file), mono=True, sr=sr)
        assert _sr == sr
        np.save(os.path.join(savepath, file[:-4]+'.npy'), signal)
    return savepath

def process_bvh(bvhpath, rot6dpath, rot3dpath, pospath, takes):
    # Create paths
    for path in [rot6dpath, rot3dpath, pospath]:
        if not os.path.exists(path):
            os.mkdir(path)
    for file in tqdm(os.listdir(bvhpath)):
        anim = bvhsdk.ReadFile(os.path.join(bvhpath, file))
        npyrot6d, npyrot, npypos = bvh2representations1(anim)
        np.save(os.path.join(rot6dpath, file[:-4]), npyrot6d)
        np.save(os.path.join(rot3dpath, file[:-4]), npyrot)
        np.save(os.path.join(pospath, file[:-4]), npypos)

def compute_meanstd(path, savepath, npstep=1, vel=False):
    all_data = []
    for f in os.listdir(path)[::npstep]:
        data = np.load(os.path.join(path, f))
        if vel:
            data = data[1:,:] - data[:-1,:]
            data[0,:] = np.zeros(data.shape[1])
        all_data.append(data)
    all_data = np.vstack(all_data)
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    np.save(savepath + '_Mean.npy', mean)
    np.save(savepath + '_Std.npy', std)

def paths_get_and_check(data_dir):
    assert os.path.exists(data_dir), 'Data directory does not exist'
    motionpath = os.path.join(data_dir, 'motion')
    audiopath = os.path.join(data_dir, 'audio')
    bvhpath = os.path.join(motionpath, 'bvh_twh')
    wavpath = os.path.join(audiopath, 'wav')
    assert os.path.exists(motionpath), 'Motion directory does not exist'
    assert os.path.exists(audiopath), 'Audio directory does not exist'
    assert os.path.exists(bvhpath), 'BVH directory does not exist'
    assert os.path.exists(wavpath), 'WAV directory does not exist'
    rot6dpath = os.path.join(motionpath, 'rot6d')
    rot3dpath = os.path.join(motionpath, 'rot3d')
    pospath = os.path.join(motionpath, 'pos')
    npy16k = os.path.join(audiopath, 'npy16k')
    wavlmpath = os.path.join(audiopath, 'wavlm')
    assert not os.path.exists(rot6dpath), 'rot6d directory already exists'
    assert not os.path.exists(rot3dpath), 'rot3d directory already exists'
    assert not os.path.exists(pospath), 'pos directory already exists'
    assert not os.path.exists(npy16k), 'npy16k directory already exists'
    assert not os.path.exists(wavlmpath), 'wavlm directory already exists'
    return bvhpath, wavpath, rot6dpath, rot3dpath, pospath, npy16k, wavlmpath
    
def takes_get_and_check(bvhpath, wavpath):
    takes = []
    assert len(os.listdir(bvhpath)) == len(os.listdir(wavpath)), 'Number of BVH files does not match number of WAV files'
    for take in os.listdir(bvhpath):
        takes.append(take[:-4])
    for take in os.listdir(wavpath):
        assert take[:-4] in takes, 'WAV file {} does not have a corresponding BVH file'.format(take[:-4])
    return takes

        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset/PTBRGestures', help='path to the dataset directory')
    parser.add_argument('--step', type=str, default='all', help='Which step to process. Use \'all\' to process all steps. Options: \'bvh\', \'wav\', \'wavlm\'')
    args = parser.parse_args()
    main(args)