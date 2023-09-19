from argparse import ArgumentParser
import os
import numpy as np
import torch
from speechbrain.pretrained import VAD
import torchaudio
from scipy.signal import resample
from tqdm import tqdm



def main(args):
    #paths_check(args.data_dir)

    sourcepath = os.path.join(args.data_dir, 'audio', 'wav')
    savepath = os.path.join(args.data_dir, 'audio', 'vad')
    print('Processing VAD.')
    process_vad(sourcepath, savepath, args.data_dir)
    
    
def process_vad(sourcepath, savepath, datadir):
    sr=16000
    fps=30
    _VAD = VAD.from_hparams(source= "speechbrain/vad-crdnn-libriparty", savedir= os.path.join(datadir, '..','..','speechbrain', 'pretrained_models', 'vad-crdnn-libriparty'))
    #assert not os.path.exists(savepathrot), f"vad already exists in {savepathrot}. Delete it to process again."
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    # VAD requires a torch tensor with sample rate = 16k. This process saves a temporary wav file with 16k sr. It can be deleted after processing.
    for file in tqdm(os.listdir(sourcepath)):
        if not os.path.exists(os.path.join(savepath, file[:-4]+'.npy')):
            audio, old_sr = torchaudio.load(os.path.join(sourcepath,file))
            audio = torchaudio.functional.resample(audio, orig_freq=44100, new_freq=sr)
            tmpfile = "tmp.wav"
            torchaudio.save(
            tmpfile , audio, sr
            )
            boundaries = _VAD.get_speech_prob_file(audio_file=tmpfile, large_chunk_size=4, small_chunk_size=0.2)
            boundaries = resample(boundaries[0,:,0], int(boundaries.shape[1]*fps/100))
            boundaries[boundaries>=0.5] = 1
            boundaries[boundaries<0.5] = 0
            np.save(os.path.join(savepath, file[:-4]+'.npy'), boundaries)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset/PTBRGestures', help='path to the dataset directory')
    args = parser.parse_args()
    main(args)