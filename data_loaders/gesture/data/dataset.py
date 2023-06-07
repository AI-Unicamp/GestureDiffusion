import torch
from torch.utils import data
import csv
import os
import numpy as np
from python_speech_features import mfcc
import librosa
from wavlm.WavLM import WavLM, WavLMConfig
import torch.nn.functional as F

class Genea2023(data.Dataset):
    def __init__(self, name, split='train', datapath='./dataset/Genea2023/', step=30, window=80, fps=30, sr=22050, n_seed_poses=10, use_wavlm=False):

        self.split = split
        if self.split in ['train', 'hands']:
            srcpath = os.path.join(datapath, 'trn/main-agent/')
        elif self.split == 'val':
            srcpath = os.path.join(datapath, 'val/main-agent/')
        elif self.split == 'tst':
            srcpath = os.path.join(datapath, 'tst/main-agent/')
        else:
            raise NotImplementedError

        if use_wavlm:
            self.sr = 16000
            self.audiopath = os.path.join(srcpath, 'audio16k_npy')
        else:
            self.sr = sr
            self.audiopath = os.path.join(srcpath, 'audio_npy')

        self.name = name
        self.step = step

        self.datapath = datapath
        self.window=window
        self.fps = fps
        self.n_seed_poses = n_seed_poses

        self.loadstats(os.path.join(datapath, 'trn/main-agent/'))
        self.std = np.array([ item if item != 0 else 1 for item in self.std ])
        self.vel_std = np.array([ item if item != 0 else 1 for item in self.vel_std ])
        self.rot6dpos_std = np.array([ item if item != 0 else 1 for item in self.rot6dpos_std ])

        if self.split in ['trn', 'val', 'hands']:
            self.motionpath = os.path.join(srcpath, 'motion_npy_rotpos')
            self.motionpath_rot6d = os.path.join(srcpath, 'motion_npy_rot6dpos')
            self.frames = np.load(os.path.join(srcpath, 'rotpos_frames.npy'))
        else:
            self.frames = []
            for audiofile in os.listdir(self.audiopath):
                if audiofile.endswith('.npy'):
                    audio = np.load(os.path.join(self.audiopath, audiofile))
                    self.frames.append(audio.shape[0])
            self.frames = np.array(self.frames)
            
        self.samples_per_file = [int(np.floor( (n - self.window ) / self.step)) for n in self.frames]
        self.samples_cumulative = [np.sum(self.samples_per_file[:i+1]) for i in range(len(self.samples_per_file))]
        self.length = self.samples_cumulative[-1]
        self.textpath = os.path.join(srcpath, 'tsv')
   
        self.use_wavlm = use_wavlm
        if self.use_wavlm:
            checkpoint = torch.load('./wavlm/WavLM-Large.pt')
            self.wavlm_cfg = WavLMConfig(checkpoint['cfg'])
            self.wavlm = WavLM(self.wavlm_cfg)
            self.wavlm.load_state_dict(checkpoint['model'])
            self.wavlm.eval()
            print('Selected Features: WavLM Representations')

        with open(os.path.join(srcpath, '../metadata.csv')) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            self.takes = [take for take in reader]
            self.takes = self.takes[1:]
            for take in self.takes:
                take[0] += '_main-agent'

        if False:
            for take in self.takes:
                name = take[0]
                m = os.path.join(self.motionpath, name+'.npy')
                a = os.path.join(self.audiopath, name+'.npy')
                t = os.path.join(self.textpath, name+'.tsv')
                assert os.path.isfile( m ), "Motion file {} not found".format(m)
                assert os.path.isfile( a ), "Audio file {} not found".format(a)
                assert os.path.isfile( t ), "Text file {} not found".format(t)

        # Recompute dataset params to include only takes with hand information
        if split == 'hands':
            tokeep = [i for i, take in enumerate(self.takes) if take[2] == 'finger_incl']
            self.takes = [self.takes[i] for i in tokeep]
            self.frames = self.frames[tokeep]
            self.samples_per_file = [int(np.floor( (n - self.window ) / self.step)) for n in self.frames]
            self.samples_cumulative = [np.sum(self.samples_per_file[:i+1]) for i in range(len(self.samples_per_file))]
            self.length = self.samples_cumulative[-1]

  
    def __getitem__(self, idx):
        # find the file that the sample belongs two
        file_idx = np.searchsorted(self.samples_cumulative, idx+1, side='left')
        # find sample's index
        if file_idx > 0:
            sample = idx - self.samples_cumulative[file_idx-1]
        else:
            sample = idx
        take_name = self.takes[file_idx][0]
        if self.split not in 'tst':
            motion, seed_poses = self.__getmotion( file_idx, sample)
        else:
            feats = 1245 if self.name == 'genea2023+' else 498
            motion, seed_poses = np.zeros((self.window, feats)), np.zeros((self.n_seed_poses, feats)) #dummy
        audio, audio_rep = self.__getaudiofeats(file_idx, sample)
        n_text, text, tokens = self.__gettext(file_idx, sample)
        return motion, text, self.window, audio, audio_rep, seed_poses

    def __len__(self):
        return self.length

    def __getmotion(self, file, sample):
        if self.name == 'genea2023+':
            # loading rot6d and position representations
            rot6dpos_file = np.load(os.path.join(self.motionpath_rot6d,self.takes[file][0]+'.npy'))
            rot6dpos = (rot6dpos_file[sample*self.step: sample*self.step + self.window,:] - self.rot6dpos_mean) / self.rot6dpos_std
            
            # loading rotpos representation and computing velocity
            rotpos_file = np.load(os.path.join(self.motionpath,self.takes[file][0]+'.npy'))
            rotpos_file[1:,:] = rotpos_file[1:,:] - rotpos_file[:-1,:]
            rotpos_file[0,:] = np.zeros(rotpos_file.shape[1])
            rotpos = (rotpos_file[sample*self.step: sample*self.step + self.window,:] - self.vel_mean) / self.vel_std
            if sample*self.step - self.n_seed_poses < 0:    
                rot6dpos_seed = np.zeros((self.n_seed_poses, rot6dpos.shape[1]))
                rotpos_seed = np.zeros((self.n_seed_poses, rotpos.shape[1]))
            else:
                rot6dpos_seed = (rot6dpos_file[sample*self.step - self.n_seed_poses: sample*self.step ,:] - self.rot6dpos_mean) / self.rot6dpos_std
                rotpos_seed = (rotpos_file[sample*self.step - self.n_seed_poses: sample*self.step,:] - self.vel_mean) / self.vel_std

            motion = np.concatenate((rot6dpos, rotpos), axis=1)
            seed_poses = np.concatenate((rot6dpos_seed, rotpos_seed), axis=1)
            
        else:
            motion_file = np.load(os.path.join(self.motionpath,self.takes[file][0]+'.npy'))
            motion = (motion_file[sample*self.step: sample*self.step + self.window,:] - self.mean) / self.std
            if sample*self.step - self.n_seed_poses < 0:
                seed_poses = np.zeros((self.n_seed_poses, motion.shape[1]))
            else:
                seed_poses = (motion_file[sample*self.step - self.n_seed_poses: sample*self.step,:] - self.mean) / self.std    
            
        return motion, seed_poses
    
    def __oldgetmotion(self, file, sample):
        motion_file = np.load(os.path.join(self.motionpath,self.takes[file][0]+'.npy'))
        motion = (motion_file[sample*self.step: sample*self.step + self.window,:] - self.mean) / self.std
        seed_poses = (motion_file[sample*self.step: sample*self.step + self.n_seed_poses,:] - self.mean) / self.std
        return motion, seed_poses

    def __getaudiofeats(self, file, sample):
        # Load Audio
        #signal, sr = librosa.load(os.path.join(self.audiopath,self.takes[file][0]+'.wav'), mono=True, sr=self.sr)
        signal = np.load(os.path.join(self.audiopath,self.takes[file][0]+'.npy'))
        
        # Chunk
        i = sample*self.sr*self.step/self.fps
        signal = signal[ int(i) : int(i+self.window*self.sr/self.fps) ]

        # WavLM Representations
        if self.use_wavlm:
            with torch.no_grad():
                wav = torch.tensor(signal).unsqueeze(0)                        # [1, AUDIO_LEN]
                if self.wavlm_cfg.normalize:
                    wav = torch.nn.functional.layer_norm(wav , wav.shape)      #  [1, AUDIO_LEN]
                reps = self.wavlm.extract_features(wav)[0]                     #  [1, CONVS_OUT_DIM , 768], CONVS_OUT_DIM for 4 seconds of 16khz audio is 199
                interp_reps = F.interpolate(reps.transpose(1, 2), size=self.window, align_corners=True, mode='linear').unsqueeze(2) # should be [1, 768, 1, CHUNK_LEN]
            return signal, interp_reps.cpu().detach().data.cpu().numpy()
        else:
            # MFCCs
            mfcc_vectors = mfcc(signal, winlen=0.06, winstep= (1/self.fps), samplerate=self.sr, numcep=27, nfft=5000)

            # Normalize
            mfcc_vectors = (mfcc_vectors - self.mfcc_mean) / self.mfcc_std

            # Format
            mfcc_vectors = mfcc_vectors.T
            mfcc_vectors = np.expand_dims(mfcc_vectors, 1)
            mfcc_vectors = np.expand_dims(mfcc_vectors, 0)  # should be [1, MFCC_DIM, 1, CHUNK_LEN]
            return signal, mfcc_vectors

    def __gettext(self, file, sample):
        with open(os.path.join(self.textpath, self.takes[file][0]+'.tsv')) as tsv:
            reader = csv.reader(tsv, delimiter='\t')
            file = [ [float(word[0])*self.fps, float(word[1])*self.fps, word[2]] for word in reader]
        begin = self.search_time(file, sample*self.step)
        end = self.search_time(file, sample*self.step + self.window)
        text = [ word[-1] for word in file[begin: end] ]
        tokens = self.__gentokens(text)
        return len(text), ' '.join(text), tokens
    
    def __gentokens(self, text):
        tokens = [ word+'/OTHER' for word in text]
        tokens = '_'.join(tokens)
        tokens = 'sos/OTHER_' + tokens + '_eos/OTHER'
        return tokens

    def search_time(self, text, frame):
        for i in range(len(text)):
            if frame <= text[i][0]:
                return i if (frame > text[i-1][1] or i==0) else i-1
    
    def inv_transform(self, data):
        if self.name == 'genea2023':
            return data * self.std + self.mean
        elif self.name == 'genea2023+':
            return data * np.concatenate((self.rot6dpos_std, self.vel_std)) + np.concatenate((self.rot6dpos_mean, self.vel_mean))
        else:
            raise ValueError('Dataset name not recognized')


    def gettime(self):
        import time
        start = time.time()
        for i in range(200):
            sample = self.__getitem__(i)
        print(time.time()-start)

    def loadstats(self, statspath):
        self.std = np.load(os.path.join(statspath, 'rotpos_Std.npy'))
        self.mean = np.load(os.path.join(statspath, 'rotpos_Mean.npy'))
        self.mfcc_std = np.load(os.path.join(statspath, 'mfccs_Std.npy'))
        self.mfcc_mean = np.load(os.path.join(statspath, 'mfccs_Mean.npy'))
        self.rot6dpos_std = np.load(os.path.join(statspath, 'rot6dpos_Std.npy'))
        self.rot6dpos_mean = np.load(os.path.join(statspath, 'rot6dpos_Mean.npy'))
        self.vel_std = np.load(os.path.join(statspath, 'velrotpos_Std.npy'))
        self.vel_mean = np.load(os.path.join(statspath, 'velrotpos_Mean.npy'))
