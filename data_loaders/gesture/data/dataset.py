import torch
from torch.utils import data
import csv
import os
import numpy as np
from python_speech_features import mfcc
import librosa
import torch.nn.functional as F

class Genea2023(data.Dataset):
    def __init__(self, name, split='train', datapath='./dataset/Genea2023/', step=30, window=80, fps=30, sr=22050, n_seed_poses=10, use_wavlm=False, use_vad=False):

        self.split = split
        if self.split=='train':
            srcpath = os.path.join(datapath, 'trn/main-agent/')
        elif self.split in ['val']:
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

        if self.split in ['train', 'val']:
            self.motionpath = os.path.join(srcpath, 'motion_npy_rotpos')
            self.motionpath_rot6d = os.path.join(srcpath, 'motion_npy_rot6dpos')
            self.frames = np.load(os.path.join(srcpath, 'rotpos_frames.npy'))
        else:
            self.frames = []
            for audiofile in os.listdir(self.audiopath):
                if audiofile.endswith('.npy'):
                    audio = np.load(os.path.join(self.audiopath, audiofile))
                    self.frames.append( int(audio.shape[0]/self.sr*self.fps))
            self.frames = np.array(self.frames)

        self.samples_per_file = [int(np.floor( (n - self.window ) / self.step)) for n in self.frames]
        self.samples_cumulative = [np.sum(self.samples_per_file[:i+1]) for i in range(len(self.samples_per_file))]
        self.length = self.samples_cumulative[-1]
        self.textpath = os.path.join(srcpath, 'tsv')

        self.use_wavlm = use_wavlm
        if self.use_wavlm:
            self.wavlm_rep_path = os.path.join(srcpath, 'wavlm_representations')

        self.use_vad = use_vad
        if self.use_vad:
            self.vad_path = os.path.join(srcpath, "vad")

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
  
    def __getitem__(self, idx):
        if self.split == 'tst':
            raise ValueError('Test set does should not use __getitem__(), use gettestbatch() instead')
        # find the file that the sample belongs two
        file_idx = np.searchsorted(self.samples_cumulative, idx+1, side='left')
        # find sample's index
        if file_idx > 0:
            sample = idx - self.samples_cumulative[file_idx-1]
        else:
            sample = idx
        take_name = self.takes[file_idx][0]
        motion, seed_poses = self.__getmotion( file_idx, sample)
        audio, audio_rep = self.__getaudiofeats(file_idx, sample)
        n_text, text, tokens = self.__gettext(file_idx, sample)
        if self.use_vad:
            vad = self.__getvad(file_idx, sample)
        else:
            vad = np.ones(int(self.window))     # Dummy
        return motion, text, self.window, audio, audio_rep, seed_poses, vad

    def __len__(self):
        return self.length

    def __getvad(self, file, sample):
        # Cut Chunk
        vad_file = np.load(os.path.join(self.vad_path,self.takes[file][0]+'.npy'))
        vad_vals = vad_file[sample*self.step: sample*self.step + self.window]             # [CHUNK_LEN, ]

        # Reshape
        vad_vals = np.expand_dims(vad_vals, 1)                                              # [CHUNK_LEN, 1]
        vad_vals = np.transpose(vad_vals, (1,0))                                            # [1, CHUNK_LEN]
        return vad_vals

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

    def __getaudiofeats(self, file, sample):

        # Load Audio
        signal = np.load(os.path.join(self.audiopath,self.takes[file][0]+'.npy'))
        
        # Cut Chunk
        i = sample*self.sr*self.step/self.fps
        signal = signal[ int(i) : int(i+self.window*self.sr/self.fps) ]

        if self.use_wavlm:
            # Cut Chunk
            representation_file = np.load(os.path.join(self.wavlm_rep_path,self.takes[file][0]+'.npy'))
            wavlm_reps = representation_file[sample*self.step: sample*self.step + self.window,:]            # [CHUNK_LEN, WAVLM_DIM]

            # Reshape
            wavlm_reps = np.transpose(wavlm_reps, (1,0))                                                    # [WAVLM_DIM, CHUNK_LEN]
            wavlm_reps = np.expand_dims(wavlm_reps, 1)                                                      # [WAVLM_DIM, 1, CHUNK_LEN]
            wavlm_reps = np.expand_dims(wavlm_reps, 0)                                                      # [1, WAVLM_DIM, 1, CHUNK_LEN]
            return signal, wavlm_reps
        else:  
            return self.__compute_audiofeats(signal)
        
    def __compute_audiofeats(self, signal):
           
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

    def gettestbatch(self, num_samples):
        max_length = max(self.frames[:num_samples])
        max_length = max_length + self.window - max_length%self.window # increase length so it can be divisible by window
        batch_audio = []
        batch_audio_rep = []
        batch_text = []
        vad_vals = []
        for i, _ in enumerate(self.takes[:num_samples]):
            # Get audio file
            audio_feats = []
            signal  = np.zeros(int(max_length*self.sr/self.fps))
            signal_ = np.load(os.path.join(self.audiopath,self.takes[i][0]+'.npy'))
            signal[:len(signal_)] = signal_

            if self.use_wavlm:
                # Cut Chunk
                wavlm_reps_ = np.load(os.path.join(self.wavlm_rep_path,self.takes[i][0]+'.npy'))
                audio_feat = np.zeros((max_length, wavlm_reps_.shape[1]))
                audio_feat[:wavlm_reps_.shape[0],:] = wavlm_reps_

                # Reshape
                audio_feat = np.transpose(audio_feat, (1,0))                                                    # [WAVLM_DIM, CHUNK_LEN]
                audio_feat = np.expand_dims(audio_feat, 1)                                                      # [WAVLM_DIM, 1, CHUNK_LEN]
                audio_feat = np.expand_dims(audio_feat, 0)                                                      # [1, WAVLM_DIM, 1, CHUNK_LEN]
                audio_feats.append(audio_feat)

            if self.use_vad:
                # Cut Chunk
                vad_val_ = np.load(os.path.join(self.vad_path,self.takes[i][0]+'.npy'))
                vad_val = np.zeros(max_length)
                vad_val[:vad_val_.shape[0]] = vad_val_                                           # [CHUNK_LEN, ]          

                # Reshape
                vad_val = np.expand_dims(vad_val, 1)                                              # [CHUNK_LEN, 1]
                vad_val = np.transpose(vad_val, (1,0))                                            # [1, CHUNK_LEN]
                vad_vals.append(vad_val)

            # Get text file
            text_feats = []
            with open(os.path.join(self.textpath, self.takes[i][0]+'.tsv')) as tsv:
                reader = csv.reader(tsv, delimiter='\t')
                file = [ [float(word[0])*self.fps, float(word[1])*self.fps, word[2]] for word in reader]

            for chunk in range(int(max_length/self.window)):
                if not self.use_wavlm:
                    # Get audio features
                    k = chunk*self.window*self.sr/self.fps
                    _, audio_feat = self.__compute_audiofeats(signal[int(k) : int(k+self.window*self.sr/self.fps)])
                    audio_feats.append(audio_feat)

                # Get text
                begin = self.search_time(file, chunk*self.window)
                end = self.search_time(file, chunk*self.window + self.window)
                text = [ word[-1] for word in file[begin: end] ] if begin or end else []
                text_feats.append(' '.join(text))

            
            audio_feats = np.concatenate(audio_feats, axis=-1)
            end_audio = int(len(signal_)/self.sr*self.fps)
            audio_feats[..., end_audio:] = np.zeros_like(audio_feats[..., end_audio:]) # zero audio feats after end of audio
            batch_audio_rep.append(audio_feats)
            batch_text.append(text_feats)
            batch_audio.append(signal_)

        # Dummy motions and seed poses
        feats = 1245 if self.name == 'genea2023+' else 498
        motion, seed_poses = np.zeros((self.window, feats)), np.zeros((self.n_seed_poses, feats)) #dummy

        # Attention: this is not collate-ready!
        return motion, batch_text, self.window, batch_audio, batch_audio_rep, seed_poses, max_length, vad_vals