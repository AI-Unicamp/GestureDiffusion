import torch
from torch.utils import data
import csv
import os
import numpy as np
from python_speech_features import mfcc
import librosa

class Genea2022(data.Dataset):
    def __init__(self, split='train', datapath='./dataset/Genea/trn', step=30, window=200, fps=30, sr=22050, num_frames=None):
        self.datapath = datapath
        self.window=window
        self.step = step
        self.fps = fps
        self.sr = sr
        self.motionpath = os.path.join(datapath, 'motion_npy')
        self.audiopath = os.path.join(datapath, 'wav')
        self.textpath = os.path.join(datapath, 'tsv')
        self.std = np.load(os.path.join(datapath, 'Std.npy'))
        self.mean = np.load(os.path.join(datapath, 'Mean.npy'))
        self.mfcc_std = np.load(os.path.join(datapath, 'mfccs_Std.npy'))
        self.mfcc_mean = np.load(os.path.join(datapath, 'mfccs_Mean.npy'))
        self.frames = np.load(os.path.join(datapath, 'frames.npy'))
        self.samples_per_file = [int(np.floor( (n-self.window) / self.step)) for n in self.frames]
        self.samples_cumulative = [np.sum(self.samples_per_file[:i+1]) for i in range(len(self.samples_per_file))]

        self.std = np.array([ item if item != 0 else 1 for item in self.std ])

        with open(os.path.join(datapath, 'trn_2022_v1_metadata.csv')) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            self.takes = [take for take in reader]

        for take in self.takes:
            name = take[0]
            m = os.path.join(self.motionpath, name+'.npy')
            a = os.path.join(self.audiopath, name+'.wav')
            t = os.path.join(self.textpath, name+'.tsv')
            assert os.path.isfile( m ), "Motion file {} not found".format(m)
            assert os.path.isfile( a ), "Audio file {} not found".format(a)
            assert os.path.isfile( t ), "Text file {} not found".format(t)

        if split=='train':
            self.begin, self.end = 0, int(self.samples_cumulative[-1]*0.7)
        else:
            self.begin, self.end =  int(self.samples_cumulative[-1]*0.7), self.samples_cumulative[-1]
        self.length = self.end - self.begin

    
    def __getitem__(self, idx):
        idx += self.begin
        # find the file that the sample belongs two
        file_idx = np.searchsorted(self.samples_cumulative, idx+1, side='left')
        # find sample's index
        if file_idx > 0:
            sample = idx - self.samples_cumulative[file_idx-1]
        else:
            sample = idx
        take_name = self.takes[file_idx][0]
        motion = self.__getmotion( file_idx, sample)
        audio, mfcc = self.__getaudiofeats(file_idx, sample)
        n_text, text, tokens = self.__gettext(file_idx, sample)
        return motion, text, self.window, audio, mfcc

    def __len__(self):
        return self.length

    def __getmotion(self, file, sample):
        motion_file = np.load(os.path.join(self.motionpath,self.takes[file][0]+'.npy'))
        return (motion_file[sample*self.step: sample*self.step + self.window ,:] - self.mean) / self.std

    def __getaudiofeats(self, file, sample):
        # Load Audio
        signal, sr = librosa.load(os.path.join(self.audiopath,self.takes[file][0]+'.wav'), mono=True, sr=self.sr)
        
        # Chunk
        i = sample*sr*self.step/self.fps
        signal = signal[ int(i) : int(i+self.window*sr/self.fps) ]

        # MFCCs
        mfcc_vectors = mfcc(signal, winlen=0.06, winstep= (1/self.fps), samplerate=sr, numcep=27, nfft=5000)

        # Normalize
        mfcc_vectors = (mfcc_vectors - self.mfcc_mean) / self.mfcc_std
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
        return data * self.std + self.mean
