import os
from torch.utils import data
import csv
import numpy as np
import torch

class PTBRGesture(data.Dataset):
    def __init__(self, name, split, datapath='./dataset/PTBRGestures', step=10, window=120, fps=30, sr=22050, n_seed_poses=10, use_wavlm=False, use_vad=False, vadfromtext=False):
        
        # Hard-coded because it IS 30 fps
        self.fps = 30

        self.window = window
        self.step = step
        self.n_seed_poses = n_seed_poses

        # Get all paths
        audio_path, wav_path, audio16k_path, vad_path, wavlm_path, \
        motion_path, pos_path, rot3d_path, rot6d_path = self.getpaths(datapath)
        # Get takes from wav path and check if all paths have the same takes
        takes = self.gettakes(wav_path, [audio16k_path, vad_path, wavlm_path, pos_path, rot3d_path, rot6d_path])
        print('Data integrity check passed.')
        # Register takes as Take objects
        self.__registered = False # flag to check if takes are registered
        self.takes = self.registertakes(takes)
        #print(f'{len(self.takes)} takes registered.')

        # Get metadata and register bvh start for audio alignment
        with open(os.path.join(datapath, 'meta.csv'), 'r', encoding='utf-16') as f:
            reader = csv.reader(f, delimiter=',')
            self.metadata = [line for line in reader]
        ratio = self.fps/120 # We are applying this ratio because the bvh_start was computed for 120 fps
        self.metadata = [ [self.gettake(line[0]), np.floor(int(line[1])*ratio).astype(int)] for line in self.metadata ]
        for line in self.metadata:
            line[0].bvh_start = line[1]

        # Hard-coded split
        if split == 'trn':
            b, e1, e2 = 0, 40, 65
        elif split == 'val':
            b, e1, e2 = 40, 45, 73

        # Categorize whole dataset (without unscripted samples)
        self.filtered = [
        self.filter_style_part_id(1,1,1)[b:e1], # id01_p01_e01_fXX
        self.filter_style_part_id(2,1,1)[b:e1], # id01_p01_e02_fXX
        self.filter_style_part_id(3,1,1)[b:e1], # id01_p01_e03_fXX
        self.filter_style_part_id(1,1,2)[b:e1], # id02_p01_e01_fXX
        self.filter_style_part_id(2,1,2)[b:e1], # id02_p01_e02_fXX
        self.filter_style_part_id(3,1,2)[b:e1], # id02_p01_e03_fXX
        self.filter_style_part_id(1,2,1)[b:e2], # id01_p02_e01_fXX
        self.filter_style_part_id(2,2,1)[b:e2], # id01_p02_e02_fXX
        self.filter_style_part_id(3,2,1)[b:e2], # id01_p02_e03_fXX
        self.filter_style_part_id(1,2,2)[b:e2], # id02_p02_e01_fXX
        self.filter_style_part_id(2,2,2)[b:e2], # id02_p02_e02_fXX
        self.filter_style_part_id(3,2,2)[b:e2], # id02_p02_e03_fXX
        ]

        # Get whole dataset given split
        self.takes = [ take for takelist in self.filtered for take in takelist ]
        
        # Load dataset
        print('Loading dataset...')
        self.rot6d = [ np.load(os.path.join(rot6d_path, take.name+'.npy')) for take in self.takes ]
        self.rot3d = [ np.load(os.path.join(rot3d_path, take.name+'.npy')) for take in self.takes ]
        self.pos = [ np.load(os.path.join(pos_path, take.name+'.npy')) for take in self.takes ]
        self.wavlm = [ np.load(os.path.join(wavlm_path, take.name+'.npy')) for take in self.takes ]
        self.vad = [ np.load(os.path.join(vad_path, take.name+'.npy')) for take in self.takes ]
        self.audio16k = [ np.load(os.path.join(audio16k_path, take.name+'.npy')) for take in self.takes ]
        self.velrot3d = [ np.diff(rot3d, axis=0, append=0) for rot3d in self.rot3d ]
        self.velpos = [ np.diff(pos, axis=0, append=0) for pos in self.pos ]
        print('Done')


        self.frames = []
        for index, take in enumerate(self.takes):
            assert self.rot6d[index].shape[0] == self.rot3d[index].shape[0] == self.pos[index].shape[0], f'{take.name} has different lengths'
            self.rot6d[index] = self.rot6d[index][take.bvh_start:]
            self.rot3d[index] = self.rot3d[index][take.bvh_start:]
            self.pos[index] = self.pos[index][take.bvh_start:]
            e = int(self.audio16k[index].shape[0]/16000*self.fps)
            # Due to audio processing, vad and wavlm arrays may have one more frame than the audio arrays
            e = np.min([e, self.vad[index].shape[0], self.wavlm[index].shape[0]])
            self.rot6d[index] = self.rot6d[index][:e]
            self.rot3d[index] = self.rot3d[index][:e]
            self.pos[index] = self.pos[index][:e]
            self.frames.append(self.rot6d[index].shape[0])
        
        #self.samples_per_file = [int(np.floor( (n - self.window  ) / self.step)) for n in self.frames]
        self.samples_per_file = [len( [i for i in np.arange(0, n, self.step) if i + self.window <= n] ) for n in self.frames]
        self.samples_cumulative = [np.sum(self.samples_per_file[:i+1]) for i in range(len(self.samples_per_file))]
        self.length = self.samples_cumulative[-1]

        # Load mean and std for normalization
        self.rot6d_mean, self.rot6d_std, \
        self.rot3d_mean, self.rot3d_std, \
        self.velrot_mean, self.velrot_std, \
        self.pos_mean, self.pos_std, \
        self.velpos_mean, self.velpos_std = self.loadstats(datapath)
        # Get rid of zeros in the std
        self.rot6d_std = np.where(self.rot6d_std == 0, 1, self.rot6d_std)
        self.rot3d_std = np.where(self.rot3d_std == 0, 1, self.rot3d_std)
        self.velrot_std = np.where(self.velrot_std == 0, 1, self.velrot_std)
        self.pos_std = np.where(self.pos_std == 0, 1, self.pos_std)
        self.velpos_std = np.where(self.velpos_std == 0, 1, self.velpos_std)

        #TODO: Do the normalization here since the whole dataset is being loaded into memory

        # Compute some useful info for the dataset that will be used later (in the __getitem__ primarily)
        self.r6d_shape = self.rot6d[0].shape[1]
        self.r3d_shape = self.rot3d[0].shape[1]
        self.pos_shape = self.pos[0].shape[1]
        self.motio_feat_shape = self.r6d_shape + self.r3d_shape + 2*self.pos_shape # 2* for velocity

    def inv_transform(self, data):
        return data * np.concatenate((self.rot6d_std, self.pos_std, self.velrot_std, self.velpos_std)) + np.concatenate((self.rot6d_mean, self.pos_mean, self.velrot_mean, self.velpos_mean))

    def __getitem__(self, index):
        # find the file that the sample belongs two
        file_idx = np.searchsorted(self.samples_cumulative, index+1, side='left')
        # find sample's index in the file
        sample = index - self.samples_cumulative[file_idx-1] if file_idx > 0 else index
        motion, seed_poses = self.__getmotion(file_idx, sample)
        audio = self.__getaudio(file_idx, sample)
        vad = self.__getvad(file_idx, sample)
        wavlm = self.__getwavlm(file_idx, sample)
        return motion, seed_poses, audio, vad, wavlm, [self.takes[file_idx].name, file_idx, sample]

    def __dummysample__(self):
        motion = np.zeros(shape=(self.window, self.motio_feat_shape))
        seed = np.zeros(shape=(self.n_seed_poses, self.motio_feat_shape))
        audio = np.zeros(shape=(int(self.window/30*16000)))
        vad = np.zeros(shape=(self.window, 1))
        vad = np.transpose(vad, (1,0))
        wavlm = np.zeros(shape=(1, 768, 1, self.window))
        return motion, seed, audio, vad, wavlm, ['dummy', 0, 0]

    def __getaudio(self, file_idx, sample):
        # Get audio data from file_idx and sample
        b = int(sample*self.step/30*16000)
        e = int((sample*self.step+self.window)/30*16000)
        return self.audio16k[file_idx][b:e]

    def __getvad(self, file_idx, sample):
        # Get vad data from file_idx and sample
        vad_vals = self.vad[file_idx][sample*self.step:sample*self.step+self.window]
        assert vad_vals.shape[0] == self.window, f'VAD shape is {vad_vals.shape[0]} instead of {self.window}'
        # Reshape
        vad_vals = np.expand_dims(vad_vals, 1)                                              # [CHUNK_LEN, 1]
        vad_vals = np.transpose(vad_vals, (1,0))                                            # [1, CHUNK_LEN]
        return vad_vals

    def __getwavlm(self, file_idx, sample):
        # Get wavlm data from file_idx and sample
        wavlm_reps = self.wavlm[file_idx][sample*self.step:sample*self.step+self.window]
        assert wavlm_reps.shape[0] == self.window, f'WAVLM shape is {wavlm_reps.shape[0]} instead of {self.window}'
        # Reshape
        wavlm_reps = np.transpose(wavlm_reps, (1,0))                                                    # [WAVLM_DIM, CHUNK_LEN]
        wavlm_reps = np.expand_dims(wavlm_reps, 1)                                                      # [WAVLM_DIM, 1, CHUNK_LEN]
        wavlm_reps = np.expand_dims(wavlm_reps, 0)                                                      # [1, WAVLM_DIM, 1, CHUNK_LEN]
        return wavlm_reps

    
    def __getmotion(self, file_idx, sample):
        # Get motion data from file_idx and sample
        motion = np.zeros(shape=(self.window, self.motio_feat_shape))
        b, e = sample*self.step, sample*self.step+self.window
        # Get motion data from rot6d
        motion[:, :self.r6d_shape] = (self.rot6d[file_idx][b:e] - self.rot6d_mean) / self.rot6d_std
        # Get motion data from rot3d
        cumulative = self.r6d_shape
        #motion[:, cumulative:cumulative+self.r3d_shape] = (self.rot3d[file_idx][b:e] - self.rot3d_mean) / self.rot3d_std
        # Get motion data from pos
        #cumulative += self.r3d_shape
        motion[:, cumulative:cumulative+self.pos_shape] = (self.pos[file_idx][b:e] - self.pos_mean) / self.pos_std 
        # Get vel from rot3d
        cumulative += self.pos_shape
        motion[:, cumulative:cumulative+self.r3d_shape] = (self.velrot3d[file_idx][b:e] - self.velrot_mean) / self.velrot_std
        # Get vel from pos
        cumulative += self.r3d_shape
        motion[:, cumulative:cumulative+self.pos_shape] = (self.velpos[file_idx][b:e] - self.velpos_mean) / self.velpos_std
        # Get seed poses
        seed = np.zeros(shape=(self.n_seed_poses, self.motio_feat_shape))
        # TODO: This behaves wrongly for step = 5 for example
        if b - self.n_seed_poses >= 0:
            seed[:, :self.r6d_shape] = (self.rot6d[file_idx][b-self.n_seed_poses:b] - self.rot6d_mean) / self.rot6d_std
            cumulative = self.r6d_shape
            #seed[:, cumulative:cumulative+self.r3d_shape] = (self.rot3d[file_idx][b-self.n_seed_poses:b] - self.rot3d_mean) / self.rot3d_std
            #cumulative += self.r3d_shape
            seed[:, cumulative:cumulative+self.pos_shape] = (self.pos[file_idx][b-self.n_seed_poses:b] - self.pos_mean) / self.pos_std
            cumulative += self.pos_shape
            seed[:, cumulative:cumulative+self.r3d_shape] = (self.velrot3d[file_idx][b-self.n_seed_poses:b] - self.velrot_mean) / self.velrot_std
            cumulative += self.r3d_shape
            seed[:, cumulative:cumulative+self.pos_shape] = (self.velpos[file_idx][b-self.n_seed_poses:b] - self.velpos_mean) / self.velpos_std
        return motion, seed

    def __len__(self):
        return self.length

    def getpaths(self, datapath):
        # Create a list of paths to the dataset folders
        # and check if all paths exist
        audio_path = os.path.join(datapath, 'audio')
        wav_path = os.path.join(audio_path, 'wav')
        audio16k_path = os.path.join(audio_path, 'npy16k')
        vad_path = os.path.join(audio_path, 'vad')
        wavlm_path = os.path.join(audio_path, 'wavlm')
        motion_path = os.path.join(datapath, 'motion')
        pos_path = os.path.join(motion_path, 'pos')
        rot3d_path = os.path.join(motion_path, 'rot3d')
        rot6d_path = os.path.join(motion_path, 'rot6d')
        for path in [audio_path, wav_path, audio16k_path, vad_path, wavlm_path, motion_path, pos_path, rot3d_path, rot6d_path]:
            assert os.path.exists(path), f'{path} does not exist'
        return audio_path, wav_path, audio16k_path, vad_path, wavlm_path, motion_path, pos_path, rot3d_path, rot6d_path
    
    def gettakes(self, reference_path, paths):
        # Create a list of take names based on the takes in the reference path
        # Also checks if all paths have the same takes
        takes = []
        for file in os.listdir(reference_path):
            for path in paths:
                assert file[:-4] in [os.path.basename(f)[:-4] for f in os.listdir(path)], f'{file} not found in {path}'
            takes.append(file[:-4])
        #print(f'Found {len(takes)} takes.')
        return takes
    
    def registertakes(self, takes):
        # Sort takes and create Take objects
        takes.sort()
        class_takes = []
        for take in takes:
            class_takes.append(Take(take))
        self.__registered = True
        return class_takes

    def filter_id(self, id, include_unscripted=False):
        # Get list of takes from id
        assert self.__registered, 'Takes are not registered'
        takelist = []
        for take in self.takes:
            if take.id == id:
                if include_unscripted:
                    takelist.append(take)
                else:
                    if take.type == 'scripted':
                        takelist.append(take)
        assert takelist, f'No takes found for id {id}'
        return takelist
    
    def filter_part_id(self, part, id, include_unscripted=False):
        # Get list of takes from id, then remove takes that are not from part
        assert self.__registered, 'Takes are not registered'
        takelist = self.filter_id(id, include_unscripted)
        takelist = [take for take in takelist if take.part == part]
        assert takelist, f'No takes found for id {id} and part {part}'
        return takelist
    
    def filter_style_part_id(self, style, part, id, include_unscripted=False):
        # Get list of takes from id, then remove takes that are not from part and style
        assert self.__registered, 'Takes are not registered'
        takelist = self.filter_part_id(part, id, include_unscripted)
        takelist = [take for take in takelist if take.style == style]
        assert takelist, f'No takes found for id {id}, part {part} and style {style}'
        return takelist
    
    def gettake(self, name):
        # Get take from name
        assert self.__registered, 'Takes are not registered'
        for take in self.takes:
            if take.name == name:
                return take
        assert False, f'Take {name} not found'

    def loadstats(self, statspath):
        rot6d_mean = np.load(os.path.join(statspath, 'rot6d_Mean.npy'))
        rot6d_std = np.load(os.path.join(statspath, 'rot6d_Std.npy'))
        rot3d_mean = np.load(os.path.join(statspath, 'rot3d_Mean.npy'))
        rot3d_std = np.load(os.path.join(statspath, 'rot3d_Std.npy'))
        velrot_mean = np.load(os.path.join(statspath, 'velrot_Mean.npy'))
        velrot_std = np.load(os.path.join(statspath, 'velrot_Std.npy'))
        pos_mean = np.load(os.path.join(statspath, 'pos_Mean.npy'))
        pos_std = np.load(os.path.join(statspath, 'pos_Std.npy'))
        velpos_mean = np.load(os.path.join(statspath, 'velpos_Mean.npy'))
        velpos_std = np.load(os.path.join(statspath, 'velpos_Std.npy'))
        return rot6d_mean, rot6d_std, rot3d_mean, rot3d_std, velrot_mean, velrot_std, pos_mean, pos_std, velpos_mean, velpos_std
    


class Take():
    def __init__(self, name):
        self.name = name
        splited = name.split('_')
        self.id = self.getint(splited[0])
        self.type = 'unscripted' if splited[1] == 'un' else 'scripted'
        if self.type == 'scripted':
            self.part = self.getint(splited[1])
            self.phrase = self.getint(splited[3])
        else:
            self.part = None
            self.phrase = None
        self.style = self.getint(splited[2])
        self.bvh_start = 0

    def getint(self, string):
        # Get integer from string. Note: 'something01' -> 1
        return int(''.join(filter(str.isdigit, string)))
        
        