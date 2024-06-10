import os
import numpy as np
import bvhsdk
from tqdm import tqdm
import sys
import csv

def psize(arr, name=None):
    """Prints the size of a NumPy array in megabytes."""
    # Get size in bytes using numpy.nbytes or getsizeof
    size_in_bytes = arr.nbytes if isinstance(arr, np.ndarray) else sys.getsizeof(arr)
    shape = arr.shape if isinstance(arr, np.ndarray) else len(arr)
    # Convert bytes to megabytes
    size_in_mb = size_in_bytes / (1024 * 1024)
    if name:
        print(f"{name} size is: {size_in_mb:.2f} MB. Shape: {shape}")
    else:
        print(f"np array size is: {size_in_mb:.2f} MB. Shape: {shape}")
    return size_in_mb

def sliding_window(a, L, S ):  
    # Window len = L, Stride len or stepsize = S
    nrows = ((a.shape[0]-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def text2class_label(string):
    if 'e01' in string:
        class_label = 0
    elif 'e02' in string:
        class_label = 1
    elif 'e03' in string:
        class_label = 2
    else:
        raise ValueError("Could not find style label in bvh title")
    class_label += 3 if 'p02' in string else 0
    class_label += 6 if 'id02' in string else 0
    return class_label

class DatasetBVHLoader():
    def __init__(self,
                 name,
                 path,
                 data_rep = 'pos',
                 step=60,
                 window=120,
                 fps=30,
                 pos_mean = 'dataset/PTBRGestures/pos_Mean.npy',
                 pos_std = 'dataset/PTBRGestures/pos_Std.npy',
                 rot3d_mean = 'dataset/PTBRGestures/rot3d_Mean.npy',
                 rot3d_std = 'dataset/PTBRGestures/rot3d_Std.npy',
                 skiptjoints = 1,
                 metadata = False,
                 metadata_path = 'dataset/PTBRGestures/meta.csv',
                 **kwargs) -> None:
        
        self.step = step
        self.window = window
        self.fps = fps
        self.name = name
        self.path = path
        self.skiptjoints = skiptjoints
        self.data_rep = data_rep
        self.pos_mean = np.load(pos_mean)
        self.pos_std = np.load(pos_std)
        self.rot3d_mean = np.load(rot3d_mean)
        self.rot3d_std = np.load(rot3d_std)
        self.report_path = os.path.join(self.path, "report")
        
        if not os.path.isdir(self.report_path):
            os.mkdir(self.report_path)
        # Compose files with bvhs in path our based on a files list passed as 
        self.files = kwargs.pop('files', [file for file in os.listdir(path) if file.endswith('.bvh')])
        self.files.sort()

        # Get parents vector (skeleton hierarchy)
        aux = bvhsdk.ReadFile(os.path.join(self.path,self.files[0]))
        self.parents = aux.arrayParent()

        # If load = True, loads already processed data
        if kwargs.pop('load', False):
            #Check if path is a file ending with ".npy"
            self.pos = np.load(os.path.join(self.report_path, self.name + "_bvh_positions.npy"), allow_pickle = True)
            self.rot3d = np.load(os.path.join(self.report_path, self.name + "_bvh_3drotations.npy"), allow_pickle = True)
            self.labels = np.load(os.path.join(self.report_path, self.name + "_labels.npy"), allow_pickle = True)
        else:
            self.__data2samples(**kwargs)
            # Align BVH with audio given metada
            if metadata:
                # Get metadata and register bvh start for audio alignment
                with open(metadata_path, 'r', encoding='utf-16') as f:
                    reader = csv.reader(f, delimiter=',')
                    self.metadata = [line for line in reader]
                ratio = self.fps/120 # We are applying this ratio because the bvh_start was computed for 120 fps
                self.metadata = [ np.floor(int(line[1])*ratio).astype(int) for line in self.metadata if line[0] in self.files]
                for i, line in enumerate(self.metadata):
                    self.pos[i] = self.pos[i][line:]
                    self.rot3d[i] = self.rot3d[i][line:]
            # This does not actually save a np array due to different lens of each take
            np.save(file = os.path.join(self.report_path, self.name + "_bvh_positions.npy"),
                    arr = self.pos,
                    allow_pickle = True)
            np.save(file = os.path.join(self.report_path, self.name + "_bvh_3drotations.npy"),
                    arr = self.rot3d,
                    allow_pickle = True)
            np.save(file = os.path.join(self.report_path, self.name + "_labels.npy"),
                    arr = self.labels,
                    allow_pickle = True)
        
        self.frames = [len(take) for take in self.pos]
        self.samples_per_take = [len( [i for i in np.arange(0, n, self.step) if i + self.window <= n] ) for n in self.frames]
        self.samples_cumulative = [np.sum(self.samples_per_take[:i+1]) for i in range(len(self.samples_per_take))]
        self.length = self.samples_cumulative[-1]

    def __getitem__(self, index):
        file_idx = np.searchsorted(self.samples_cumulative, index+1, side='left')
        sample = index - self.samples_cumulative[file_idx-1] if file_idx > 0 else index
        b, e = sample*self.step, sample*self.step+self.window
        if self.data_rep == 'pos':
            sample = self.norma(self.pos[file_idx][b:e, self.skipjoint:, :].reshape(-1, (83-self.skipjoint)*3), self.pos_mean[3*self.skipjoint:], self.pos_std[3*self.skipjoint:])
        elif self.data_rep == 'rot3d':
            sample = self.norma(self.rot3d[file_idx][b:e, self.skipjoint:, :].reshape(-1, (83-self.skipjoint)*3), self.rot3d_mean[3*self.skipjoint:], self.rot3d_std[3*self.skipjoint:])
        return sample, self.labels[file_idx], self.files[file_idx] + f"_{b}_{e}"
        
    def norma(self, arr_, mean, std):
        return (arr_-mean) / std
    
    def inv_norma(self, arr_, mean, std):
        return (arr_*std) + mean
        
    def __len__(self):
        return self.length

    def __data2samples(self, **kwargs):
        # Converts all files (takes) to samples
        self.pos, self.rot3d = [], []
        labels = np.zeros(len(self.files))
        print('Preparing samples...')
        for i, file in enumerate(tqdm(self.files)):
            anim = bvhsdk.ReadFile(os.path.join(self.path,file))
            p, r = self.__loadtake(anim)
            self.pos.append(p)
            self.rot3d.append(r)
            labels[i] = text2class_label(file)
        print('Done. Converting to numpy.')
        self.labels = labels
        #psize(self.pos, "Samples np")
        
    def __loadtake(self, anim):
        # Converts a single file (take) to samples
        # Compute joint position
        joint_positions, joint_rotations = [], []
        for frame in range(anim.frames):
            joint_positions.append([joint.getPosition(frame) for joint in anim.getlistofjoints()])
            joint_rotations.append([joint.rotation[frame] for joint in anim.getlistofjoints()])
        
        #size = psize(joint_positions, "All joints")
        
        return np.asarray(joint_positions), np.asarray(joint_rotations)