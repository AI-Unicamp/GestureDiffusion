from data_loaders.gesture.scripts import motion_process as mp
from data_loaders.get_data import get_dataset_loader
import numpy as np
from tqdm import tqdm
from utils import dist_util
import torch
import bvhsdk
from evaluation_metric.embedding_space_evaluator import EmbeddingSpaceEvaluator
from evaluation_metric.train_AE import make_tensor, files_to_tensor
from sample import ptbrgenerate
import matplotlib.pyplot as plt
import os, glob

# Imports for calling from command line
from utils.parser_util import generate_args
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip


class PTBREvaluator:
    def __init__(self, args, model, diffusion):
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.dataloader = get_dataset_loader(name=args.dataset, 
                                        batch_size=args.batch_size, 
                                        num_frames=args.num_frames, 
                                        step=args.num_frames, #no overlap
                                        use_wavlm=args.use_wavlm, 
                                        use_vad=True, #Hard-coded to get vad from files but the model will not use it since args.use_vad=False
                                        vadfromtext=args.vadfromtext,
                                        split='val')
        self.data = self.dataloader.dataset
        self.bvhreference = bvhsdk.ReadFile(args.bvh_reference_file, skipmotion=True)
        #self.idx_positions, self.idx_rotations = mp.get_indexes('genea2023') # hard-coded 'genea2023' because std and mean vec are computed for this representation
        self.fgd_evaluator = EmbeddingSpaceEvaluator('./evaluation_metric/output/model_checkpoint_120_ptbr.bin', args.num_frames, dist_util.dev())
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ground_truth_path = './dataset/PTBRGestures/motion/pos'
        assert os.path.exists(self.ground_truth_path), f"Ground truth path {self.ground_truth_path} does not exist. Required to compute FGD."
        

    def eval(self, samples=None, chunks=None):
        print('Starting evaluation...')
        n_samples = samples if samples else len(self.data.takes)
        n_chunks = chunks if chunks else np.min(self.data.samples_per_file)
        #rot, gt_rot, pos, gt_pos, vad  = self.sampleval(n_samples, n_chunks)
        n_joints = 83
        pos, rot, sample_names = ptbrgenerate.sample(self.args, self.model, self.diffusion, self.dataloader, self.dataloader.collate_fn, n_joints)
        pos, rot = mp.filter_and_interp(rot, pos, num_frames=self.args.num_frames)

        listpos, listposgt = [], []
        print('Converting to BVH and back to get positions...')
        for i in tqdm(range(len(pos))):
            # Transform to BVH and get positions of sampled motion
            bvhreference = mp.tobvh(self.bvhreference, rot[i], pos[i])
            listpos.append(mp.posfrombvh(bvhreference))
            # Transform to BVH and get positions of ground truth motion
            # This is just a sanity check since we could get directly from the npy files
            #bvhreference = mp.tobvh(self.bvhreference, gt_rot[i], gt_pos[i])
            #listposgt.append(mp.posfrombvh(bvhreference))

        # Compute cross-FGD
        cross_fgd = self.cross_fgd(listpos, sample_names )

        # Compute FGD (whole test set versus whole train set)
        fgd_on_feat = self.fgd(listpos, n_samples=n_samples, n_chunks=n_chunks)

        #histfig = self.getvelhist(rot, vad)
        
        return fgd_on_feat, None, cross_fgd

    def cross_fgd(self, listpos, sample_names):
        # Prepare ground truth data
        std_vec = np.load('./dataset/PTBRGestures/pos_Std.npy')
        mean_vec = np.load('./dataset/PTBRGestures/pos_Mean.npy')
        idx_positions = np.arange(len(mean_vec))
        std_vec[std_vec==0] = 1
        files = glob.glob(os.path.join(self.ground_truth_path, '*.npy'))
        files = [file for file in files if '_un_' not in os.path.basename(file)]
        div_ground_truth, div_test = [], []
        styles = ['p01_e01', 'p01_e02', 'p01_e03', 'p02_e01', 'p02_e02', 'p02_e03']
        for style in styles:
            div_files = [file for file in files if style in os.path.basename(file)]
            div_ground_truth.append(files_to_tensor(div_files, mean_vec, std_vec, n_frames=self.args.num_frames, max_files=1000).to(self.device))

            div_samples = [listpos[i] for i, name in enumerate(sample_names) if style in name]
            div_test.append(self.fgd_prep(div_samples, n_frames=self.args.num_frames).to(self.device))

        cross_fgds = {}
        for gt_style, style_in_gt in zip(div_ground_truth, styles):
            self.fgd_evaluator.reset()

            self.fgd_evaluator.push_real_samples(gt_style)
            for test_style, style_in_test in zip(div_test, styles):
                self.fgd_evaluator.push_generated_samples(test_style)
                fgd_on_feat = self.fgd_evaluator.get_fgd(use_feat_space=True)
                print(f'Cross-FGD gt {style_in_gt} vs test {style_in_test}: {fgd_on_feat:8.3f}')
                cross_fgds.update({'gt {} vs test {}'.format(style_in_gt, style_in_test): fgd_on_feat})

        
        return cross_fgds

    def getvelhist(self, motion, vad):
        joints = self.data.getjoints()
        fvad = vad[:,1:].flatten()
        wvad, wovad = [], []
        for joint, index in joints.items():
            vels = np.sum(np.abs((motion[:,index,:, 1:] - motion[:,index,:, :-1])), axis=1).flatten()
            wvad += list(vels[fvad==1])
            wovad += list(vels[fvad==0])
        n_bins =200
        fig, axs = plt.subplots(1, 1, sharex = True, tight_layout=True, figsize=(20,20))
        axs.hist(wvad, bins = n_bins, histtype='step', label='VAD = 1', linewidth=4, color='red')
        axs.hist(wovad, bins = n_bins, histtype='step', label='VAD = 0', linewidth=4, color='black')
        axs.set_yscale('log')
        return fig
    
    def fgd(self, listpos, listposgt=None, n_samples=100, n_chunks=1):
        # "Direct" ground truth positions
        real_val = make_tensor(self.ground_truth_path, self.args.num_frames, dataset='ptbr',max_files=n_samples, n_chunks=None).to(self.device)

        #gt_data = self.fgd_prep(listposgt).to(self.device)
        test_data = self.fgd_prep(listpos).to(self.device)

        fgd_on_feat = self.run_fgd(real_val, test_data)
        print(f'Sampled to validation: {fgd_on_feat:8.3f}')
        return fgd_on_feat

    def fgd_prep(self, data, n_frames=120, stride=None):
        # Prepare samples for FGD evaluation
        samples = []
        stride = n_frames // 2 if stride is None else stride
        for take in data:
            for i in range(0, len(take) - n_frames, stride):
                sample = take[i:i+n_frames]
                sample = (sample - self.data.pos_mean) / self.data.pos_std
                samples.append(sample)
        return torch.Tensor(samples)

    def run_fgd(self, gt_data, test_data):
        # Run FGD evaluation on the given data
        self.fgd_evaluator.reset()
        self.fgd_evaluator.push_real_samples(gt_data)
        self.fgd_evaluator.push_generated_samples(test_data)
        fgd_on_feat = self.fgd_evaluator.get_fgd(use_feat_space=True)
        return fgd_on_feat
    
def main():
    args = generate_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, None)
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    GeneaEvaluator(args, model, diffusion).eval()


if __name__ == '__main__':
    main()