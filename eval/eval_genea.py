from data_loaders.gesture.scripts import motion_process as mp
from data_loaders.get_data import get_dataset_loader
import numpy as np
from tqdm import tqdm
from utils import dist_util
import torch
import bvhsdk
from evaluation_metric.embedding_space_evaluator import EmbeddingSpaceEvaluator
from evaluation_metric.train_AE import make_tensor

# Imports for calling from command line
from utils.parser_util import generate_args
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip


class GeneaEvaluator:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.dataloader = get_dataset_loader(name=args.dataset, 
                                        batch_size=args.batch_size, 
                                        num_frames=args.num_frames, 
                                        step=args.num_frames, #no overlap
                                        use_wavlm=args.use_wavlm, 
                                        use_vad=args.use_vad, 
                                        vadfromtext=args.vadfromtext,
                                        split='val')
        self.data = self.dataloader.dataset
        self.bvhreference = bvhsdk.ReadFile(args.bvh_reference_file, skipmotion=True)
        idx_positions, _ = mp.get_indexes('genea2023') # hard-coded 'genea2023' because std and mean vec are computed for this representation
        self.std = self.data.std[idx_positions]
        self.mean = self.data.mean[idx_positions]
        self.fgd_evaluator = EmbeddingSpaceEvaluator(args.fgd_embedding, args.num_frames, dist_util.dev())


    def eval(self):
        rot, gt_rot, pos, gt_pos  = self.sampleval()
        pos, rot = mp.filter_and_interp(rot, pos, num_frames=self.num_frames)
        # Transform to BVH and get positions of sampled motion
        bvhreference = mp.tobvh(self.bvhreference, rot, pos)
        pos = mp.posfrombvh(bvhreference)
        # Transform to BVH and get positions of ground truth motion
        # This is just a sanity check since we could get directly from the npy files
        bvhreference = mp.tobvh(self.bvhreference, gt_rot, gt_pos)
        gt_pos = mp.posfrombvh(bvhreference)
        # "Direct" ground truth positions
        real_val = make_tensor(f'./dataset/Genea2023/val/main-agent/motion_npy_rotpos', self.args.num_frames, max_files=64).to(dist_util.dev())

        fgd_on_feat = self.run_fgd(pos, gt_pos)
        print(f'Sampled to validation from pipeline: {fgd_on_feat:8.3f}')

        fgd_on_feat = self.run_fgd(pos, real_val)
        print(f'Sampled to validation: {fgd_on_feat:8.3f}')

        fgd_on_feat = self.run_fgd(gt_pos, real_val)
        print(f'Validation from pipeline to validation (should be zero): {fgd_on_feat:8.3f}')
        
        return None

    def sampleval(self, samples=None, chunks=None):
        n_samples = samples if samples else len(self.data.takes)
        n_chunks = chunks if chunks else np.min(self.data.samples_per_file)
        assert n_chunks <= np.min(self.data.samples_per_file) # assert that we don't go over the number of chunks per file
        allsampledmotion = []
        allsampleposition = []
        allgtmotion = []
        allgtposition = []
        print('Evaluating validation set')
        for idx in tqdm(range(chunks)):
            batch = self.data.getvalbatch(idx)
            gt_motion, model_kwargs = self.dataloader.collate_fn(batch) # gt_motion: [num_samples(bs), njoints, 1, chunk_len]
            model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()} #seed: [num_samples(bs), njoints, 1, seed_len]
            if idx > 0:
                model_kwargs['y']['seed'] = sample_out[...,-self.data.n_seed_poses:]
            sample_fn = self.diffusion.p_sample_loop
            sample_out = sample_fn(
                self.model,
                (self.num_samples, self.model.njoints, self.model.nfeats, self.num_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            ) # [num_samples(bs), njoints, 1, chunk_len]

            sample = self.data.inv_transform(sample_out.cpu().permute(0, 2, 3, 1)).float() # [num_samples(bs), 1, chunk_len, njoints]
            gt_motion = self.data.inv_transform(gt_motion.cpu().permute(0, 2, 3, 1)).float() # [num_samples(bs), 1, chunk_len, njoints]

            # Split the data into positions and rotations
            gt_pos, gt_rot = mp.split_pos_rot(self.args.dataset, gt_motion)
            sample_pos, sample_rot = mp.split_pos_rot(self.args.dataset, sample)

            # Convert numpy array to euler angles
            if self.args.dataset == 'genea2023':
                gt_rot = mp.rot6d_to_euler(gt_rot)
                sample_rot = mp.rot6d_to_euler(sample_rot)

            sample_pos = sample_pos.view(sample.shape[:-1] + (-1, 3))                # [num_samples(bs), 1, chunk_len, n_joints/3, 3]
            sample_pos = sample_pos.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1) 
            gt_pos = gt_pos.view(gt_motion.shape[:-1] + (-1, 3))                # [num_samples(bs), 1, chunk_len, n_joints/3, 3]
            gt_pos = gt_pos.view(-1, *gt_motion.shape[2:]).permute(0, 2, 3, 1)

            allsampledmotion.append(sample_rot.cpu().numpy())
            allgtmotion.append(gt_rot.cpu().numpy())
            allsampleposition.append(sample_pos.cpu().numpy())
            allgtposition.append(gt_pos.cpu().numpy())

        allsampledmotion = np.concatenate(allsampledmotion, axis=3)
        allgtmotion = np.concatenate(allgtmotion, axis=3)
        allsampleposition = np.concatenate(allsampleposition, axis=3)
        allgtposition = np.concatenate(allgtposition, axis=3)

        return allsampledmotion, allgtmotion, allsampleposition, allgtposition
    
    def fgd_prep(self, data, n_frames=120, stride=None):
        samples = []
        stride = n_frames // 2 if stride is None else stride
        for i in range(0, len(data) - n_frames, stride):
            sample = data[i:i+n_frames]
            sample = (sample - self.mean) / self.std
            samples.append(sample)
        return torch.Tensor(samples)

    def run_fgd(self, gt_data, test_data):
        gt_data = self.fgd_prep(gt_data)
        test_data = self.fgd_prep(test_data)
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
    GeneaEvaluator(args, model).eval()


if __name__ == '__main__':
    main()