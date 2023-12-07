# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import gg_collate, ptbr_collate
from soundfile import write as wavwrite
import bvhsdk
import utils.rotation_conversions as geometry
from scipy.signal import savgol_filter

def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    if args.dataset in ['ptbr']:
        fps = 30
        n_joints = 83
        #TODO: change to receive args.bvh_reference_file
        bvhreference = bvhsdk.ReadFile('./dataset/PTBRGestures/motion/bvh_twh/id01_p01_e01_f01.bvh', skipmotion=True)
        collate_fn = ptbr_collate
        split = 'val'
        # iterate over samples in a take
    else:
        raise NotImplementedError
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    print('Loading dataset...')
    data = load_dataset(args, args.batch_size, split)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    #args.guidance_param = 1
    #if args.guidance_param != 1:
    #    model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    all_motions, all_motions_rot = sample(args, model, diffusion, data, collate_fn, n_joints)
    all_motions, all_motions_rot = interpolate(args, all_motions, all_motions_rot, max_chunks_in_take=np.max(data.dataset.samples_per_file))
    savebvh(data, all_motions, all_motions_rot, out_path, fps, bvhreference)

    
def sample(args, model, diffusion, data, collate_fn, n_joints):
    
    total_batches = int(np.round(len(data.dataset.samples_per_file)/args.batch_size))
    max_chunks_in_take = np.max(data.dataset.samples_per_file)

    all_motions = np.zeros(shape=(total_batches*args.batch_size, n_joints, 3, args.num_frames*max_chunks_in_take))
    all_motions_rot = np.zeros(shape=(total_batches*args.batch_size, n_joints, 3, args.num_frames*max_chunks_in_take))
    all_audios = []

    for batch_count in range(total_batches):
        print('### Sampling batch {} of {}'.format(batch_count+1, total_batches))

        chunked_motions = []
        chunked_motions_rot = []
        chunked_audios = []
        for chunk in range(max_chunks_in_take):
            batch = []
            # iterate over each take and append the sample (chunk) to the batch
            first_batch_take = batch_count * args.batch_size
            last_batch_take = (batch_count + 1) * args.batch_size
            for file_idx in range(first_batch_take, last_batch_take):
                # Append dummy samples (1) if the take has less chunks than the max 
                # or (2) if we are in the last batch and the number of takes is smaller than the batch size
                if file_idx < len(data.dataset.samples_per_file):
                    if chunk < data.dataset.samples_per_file[file_idx]:
                        item = chunk + data.dataset.samples_cumulative[file_idx-1] if file_idx > 0 else chunk
                        batch.append(data.dataset.__getitem__(item))
                        continue
                batch.append(data.dataset.__dummysample__())


            _, model_kwargs = collate_fn(batch) # gt_motion: [num_samples(bs), njoints, 1, chunk_len]
            model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()} #seed: [num_samples(bs), njoints, 1, seed_len]

            if chunk == 0: 
                pass #send mean pose
            else:
                model_kwargs['y']['seed'] = sample_out[...,-args.seed_poses:]
                
            print('### Sampling chunk {} of {}'.format(chunk+1, max_chunks_in_take))

            # add CFG scale to batch
            #if args.guidance_param != 1: # default 2.5
            #    model_kwargs['y']['scale'] = torch.ones(num_samples, device=dist_util.dev()) * args.guidance_param

            sample_fn = diffusion.p_sample_loop
            sample_out = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, args.num_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            ) # [num_samples(bs), njoints, 1, chunk_len]

            sample = data.dataset.inv_transform(sample_out.cpu().permute(0, 2, 3, 1)).float() # [num_samples(bs), 1, chunk_len, njoints]


            # Separating positions and rotations
            if args.dataset in ['ptbr']:
                idx_rotations = np.asarray([ [i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5] for i in range(n_joints) ]).flatten()
                idx_positions = np.asarray([ [498 + i*3, 498 + i*3+1, 498 + i*3+2] for i in range(n_joints) ]).flatten()

                sample, sample_rot = sample[..., idx_positions], sample[..., idx_rotations] # sample_rot: [num_samples(bs), 1, chunk_len, n_joints*6]
                
                #rotations
                sample_rot = sample_rot.view(sample_rot.shape[:-1] + (-1, 6)) # [num_samples(bs), 1, chunk_len, n_joints, 6]
                sample_rot = geometry.rotation_6d_to_matrix(sample_rot) # [num_samples(bs), 1, chunk_len, n_joints, 3, 3]
                sample_rot = geometry.matrix_to_euler_angles(sample_rot, "ZXY")[..., [1, 2, 0] ]*180/np.pi # [num_samples(bs), 1, chunk_len, n_joints, 3]
                sample_rot = sample_rot.view(-1, *sample_rot.shape[2:]).permute(0, 2, 3, 1) # [num_samples(bs)*chunk_len, n_joints, 3]

            else:
                raise ValueError(f'Unknown dataset: {args.dataset}')

            #positions
            sample = sample.view(sample.shape[:-1] + (-1, 3))                           # [num_samples(bs), 1, chunk_len, n_joints/3, 3]
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)             # [num_samples(bs), n_joints/3, 3, chunk_len]

            #rot2xyz_pose_rep = 'xyz'
            #rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
            #sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
            #                       jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
            #                       get_rotations_back=False)

            #text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            #all_text += model_kwargs['y'][text_key]
            
            chunked_audios.append(model_kwargs['y']['audio'].cpu().numpy())
            chunked_motions.append(sample.cpu().numpy())
            chunked_motions_rot.append(sample_rot.cpu().numpy())

        #total_num_samples = num_samples * chunks_per_take
        #all_audios = np.concatenate(all_audios, axis=1)
        #all_audios = audio
        b,e = batch_count*args.batch_size, (batch_count+1)*args.batch_size
        all_motions[b:e] = np.concatenate(chunked_motions, axis=3)
        #all_motions = all_motions[:total_num_samples]  # [num_samples(bs), njoints/3, 3, chunk_len*chunks]
        all_motions_rot[b:e] = np.concatenate(chunked_motions_rot, axis=3)
        #all_motions_rot = all_motions_rot[:total_num_samples]  # [num_samples(bs), njoints/3, 3, chunk_len*chunks]
        #all_text = all_text[:total_num_samples]
        #all_lengths = np.concatenate(all_lengths, axis=0)
    return all_motions, all_motions_rot#, all_audios

def interpolate(args, all_motions, all_motions_rot, max_chunks_in_take):
    # Smooth chunk transitions
    inter_range = 10 #interpolation range in frames
    for transition in np.arange(1, max_chunks_in_take-1)*args.num_frames:
        all_motions[..., transition:transition+2] = np.tile(np.expand_dims(all_motions[..., transition]/2 + all_motions[..., transition-1]/2,-1),2)
        all_motions_rot[..., transition:transition+2] = np.tile(np.expand_dims(all_motions_rot[..., transition]/2 + all_motions_rot[..., transition-1]/2,-1),2)
        for i, s in enumerate(np.linspace(0, 1, inter_range-1)):
            forward = transition-inter_range+i
            backward = transition+inter_range-i
            all_motions[..., forward] = all_motions[..., forward]*(1-s) + all_motions[:, :, :, transition-1]*s  
            all_motions[..., backward] = all_motions[..., backward]*(1-s) + all_motions[:, :, :, transition]*s
            all_motions_rot[..., forward] = all_motions_rot[..., forward]*(1-s) + all_motions_rot[:, :, :, transition-1]*s
            all_motions_rot[..., backward] = all_motions_rot[..., backward]*(1-s) + all_motions_rot[:, :, :, transition]*s
            
    all_motions = savgol_filter(all_motions, 9, 3, axis=-1)
    all_motions_rot = savgol_filter(all_motions_rot, 9, 3, axis=-1)
    
    return all_motions, all_motions_rot

def savebvh(data, all_motions, all_motions_rot, out_path, fps, bvhreference):

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions})
    #with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
    #    fw.write('\n'.join(all_text))
    #with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
    #    fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    #if args.dataset in ['genea2023+', 'genea2023']:
    #    skeleton = paramUtil.genea2022_kinematic_chain
    #else:
    #    raise NotImplementedError

    sample_files = []
    num_samples_in_out_file = 7

    #sample_print_template, row_print_template, all_print_template, \
    #sample_file_template, row_file_template, all_file_template = construct_template_variables()


    for i, take in enumerate(range(len(data.dataset.samples_per_file))):
        final_frame = data.dataset.frames[i]
        save_file = 'gen_' + data.dataset.takes[take].name
        print('Saving take {}: {}'.format(i, save_file))
        animation_save_path = os.path.join(out_path, save_file)
        caption = '' # since we are generating a ~1 min long take the caption would be too long
        positions = all_motions[i]
        positions = positions[..., :final_frame]
        positions = positions.transpose(2, 0, 1)
        #plot_3d_motion(animation_save_path + '.mp4', skeleton, positions, dataset=args.dataset, title=caption, fps=fps)
        # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        #saving samples with seed
        #aux_positions = all_sample_with_seed[i]
        #aux_positions = aux_positions.transpose(2, 0, 1)
        #plot_3d_motion(animation_save_path + '_with_seed.mp4', skeleton, aux_positions, dataset=args.dataset, title=caption, fps=fps)

        # Saving generated motion as bvh file
        rotations = all_motions_rot[i] # [njoints/3, 3, chunk_len*chunks]
        rotations = rotations[..., :final_frame]
        rotations = rotations.transpose(2, 0, 1) # [chunk_len*chunks, njoints/3, 3]
        bvhreference.frames = rotations.shape[0]
        for j, joint in enumerate(bvhreference.getlistofjoints()):
            joint.rotation = rotations[:, j, :]
            joint.translation = np.tile(joint.offset, (bvhreference.frames, 1))
        bvhreference.root.translation = positions[:, 0, :]
        #bvhreference.root.children[0].translation = positions[:, 1, :]
        print('Saving bvh file...')
        bvhsdk.WriteBVH(bvhreference, path=animation_save_path, name=None, frametime=1/fps, refTPose=False)

        # Saving audio and joinning it with the mp4 file of generated motion
        #wavfile = animation_save_path + '.wav'
        #mp4file = wavfile.replace('.wav', '.mp4')
        #wavwrite( wavfile, samplerate= 22050, data = all_audios[i])
        #joinaudio = f'ffmpeg -y -loglevel warning -i {mp4file} -i {wavfile} -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k {mp4file[:-4]}_audio.mp4'
        #os.system(joinaudio)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def load_dataset(args, batch_size, split='tst'):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=batch_size,
                              num_frames=args.num_frames,
                              split=split,
                              hml_mode='text_only',
                              step = args.num_frames,
                              use_wavlm=args.use_wavlm,
                              use_vad = args.use_vad,
                              vadfromtext = args.vadfromtext,)
    #data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
