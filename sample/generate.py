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
from data_loaders.tensors import gg_collate
from soundfile import write as wavwrite
import bvhsdk
import utils.rotation_conversions as geometry

def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    if args.dataset in ['genea2023', 'genea2023+']:
        fps = 30
        n_joints = 83
        bvhreference = bvhsdk.ReadFile('./dataset/Genea2023/trn/main-agent/bvh/trn_2023_v0_000_main-agent.bvh', skipmotion=True)
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

    # Hard-coded takes to be generated
    takes_to_generate = np.arange(41)
    chunks_per_take = 14 # TODO: implement for whole take
    num_samples = len(takes_to_generate)


    assert num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    #inputs_i = [155,271,320,400,500,600,700,800,1145,1185]
    

    print('Loading dataset...')
    data = load_dataset(args, num_samples)

    total_num_samples = num_samples * chunks_per_take

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    #iterator = iter(data)

    all_motions = [] #np.zeros(shape=(num_samples, n_joints, 3, args.num_frames*chunks_per_take))
    all_motions_rot = []
    all_lengths = []
    all_text = []
    all_audios = []
    all_gt_motions_rot = []
    all_gt_motions = []
    all_sample_with_seed = []
    all_sample_with_seed_rot = []

    for chunk in range(chunks_per_take): # Motion is generated in chunks, for each chunk we load the corresponding data from the dataset for every take in takes_to_generate

        inputs = []
        for take in takes_to_generate: # For each take we will load the current chunk
            chunk_index = 0 if take == 0 else data.dataset.samples_cumulative[take-1]
            chunk_index += chunk
            if chunk_index >= data.dataset.samples_cumulative[take]:
                raise ValueError(f'Chunk {chunk} is out of range for take {take}.') #i.e., you are getting out of the take
            inputs.append(data.dataset.__getitem__(chunk_index))

        gt_motion, model_kwargs = gg_collate(inputs) # gt_motion: [num_samples(bs), njoints, 1, chunk_len]
        model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()} #seed: [num_samples(bs), njoints, 1, seed_len]

        if chunk == 0: 
            pass #send mean pose
        else:
            model_kwargs['y']['seed'] = sample_out[...,-args.seed_poses:]
            


        print('### Sampling chunk {} of {}'.format(chunk+1, chunks_per_take))

        # add CFG scale to batch
        if args.guidance_param != 1: # default 2.5
            model_kwargs['y']['scale'] = torch.ones(num_samples, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        sample_out = sample_fn(
            model,
            (num_samples, model.njoints, model.nfeats, args.num_frames),
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

        #ground_truth
        gt_motion = data.dataset.inv_transform(gt_motion.cpu().permute(0, 2, 3, 1)).float() # [num_samples(bs), 1, chunk_len, njoints]
        
        #sample_with_seed = data.dataset.inv_transform(model_kwargs['y']['seed'].cpu().permute(1, 2, 0).unsqueeze(1)).float()
        #sample_with_seed = torch.cat([sample_with_seed, sample], dim=2) ## [num_samples(bs), 1, chunk_len, njoints]

        # Separating positions and rotations
        if args.dataset == 'genea2023':
            idx_positions = np.asarray([ [i*6+3, i*6+4, i*6+5] for i in range(n_joints) ]).flatten()
            idx_rotations = np.asarray([ [i*6, i*6+1, i*6+2] for i in range(n_joints) ]).flatten()
            sample, sample_rot = sample[..., idx_positions], sample[..., idx_rotations]

            #rotations
            sample_rot = sample_rot.view(sample_rot.shape[:-1] + (-1, 3))
            sample_rot = sample_rot.view(-1, *sample_rot.shape[2:]).permute(0, 2, 3, 1)
            

            
            gt_motion_pos, gt_motion_rot = gt_motion[..., idx_positions], gt_motion[..., idx_rotations]
            gt_motion_rot = gt_motion_rot.view(gt_motion_rot.shape[:-1] + (-1, 3))
            gt_motion_rot = gt_motion_rot.view(-1, *gt_motion_rot.shape[2:]).permute(0, 2, 3, 1)

            gt_motion_pos = gt_motion_pos.view(gt_motion_pos.shape[:-1] + (-1, 3))
            gt_motion_pos = gt_motion_pos.view(-1, *gt_motion_pos.shape[2:]).permute(0, 2, 3, 1)

            #sample with seed
            #sample_with_seed_pos, sample_with_seed_rot = sample_with_seed[..., idx_positions], sample_with_seed[..., idx_rotations]
            #sample_with_seed_pos = sample_with_seed_pos.view(sample_with_seed_pos.shape[:-1] + (-1, 3))
            #sample_with_seed_pos = sample_with_seed_pos.view(-1, *sample_with_seed_pos.shape[2:]).permute(0, 2, 3, 1)

            #sample_with_seed_rot = sample_with_seed_rot.view(sample_with_seed_rot.shape[:-1] + (-1, 3))
            #sample_with_seed_rot = sample_with_seed_rot.view(-1, *sample_with_seed_rot.shape[2:]).permute(0, 2, 3, 1)


        elif args.dataset == 'genea2023+':
            idx_rotations = np.asarray([ [i*9, i*9+1, i*9+2, i*9+3, i*9+4, i*9+5] for i in range(n_joints) ]).flatten()
            idx_positions = np.asarray([ [i*9+6, i*9+7, i*9+8] for i in range(n_joints) ]).flatten()
            sample, sample_rot = sample[..., idx_positions], sample[..., idx_rotations] # sample_rot: [num_samples(bs), 1, chunk_len, n_joints*6]
            
            #rotations
            sample_rot = sample_rot.view(sample_rot.shape[:-1] + (-1, 6)) # [num_samples(bs), 1, chunk_len, n_joints, 6]
            sample_rot = geometry.rotation_6d_to_matrix(sample_rot) # [num_samples(bs), 1, chunk_len, n_joints, 3, 3]
            sample_rot = geometry.matrix_to_euler_angles(sample_rot, "ZXY")[..., [1, 2, 0] ]*180/np.pi # [num_samples(bs), 1, chunk_len, n_joints, 3]
            sample_rot = sample_rot.view(-1, *sample_rot.shape[2:]).permute(0, 2, 3, 1) # [num_samples(bs)*chunk_len, n_joints, 3]

            #ground truth
            gt_motion_pos, gt_motion_rot = gt_motion[..., idx_positions], gt_motion[..., idx_rotations]
            gt_motion_rot = gt_motion_rot.view(gt_motion_rot.shape[:-1] + (-1, 6)) # [num_samples(bs), 1, chunk_len, n_joints, 6]
            gt_motion_rot = geometry.rotation_6d_to_matrix(gt_motion_rot) # [num_samples(bs), 1, chunk_len, n_joints, 3, 3]
            gt_motion_rot = geometry.matrix_to_euler_angles(gt_motion_rot, "ZXY")[..., [1, 2, 0] ]*180/np.pi # [num_samples(bs), 1, chunk_len, n_joints, 3]
            gt_motion_rot = gt_motion_rot.view(-1, *gt_motion_rot.shape[2:]).permute(0, 2, 3, 1)

            gt_motion_pos = gt_motion_pos.view(gt_motion_pos.shape[:-1] + (-1, 3))
            gt_motion_pos = gt_motion_pos.view(-1, *gt_motion_pos.shape[2:]).permute(0, 2, 3, 1)
            
        else:
            raise ValueError(f'Unknown dataset: {args.dataset}')

        #positions
        sample = sample.view(sample.shape[:-1] + (-1, 3))                           # [num_samples(bs), 1, chunk_len, n_joints/3, 3]
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)             # [num_samples(bs), n_joints/3, 3, chunk_len]

        rot2xyz_pose_rep = 'xyz'
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)

        text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
        all_text += model_kwargs['y'][text_key]
            
        all_audios.append(model_kwargs['y']['audio'].cpu().numpy())
        all_motions.append(sample.cpu().numpy())
        all_motions_rot.append(sample_rot.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        all_gt_motions_rot.append(gt_motion_rot.cpu().numpy())
        all_gt_motions.append(gt_motion_pos.cpu().numpy())

        #all_sample_with_seed.append(sample_with_seed_pos.cpu().numpy())
        #all_sample_with_seed_rot.append(sample_with_seed_rot.cpu().numpy())


    all_audios = np.concatenate(all_audios, axis=1)
    all_motions = np.concatenate(all_motions, axis=3)
    all_motions = all_motions[:total_num_samples]  # [num_samples(bs), njoints/3, 3, chunk_len*chunks]
    all_motions_rot = np.concatenate(all_motions_rot, axis=3)
    all_motions_rot = all_motions_rot[:total_num_samples]  # [num_samples(bs), njoints/3, 3, chunk_len*chunks]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    all_gt_motions_rot = np.concatenate(all_gt_motions_rot, axis=3) # [num_samples(bs), njoints/3, 3, chunk_len*chunks]
    all_gt_motions = np.concatenate(all_gt_motions, axis=3)

    #all_sample_with_seed = np.concatenate(all_sample_with_seed, axis=3)
    #all_sample_with_seed_rot = np.concatenate(all_sample_with_seed_rot, axis=3)
    
    #gt_motion = data.dataset.inv_transform(gt_motion.cpu().permute(0, 2, 3, 1))
    #gt_motion = gt_motion[..., idx_rotations]
    #gt_motion = gt_motion.view(gt_motion.shape[:-1] + (-1, 3))
    #gt_motion = gt_motion.view(-1, *gt_motion.shape[2:]).numpy()
    

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': len(takes_to_generate), 'num_chunks': chunks_per_take})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    if args.dataset in ['genea2023+', 'genea2023']:
        skeleton = paramUtil.genea2022_kinematic_chain
    else:
        raise NotImplementedError

    sample_files = []
    num_samples_in_out_file = 7

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables()


    for i, take in enumerate(takes_to_generate):
        save_file = data.dataset.takes[take][0]
        print('Saving take {}: {}'.format(i, save_file))
        animation_save_path = os.path.join(out_path, save_file)
        caption = '' # since we are generating a ~1 min long take the caption would be too long
        positions = all_motions[i]
        positions = positions.transpose(2, 0, 1)
        plot_3d_motion(animation_save_path + '.mp4', skeleton, positions, dataset=args.dataset, title=caption, fps=fps)
        # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        #saving samples with seed
        #aux_positions = all_sample_with_seed[i]
        #aux_positions = aux_positions.transpose(2, 0, 1)
        #plot_3d_motion(animation_save_path + '_with_seed.mp4', skeleton, aux_positions, dataset=args.dataset, title=caption, fps=fps)

        # Saving generated motion as bvh file
        rotations = all_motions_rot[i] # [njoints/3, 3, chunk_len*chunks]
        rotations = rotations.transpose(2, 0, 1) # [chunk_len*chunks, njoints/3, 3]
        bvhreference.frames = rotations.shape[0]
        for j, joint in enumerate(bvhreference.getlistofjoints()):
            joint.rotation = rotations[:, j, :]
            joint.translation = np.tile(joint.offset, (bvhreference.frames, 1))
        bvhreference.root.translation = positions[:, 0, :]
        #bvhreference.root.children[0].translation = positions[:, 1, :]
        print('Saving bvh file...')
        bvhsdk.WriteBVH(bvhreference, path=animation_save_path, name=None, frametime=1/fps, refTPose=False)

        # Saving gorund truth motions as bvh file
        rotations = all_gt_motions_rot[i] # [njoints/3, 3, chunk_len*chunks]
        rotations = rotations.transpose(2, 0, 1) # [chunk_len*chunks, njoints/3, 3]
        positions = all_gt_motions[i]
        positions = positions.transpose(2, 0, 1)
        bvhreference.frames = rotations.shape[0]
        for j, joint in enumerate(bvhreference.getlistofjoints()):
            joint.rotation = rotations[:, j, :]
            joint.translation = np.tile(joint.offset, (bvhreference.frames, 1))
        bvhreference.root.translation = positions[:, 0, :]
        #bvhreference.root.children[0].translation = positions[:, 1, :]
        # Move the ground truth motion to the position of the interlocutor
        # We are doing this to compare the generated motion with the ground truth motion using the oficial genea visualization
        # Get rotation matrix to be performed
        matrix = bvhsdk.mathutils.matrixRotation(180, y=1, shape=4)
        # Get joint's current global rotation matrix
        transmat = np.array([bvhreference.root.children[0].getGlobalTransform(i) for i in range(bvhreference.frames) ])
        # Apply rotation
        newmat = np.array([np.dot(matrix, transmat[i]) for i in range(bvhreference.frames)])
        # Get new local euler angles
        bvhreference.root.children[0].rotation = np.array([bvhsdk.mathutils.eulerFromMatrix(newmat[i], bvhreference.root.children[0].order)[0] for i in range(bvhreference.frames)])
        # Adjust root position
        hips_height =  np.asarray([0,91.5,0])
        distance = np.array([0,0,150])
        bvhreference.root.translation = np.array([newmat[i][:,-1][:-1] -hips_height + distance for i in range(bvhreference.frames)])
        print('Saving bvh file...')
        bvhsdk.WriteBVH(bvhreference, path=animation_save_path + '_gt', name=None, frametime=1/fps, refTPose=False)

        # Saving sample with seed as bvh file
        #rotations = all_sample_with_seed_rot[i] # [njoints/3, 3, chunk_len*chunks]
        #rotations = rotations.transpose(2, 0, 1) # [chunk_len*chunks, njoints/3, 3]
        #bvhreference.frames = rotations.shape[0]
        #for j, joint in enumerate(bvhreference.getlistofjoints()):
        #    joint.rotation = rotations[:, j, :]
        #    joint.translation = np.tile(joint.offset, (bvhreference.frames, 1))
        #print('Saving bvh file...')
        #bvhsdk.WriteBVH(bvhreference, path=animation_save_path + '_with_seed', name=None, frametime=1/fps, refTPose=False)

        # Saving audio and joinning it with the mp4 file of generated motion
        wavfile = animation_save_path + '.wav'
        mp4file = wavfile.replace('.wav', '.mp4')
        wavwrite( wavfile, samplerate= 22050, data = all_audios[i])
        joinaudio = f'ffmpeg -y -loglevel warning -i {mp4file} -i {wavfile} -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k {mp4file[:-4]}_audio.mp4'
        os.system(joinaudio)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []

    return sample_files



def construct_template_variables():
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
    sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
    row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
    all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, batch_size):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=batch_size,
                              num_frames=args.num_frames,
                              split='val',
                              hml_mode='text_only')
    #data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
