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
import matplotlib.pyplot as plt

def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    if args.dataset in ['genea2022', 'genea2023']:
        #n_frames = 200
        fps = 30
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


    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args)
    total_num_samples = args.num_samples * args.num_repetitions

    #takes_to_generate = [ 5 ,  9, 20]
    takes_to_generate = [ 5 ]

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    audios = []
    motions = []
    motions_rot = []
    gt_motions_rot = []
    all_text = []
    all_lengths = []
    for take in takes_to_generate:
        print("### Sampling take #{0}: {1}".format(take,data.dataset.takes[take][0]))
        start = 0 if take == 0 else data.dataset.samples_cumulative[take-1]
        end = data.dataset.samples_cumulative[take]
        inputs = [ data.dataset.__getitem__(input) for input in range(start, end) ]
        gt_motion, model_kwargs = gg_collate(inputs)
        model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(len(inputs), device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (len(inputs), model.njoints, model.nfeats, args.num_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        
        #teste = sample.cpu()
        #teste_gt = gt_motion.cpu()
        #print(teste.shape)
        #teste2 = np.concatenate((teste[0,:,:,:], teste[1,:,:,:], teste[2,:,:,:]), axis=2)
        #teste2_gt = np.concatenate((teste_gt[0,:,:,:], teste_gt[1,:,:,:], teste_gt[2,:,:,:]), axis=2)
        #for i in range(3):
        #    plt.plot(teste2[6+i,0,:])
        #    plt.plot(teste2_gt[6+i,0,:], linestyle='dashed')
        #plt.savefig('teste.png')

        
        if model.data_rep == 'genea_vec':
            n_joints = 83
            sample = data.dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            idx_positions = np.asarray([ [i*6+3, i*6+4, i*6+5] for i in range(n_joints) ]).flatten()
            idx_rotations = np.asarray([ [i*6, i*6+1, i*6+2] for i in range(n_joints) ]).flatten()
            sample, sample_rot = sample[..., idx_positions], sample[..., idx_rotations]
            #position
            sample = sample.view(sample.shape[:-1] + (-1, 3))
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
            motions.append(sample.cpu().numpy().reshape( n_joints, 3, -1 ))
            #rotations
            sample_rot = sample_rot.view(sample_rot.shape[:-1] + (-1, 3))
            sample_rot = sample_rot.view(-1, *sample_rot.shape[2:]).permute(0, 2, 3, 1)
            motions_rot.append(sample_rot.cpu().numpy().reshape( n_joints, 3, -1 ))
            #ground truth
            gt_motion = data.dataset.inv_transform(gt_motion.cpu().permute(0, 2, 3, 1))
            gt_motion = gt_motion[..., idx_rotations]
            gt_motion = gt_motion.view(gt_motion.shape[:-1] + (-1, 3))
            gt_motion = gt_motion.view(-1, *gt_motion.shape[2:]).numpy()
            gt_motions_rot.append(gt_motion.reshape( -1, n_joints, 3 ))
        else:
            raise NotImplementedError

        all_text += model_kwargs['y']['text']
        audios.append(model_kwargs['y']['audio'].cpu().numpy().reshape(-1))
        all_lengths.append(gt_motion.shape[0])

    #audios = np.concatenate(audios, axis = 0)
    #motions = np.concatenate(motions, axis = 0)
    #motions_rot = np.concatenate(motions_rot, axis = 0)
    #gt_motions_rot = np.concatenate(gt_motions_rot, axis = 0)
  
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': motions, 'text': all_text, 'lengths': all_lengths,
             'takes_generated': takes_to_generate})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    if args.dataset in ['genea2022', 'genea2023']:
        skeleton = paramUtil.genea2022_kinematic_chain
    else:
        raise NotImplementedError

    sample_files = []
    num_samples_in_out_file = 7

    #sample_print_template, row_print_template, all_print_template, \
    #sample_file_template, row_file_template, all_file_template = construct_template_variables()


    for sample_i in range(len(takes_to_generate)):
        save_file = data.dataset.takes[takes_to_generate[sample_i]][0]
        print('Saving sample {}: {}'.format(sample_i, save_file))
        animation_save_path = os.path.join(out_path, save_file)

        caption = '' # since we are generating a ~1 min long take the caption would be too long
        motion = motions[sample_i].transpose(2, 0, 1)
        motion_rot = motions_rot[sample_i].transpose(2, 0, 1)
        plot_3d_motion(animation_save_path + '.mp4', skeleton, motion, dataset=args.dataset, title=caption, fps=fps)
        # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        # Saving generated motion as bvh file
        print('motion shape')
        print(motion_rot.shape)
        bvhreference.frames = motion_rot.shape[0]
        for i, joint in enumerate(bvhreference.getlistofjoints()):
            joint.rotation = motion_rot[:, i, :]
            joint.translation = np.tile(joint.offset, (bvhreference.frames, 1))
        print('Saving bvh file...')
        bvhsdk.WriteBVH(bvhreference, path=animation_save_path, name=None, frametime=1/fps, refTPose=False)

        # Saving ground truth motion as bvh file
        for i, joint in enumerate(bvhreference.getlistofjoints()):
            joint.rotation = gt_motions_rot[sample_i][:, i, :]
            joint.translation = np.tile(joint.offset, (bvhreference.frames, 1))
        bvhsdk.WriteBVH(bvhreference, path=animation_save_path + '_gt', name=None, frametime=1/fps, refTPose=False)

        # Saving audio and joinning it with the mp4 file of generated motion
        wavfile = animation_save_path + '.wav'
        mp4file = wavfile.replace('.wav', '.mp4')
        wavwrite( wavfile, samplerate= 22050, data = audios[sample_i])
        joinaudio = f'ffmpeg -y -loglevel warning -i {mp4file} -i {wavfile} -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k {mp4file[:-4]}_audio.mp4'
        os.system(joinaudio)

    ###################################################################
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


def load_dataset(args):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=args.num_frames,
                              split='val',
                              hml_mode='text_only')
    #data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
