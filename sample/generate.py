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
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel, AutoRegressiveSampler
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric, get_target_location, sample_goal
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate
from moviepy.editor import clips_array


def main(args=None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = generate_args()
        #generate_args()：读取命令行参数（模型路径、文本等）
        #fixseed：固定随机种子（保证结果可复现）
    fixseed(args.seed)
    out_path = args.output_dir
    '''控制生成的动作：

    关节数（骨架结构）
    最大帧数
    帧率（FPS）
    最终生成长度 
    本质：
    控制生成的时间序列长度'''
    n_joints = 22 if args.dataset == 'humanml' else 21
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    '''
    | 输入方式              | 说明           |
| ----------------- | ------------ |
| text_prompt       | 单条文本         |
| input_text        | txt文件        |
| dynamic_text_path | 每一帧不同文本（很高级） |
| action_name       | 动作标签         |

    '''
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    if args.context_len > 0:
        is_using_data = True  # For prefix completion, we need to sample a prefix
    dist_util.setup_dist(args.device)
    #构建输出路径
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')
        elif args.dynamic_text_path != '':
            out_path += '_' + os.path.basename(args.dynamic_text_path).replace('.txt', '').replace(' ', '_').replace('.', '')

    # this block must be called BEFORE the dataset is loaded
    texts = None
    if args.text_prompt != '':
        texts = [args.text_prompt] * args.num_samples
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.dynamic_text_path != '':
        assert os.path.exists(args.dynamic_text_path)
        assert args.autoregressive, "Dynamic text sampling is only supported with autoregressive sampling."
        with open(args.dynamic_text_path, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        n_frames = len(texts) * args.pred_len  # each text prompt is for a single prediction
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)

    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    #加载数据集
    '''  注意：

    即使是生成，也会加载 dataset，因为：

    需要 normalization / skeleton / tokenizer 等'''
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    #创建模型 + diffusion
    ''' 这是核心：

    model：Transformer / UNet（生成器）
    diffusion：扩散过程（采样逻辑）'''
    model, diffusion = create_model_and_diffusion(args, data)
    #选择采样方式默认：DDPM sampling
    sample_fn = diffusion.p_sample_loop
    #自回归生成（一步步生成）
    if args.autoregressive:
        sample_cls = AutoRegressiveSampler(args, sample_fn, n_frames)
        sample_fn = sample_cls.sample

    print(f"Loading checkpoints from [{args.model_path}]...")
    #加载模型权重,就是 .pt checkpoint
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    if args.guidance_param != 1:
        #CFG（Classifier-Free Guidance）,x = x_cond + scale * (x_cond - x_uncond),scale越大 → 越“听话”
        ''' | 模式     | 学的东西     |
            | ------ | -------- |
            | cond   | “怎么符合文本” |
            | uncond | “自然动作分布” |
            eps_cond - eps_uncond是“文本带来的偏移量”，+ scale *相当于：“把文本影响放大”，CFG = 用无条件当 baseline，用条件做方向修正
        '''
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    motion_shape = (args.batch_size, model.njoints, model.nfeats, n_frames)
    #准备输入（model_kwargs）
    #用数据集
    if is_using_data:
        iterator = iter(data)
        input_motion, model_kwargs = next(iterator)
        input_motion = input_motion.to(dist_util.dev())
        if texts is not None:
            model_kwargs['y']['text'] = texts
    #用文本生成
    else:
        '''这一步做：
        padding
        mask
        text embedding '''
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            # t2m
            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        else:
            # a2m
            action = data.dataset.action_name_to_action(action_text)
            collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
                            arg, one_action, one_action_text in zip(collate_args, action, action_text)]
        _, model_kwargs = collate(collate_args)
    ##文本编码优化,提前encode，避免重复计算
    model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}
    init_image = None    
    
    all_motions = []
    all_lengths = []
    all_text = []

    # add CFG scale to batch
    if args.guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    
    if 'text' in model_kwargs['y'].keys():
        # encoding once instead of each iteration saves lots of time
        model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])
    
    if args.dynamic_text_path != '':
        # Rearange the text to match the autoregressive sampling - each prompt fits to a single prediction
        # Which is 2 seconds of motion by default
        model_kwargs['y']['text'] = [model_kwargs['y']['text']] * args.num_samples
        if args.text_encoder_type == 'bert':
            model_kwargs['y']['text_embed'] = (model_kwargs['y']['text_embed'][0].unsqueeze(0).repeat(args.num_samples, 1, 1, 1), 
                                               model_kwargs['y']['text_embed'][1].unsqueeze(0).repeat(args.num_samples, 1, 1))
        else:
            raise NotImplementedError('DiP model only supports BERT text encoder at the moment. If you implement this, please send a PR!')
    #进入采样循环（最核心）,每个prompt生成多个样本
    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')
        #diffusion采样,noise → 一步步去噪 → motion
        sample = sample_fn(
            model,
            motion_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=init_image,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Recover XYZ *positions* from humanml vector representation,数据后处理（非常重要）
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            print("sample stats:", sample.min().item(), sample.max().item(), sample.mean().item())
            #表示转换,模型内部表示 → 真实关节坐标
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        #转成 xyz,最终得到：[batch, joints, xyz, frames]
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]

        all_motions.append(sample.cpu().numpy())
        _len = model_kwargs['y']['lengths'].cpu().numpy()
        if 'prefix' in model_kwargs['y'].keys():
            _len[:] = sample.shape[-1]
        all_lengths.append(_len)

        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    #保存数据
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    if args.dynamic_text_path != '':
        text_file_content = '\n'.join(['#'.join(s) for s in all_text])
    else:
        text_file_content = '\n'.join(all_text)
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write(text_file_content)
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)
    max_vis_samples = 6
    num_vis_samples = min(args.num_samples, max_vis_samples)
    animations = np.empty(shape=(args.num_samples, args.num_repetitions), dtype=object)
    max_length = max(all_lengths)

    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            if args.dynamic_text_path != '':  # caption per frame
                assert type(caption) == list
                caption_per_frame = []
                for c in caption:
                    caption_per_frame += [c] * args.pred_len
                caption = caption_per_frame

            
            # Trim / freeze motion if needed
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:max_length]
            if motion.shape[0] > length:
                motion[length:-1] = motion[length-1]  # duplicate the last frame to end of motion, so all motions will be in equal length

            save_file = sample_file_template.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            gt_frames = np.arange(args.context_len) if args.context_len > 0 and not args.autoregressive else []
            #每个样本生成一个视频
            animations[sample_i, rep_i] = plot_3d_motion(animation_save_path, 
                                                         skeleton, motion, dataset=args.dataset, title=caption, 
                                                         fps=fps, gt_frames=gt_frames)
            print("motion shape:", motion.shape)
            print("motion first frame:", motion[0][:10])
            rep_files.append(animation_save_path)

    save_multiple_samples(out_path, {'all': all_file_template}, animations, fps, max(list(all_lengths) + [n_frames]))

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')

    return out_path

#每3个样本拼一个视频
def save_multiple_samples(out_path, file_templates,  animations, fps, max_frames, no_dir=False):
    
    num_samples_in_out_file = 3
    n_samples = animations.shape[0]
    
    for sample_i in range(0,n_samples,num_samples_in_out_file):
        last_sample_i = min(sample_i+num_samples_in_out_file, n_samples)
        all_sample_save_file = file_templates['all'].format(sample_i, last_sample_i-1)
        if no_dir and n_samples <= num_samples_in_out_file:
            all_sample_save_path = out_path
        else:
            all_sample_save_path = os.path.join(out_path, all_sample_save_file)
            print(f'saving {os.path.split(out_path)[1]}/{all_sample_save_file}')

        clips = clips_array(animations[sample_i:last_sample_i])
        clips.duration = max_frames/fps
        
        # import time
        # start = time.time()
        clips.write_videofile(all_sample_save_path, fps=fps, threads=4, logger=None)
        # print(f'duration = {time.time()-start}')
        
        for clip in clips.clips: 
            # close internal clips. Does nothing but better use in case one day it will do something
            clip.close()
        clips.close()  # important
 
#只是控制文件名格式
def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='train' if args.pred_len > 0 else 'text_only',  # We need to sample a prefix from the dataset
                              fixed_len=args.pred_len + args.context_len, pred_len=args.pred_len, device=dist_util.dev())
    data.fixed_length = n_frames
    return data


def is_substr_in_list(substr, list_of_strs):
    return np.char.find(list_of_strs, substr) != -1  # [substr in string for string in list_of_strs]

if __name__ == "__main__":
    main()
