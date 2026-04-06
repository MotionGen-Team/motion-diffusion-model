from sample.predict import get_args
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

from visualize.joints2smpl.fit_seq import device


def main():
    args=get_args()
    fixseed(args.seed)
    model,diffusion=load_model(args)
    text=args.text
    text_embed=model.encode_text([text]).to(device)
    sample=sample_motion(model,diffusion,text_embed,args)
    sample=sample.cpu().numpy()
    print(sample.shape)
    np.save("result.npy",sample)

def load_model(args):
    model,diffusion=create_model_and_diffusion(args)
    load_saved_model(model,args.model_path)
    model.eval()
    model.to(device)
    return model,diffusion

def sample_motion(model, diffusion, text_embed, args):
    shape=(1,model.njoints,model.nfeats,args.nframes)
    model_kwargs={
        "y":{
            "text_embed":text_embed
        }
    }
    if args.scale!=1:
        model=ClassifierFreeSampleModel(model)
        model_kwargs["y"]["scale"]=torch.ones(1,device=device)*args.scale
    sample=diffusion.p_sample_loop(
        model,
        shape,
        model_kwargs=model_kwargs
    )
    return sample

parents=[
    -1,# root（没有父节点）
    0,# joint1 的父节点是 root
    1,# joint2 的父节点是 joint1
    0# joint3 的父节点是 root
]
def recover_from_ric(relative_pos,root_pos,parents):
    """
       relative_pos: (J, 3)
       root_pos: (3,)
       parents: list of length J
    """
    j=len(parents)
    positions=np.zeros((j,3))
    positions[0]=root_pos
    for i in range(1,j):
        parent=parents[i]
        positions[i]=relative_pos[i]+positions[parent]
    return positions