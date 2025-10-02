
import os
import cv2
import dnnlib
import torch
import pickle
import mrcfile
import pyshtools
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

import math
from torch_utils import misc
import random

import imageio
from torchvision import transforms
from torchvision.utils import make_grid

from matplotlib import pyplot as plt
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from training.triplane_mod import TriPlaneGenerator_base



import warnings
warnings.filterwarnings("ignore")
from torch import nn
from torch.nn import functional as F
imageio.plugins.freeimage.download()
import lpips

import argparse
parser = argparse.ArgumentParser(description='parameter for evaluation')
parser.add_argument('--mode', help='one from base, eyes, nose, chin')
parser.add_argument('--network', help='network for evaluate')
parser.add_argument('--seed',default=0, type=int,help='set a seed')
parser.add_argument('--overlap_weight', type=float,help='set ovlp weight')
parser.add_argument('--target_image', help='set target', default='70')

args = parser.parse_args()

device = "cuda"






def calc_dice_loss(pred, target, smooth=1):
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
    
    pred = torch.nn.functional.softmax(pred, dim=1)
    
    pred_flat = pred.contiguous().reshape(pred.size(0), pred.size(1), -1)
    target_flat = target_one_hot.contiguous().reshape(target_one_hot.size(0), target_one_hot.size(1), -1)
    
    intersection = (pred_flat * target_flat).sum(2)
    denominator = pred_flat.sum(2) + target_flat.sum(2)
    
    dice_score = (2. * intersection + smooth) / (denominator + smooth)
    dice_loss = 1 - dice_score.mean(1)
    return dice_loss.mean()






@torch.no_grad()
def render_tensor(img: torch.Tensor, normalize: bool = True, nrow: int = 8) -> Image.Image:
    if type(img) == list:
        img = torch.cat(img, dim=0).expand(-1, 3, -1, -1)
    elif len(img.shape) == 3:
        img = img.expand(3, -1, -1)
    elif len(img.shape) == 4:
        img = img.expand(-1, 3, -1, -1)
    
    img = img.squeeze()
    
    if normalize:
        img = img / 2 + .5
        
    
    if len(img.shape) == 3:
        return Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    elif len(img.shape) == 2:
        return Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8))
    elif len(img.shape) == 4:
        return Image.fromarray((make_grid(img, nrow=nrow).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    

    

@torch.no_grad()
def vis_parsing_maps(im: torch.Tensor, inverse: bool = False, argmax: bool = True):
    if args.mode == 'base':
        part_colors = [
        [0, 0, 0], # Background
        [127, 212, 255], # Skin
        [255, 212, 255], # Eye Brow
        [255, 255, 170], # Eye
        [255, 255, 130], # Glass
        [76, 153, 0], # Ear
        [0, 255, 170], # Ear Ring
        [244, 124, 244], # Nose
        [30, 162, 230], # Mouth
        [127, 255, 255], # Lip
        [127, 170, 255], # Neck
        [85, 0, 255], # Neck-lace
        [255, 170, 127], # Cloth
        [212, 127, 255], # Hair
        [0, 170, 255], # Hat
        [255, 255, 255], # nasal ala
        [0,0,100], #iris
    ]
    elif args.mode == 'eyes':
        part_colors = [
        [0, 0, 0], # Background
        [127, 212, 255], # Skin
        [255, 212, 255], # Eye Brow
        [255, 255, 170], # Eye
        [255, 255, 130], # Glass
        [76, 153, 0], # Ear
        [0, 255, 170], # Ear Ring
        [244, 124, 244], # Nose
        [30, 162, 230], # Mouth
        [127, 255, 255], # Lip
        [127, 170, 255], # Neck
        [85, 0, 255], # Neck-lace
        [255, 170, 127], # Cloth
        [212, 127, 255], # Hair
        [0, 170, 255], # Hat
        [255, 255, 255], # nasal ala
        [0,0,100], #iris
        [127,127,127], # pupil
        [163,73,164] # eyelid
        #[181,230,29], # nose ridge 
        #[136,0,21] # nostril
    ]
    elif args.mode == 'nose':
        part_colors = [
        [0, 0, 0], # Background
        [127, 212, 255], # Skin
        [255, 212, 255], # Eye Brow
        [255, 255, 170], # Eye
        [255, 255, 130], # Glass
        [76, 153, 0], # Ear
        [0, 255, 170], # Ear Ring
        [244, 124, 244], # Nose
        [30, 162, 230], # Mouth
        [127, 255, 255], # Lip
        [127, 170, 255], # Neck
        [85, 0, 255], # Neck-lace
        [255, 170, 127], # Cloth
        [212, 127, 255], # Hair
        [0, 170, 255], # Hat
        [255, 255, 255], # nasal ala
        [0,0,100], #iris
        #[127,127,127], # pupil
        #[163,73,164] # eyelid
        [181,230,29], # nose ridge
        [136,0,21] # nostril
    ]
    elif args.mode == 'chin':
        part_colors = [
        [0, 0, 0], # Background
        [127, 212, 255], # Skin
        [255, 212, 255], # Eye Brow
        [255, 255, 170], # Eye
        [255, 255, 130], # Glass
        [76, 153, 0], # Ear
        [0, 255, 170], # Ear Ring
        [244, 124, 244], # Nose
        [30, 162, 230], # Mouth
        [127, 255, 255], # Lip
        [127, 170, 255], # Neck
        [85, 0, 255], # Neck-lace
        [255, 170, 127], # Cloth
        [212, 127, 255], # Hair
        [0, 170, 255], # Hat
        [255, 255, 255], # nasal ala
        [0,0,100], #iris
        [0 , 200, 120], # jaw
        [10, 10, 160], # chin
        [250, 250, 200], # teeth
        ]
    
    
    
    
    if inverse == False:
        if argmax:
            im = torch.argmax(im, dim=1, keepdim=True)
            
            
            
        out = torch.zeros((im.size(0), 3, im.size(2), im.size(3)), device=im.device, dtype=torch.float32)

        for index in range(len(part_colors)):
            color = torch.from_numpy(np.array(part_colors[index])).to(out.device).to(out.dtype).view(1, 3, 1, 1).expand_as(out)
            out = torch.where(im == index, color, out)

        out = out / 255.0 * 2 - 1
        return out
    else:
        out = torch.zeros((im.size(0), 1, im.size(2), im.size(3)), device=im.device, dtype=torch.int64)
        
        for index in range(len(part_colors)):
            color = torch.from_numpy(np.array(part_colors[index])).to(im.device).to(im.dtype).view(1, 3, 1, 1).expand_as(im) / 255.0 * 2 - 1
            out = torch.where(torch.all((im - color).abs() <= 1e-2, dim=1, keepdim=True), torch.ones((im.size(0), 1, im.size(2), im.size(3)), device=out.device, dtype=torch.int64) * index, out)
        
        return out 






def mapping(G, z: torch.Tensor, conditioning_params: torch.Tensor, truncation_psi=1., truncation_cutoff=14, update_emas=True):
    return G.backbone.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)



def encode(G, ws, **synthesis_kwargs):
    planes = G.backbone.synthesis(ws, **synthesis_kwargs)
    planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
    return planes



def decode(G, ws, cam, norm_planes, denorm_planes, **synthesis_kwargs):
    cam2world_matrix = cam[:, :16].view(-1, 4, 4)
    intrinsics = cam[:, 16:25].view(-1, 3, 3)
    neural_rendering_resolution = G.neural_rendering_resolution
    
    # Create a batch of rays for volume rendering
    ray_origins, ray_directions = G.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
    N, M, _ = ray_origins.shape
    
    # Perform volume rendering
    feature_samples, seg_samples, depth_samples, weights_samples = \
        G.renderer(norm_planes, denorm_planes, G.decoder, ray_origins, ray_directions, G.rendering_kwargs)
    
    # Reshape into 'raw' neural-rendered image
    H = W = G.neural_rendering_resolution
    feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
    seg_image = seg_samples.permute(0, 2, 1).reshape(N, seg_samples.shape[-1], H, W).contiguous()
    depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
    
    # Run superresolution to get final image
    rgb_image = feature_image[:, :3]
    sr_image = G.superresolution(
        rgb_image, 
        feature_image, 
        ws, 
        noise_mode=G.rendering_kwargs['superresolution_noise_mode'], 
        **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'}
    )
    
    return {
        'image_raw': rgb_image, 
        'image': sr_image, 
        'image_depth': depth_image, 
        'image_seg': seg_image, 
    }

def compute_mean_var(planes):
    # (N, 3, C, H, W)
    mean = torch.mean(planes, dim=(-1, -2), keepdim=True)
    var = torch.sqrt(torch.var(planes, dim=(-1, -2), keepdim=True))
    return mean, var

def normalize_plane(planes):
    mean, var = compute_mean_var(planes)
    planes = (planes - mean) / (var + 1e-8)
    return planes, mean, var
def denormalize_plane(planes, mean, var):
    return planes * var + mean



if __name__ == "__main__":
    test_list =  list(args.target_image) if isinstance(args.target_image, list) else [int(args.target_image)]
    
    lpips_loss = lpips.LPIPS(net='vgg').to(device)
    read_seg = lambda fn: (transforms.ToTensor()(Image.open(fn).convert('RGB')) * 2 - 1).unsqueeze(0).to(device)
    
    G = torch.load(f"./networks/{args.network}").to(device).eval().requires_grad_(False)
    
    G.rendering_kwargs['depth_resolution'] = 96
    G.rendering_kwargs['depth_resolution_importance'] = 96
    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    
    
    for i in test_list:
    
        
    
        input1_dir = f'./data/ws/ws{i:04d}.pt'
        input2_dir = f'./data/camera_params/c{i:04d}.pt'
        
        ws = torch.load(input1_dir).to(device)
        camera_params = torch.load(input2_dir).to(device)    
        
        
    
        planes = encode(G, ws, noise_mode='const')
        norm_planes, mean, var = normalize_plane(planes)
        denorm_planes = planes
        
    
        G_decoder = torch.load(f"./networks/{args.network}").to(device).eval().requires_grad_(False)
        # G_decoder = G        
        
        out = decode(G, ws, camera_params, norm_planes, denorm_planes, noise_mode='const')
    
    
    
        original_image = out["image_raw"]
        original_seg = out["image_seg"]
        
        if args.mode == 'eyes':   
            target_seg = F.interpolate(vis_parsing_maps(read_seg(f"./data/test_image/images_test_eyes/{i:05d}/seg{i:04d}.png"), inverse=True).float(), (128, 128), mode='nearest').long()
        elif args.mode == 'nose':       
            target_seg = F.interpolate(vis_parsing_maps(read_seg(f"./data/test_image/images_test_nose/{i:05d}/seg{i:04d}.png"), inverse=True).float(), (128, 128), mode='nearest').long()
        elif args.mode == 'chin':       
            target_seg = F.interpolate(vis_parsing_maps(read_seg(f"./data/test_image/images_test_chin/{i:05d}/seg{i:04d}.png"), inverse=True).float(), (128, 128), mode='nearest').long()
    
    
        modified_mask = (original_seg.argmax(dim=1, keepdims=True) != target_seg ).float()
    
    
        with torch.no_grad():
            delta_w = torch.randn_like(ws[:, :1]).repeat(1, ws.size(1), 1) * 0.01
        delta_w = delta_w.requires_grad_(True)
        optimizer = torch.optim.Adam([delta_w], lr=0.02, betas=(0.9, 0.999))
    
        os.makedirs(f"./results/{i:05d}/", exist_ok=True)
    
        out = decode(G_decoder, ws, camera_params, norm_planes, denorm_planes, noise_mode='const')
        
        render_tensor(out['image'].clamp(-1,1)).save(f"./results/{i:05d}/img{i:05d}_edit_{args.mode}_source.png") 
        for step in tqdm(range(100)):
            optimizer.zero_grad(set_to_none=True)

            w_opt = ws + delta_w

            planes = encode(G, w_opt, noise_mode='const')
            norm_planes, _, _ = normalize_plane(planes)
            denorm_planes = denormalize_plane(norm_planes, mean, var)
            out = decode(G_decoder, w_opt, camera_params, norm_planes, denorm_planes, noise_mode='const')
    
            pred_image = out["image_raw"]
            pred_image_512 = out["image"]
            pred_seg = out["image_seg"]
    
            pixel_correct = torch.nn.CrossEntropyLoss()(pred_seg, target_seg.squeeze(1))
    
            seg_overlap = calc_dice_loss(pred_seg, target_seg.squeeze(1))*args.overlap_weight
            loss_consist = lpips_loss((1 - modified_mask) * original_image, (1 - modified_mask) * pred_image).mean() + \
                            torch.nn.MSELoss()((1 - modified_mask) * original_image, (1 - modified_mask) * pred_image)
    
            loss = pixel_correct + loss_consist + seg_overlap

            loss.backward(retain_graph=False)
            optimizer.step()
            
            # Clear GPU memory cache periodically
            if step % 10 == 0:
                torch.cuda.empty_cache()
                
        render_tensor(out['image'].clamp(-1,1)).save(f"./results/{i:05d}/img{i:05d}_edit_{args.mode}.png") 
    