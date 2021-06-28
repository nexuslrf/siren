'''
The migration of 3d occupancy learning tasks from fourier-feat paper to SIREN
'''

# Enable import from parent package
import sys
import os

import imageio
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import time
import dataio, meta_modules, utils, training, loss_functions, modules
import torch
from torch.utils.data import DataLoader
import configargparse
from functools import partial
from ray_rendering import *
from tqdm import tqdm
import mcubes
import trimesh

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--points_per_batch', type=int, default=32768)

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--mesh_path', type=str, default='data/Armadillo.ply')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--split_mlp', action='store_true')
p.add_argument('--approx_layers', type=int, default=2)
p.add_argument('--act_scale', type=float, default=1)
p.add_argument('--fusion_operator', type=str, choices=['sum', 'prod'], default='prod')
p.add_argument('--fusion_before_act', action='store_true')
p.add_argument('--split_train', action='store_true')
p.add_argument('--resolution', type=int, default=256)
p.add_argument('--recenter', type=str, choices=['fourier', 'siren'], default='fourier')
p.add_argument('--lr_decay', type=float, default=0.9995395890030878) # 0.1 ** (1/5000) = 0.9995395890030878
p.add_argument('--use_atten', action='store_true')
p.add_argument('--rbatch', type=int, default=0)
p.add_argument('--grid_batch', type=int, default=16)
p.add_argument('--rbatches', type=int, default=32)
p.add_argument('--test_mode', type=str, choices=['volrend', 'mcube'], default='volrend')
p.add_argument('--fine_pass', action='store_true')
p.add_argument('--split_accel', action='store_true')
opt = p.parse_args()

mesh_dataset = dataio.Mesh(opt.mesh_path, pts_per_batch=opt.points_per_batch, num_batches=opt.batch_size, recenter=opt.recenter, split_coord=opt.split_train)

# Define the model.
if opt.model_type == 'fourier':
    model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3, split_mlp=opt.split_mlp, freq_params=[6., 256//3], include_coord=False,
        approx_layers=opt.approx_layers, act_scale=opt.act_scale, fusion_operator=opt.fusion_operator, fusion_before_act=opt.fusion_before_act, use_atten=opt.use_atten)
elif opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3, split_mlp=opt.split_mlp,
        approx_layers=opt.approx_layers, act_scale=opt.act_scale, fusion_operator=opt.fusion_operator, fusion_before_act=opt.fusion_before_act, use_atten=opt.use_atten)
else:
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3, split_mlp=opt.split_mlp,
        approx_layers=opt.approx_layers, act_scale=opt.act_scale, fusion_operator=opt.fusion_operator, fusion_before_act=opt.fusion_before_act, use_atten=opt.use_atten)

model.load_state_dict(torch.load(opt.checkpoint_path))
model.cuda()
root_path = os.path.join(opt.logging_root, opt.experiment_name)
print("load model!")
if opt.test_mode == 'volrend':
    if opt.split_accel:
        assert opt.split_mlp
        render_fn = lambda m,d,b,r,a: vol_render_split(m,d,b,r,a,resolution=opt.resolution)
    else:
        render_fn = lambda m,d,b,r,a: vol_render_nosplit(m,d,b,r,a, resolution=opt.resolution, grid_batch=opt.grid_batch)

    R = 2.
    c2w = pose_spherical(90. + 10 + 45, -30., R)
    N_samples = 256
    N_samples_2 = 256 # We will consider second level sampling later
    H = 512
    W = H
    focal = H * .9
    rays = get_rays(H, W, focal, c2w[:3,:4])
    rbatch = H // opt.rbatches if opt.rbatch == 0 else opt.rbatch
    render_args_hr = [mesh_dataset.corners, R-1, R+1, N_samples, N_samples_2,
                        True, mesh_dataset.pts_trans_fn, opt.fine_pass]
    depth_map, acc_map = render_fn(model, mesh_dataset, rbatch, rays, render_args_hr)
    norm_map = ((make_normals(rays, depth_map) * .5 + .5) * 255).cpu().numpy().astype(np.uint8)
    img_path = os.path.join(root_path, f"norm_map_{H}.png")
    imageio.imsave(img_path, norm_map)
    print(f"save {img_path}")

elif opt.test_mode == 'mcube':
    c0, c1 = (torch.as_tensor(c) for c in mesh_dataset.corners)
    th = 0.
    N_samples = opt.resolution
    pts = torch.stack(torch.meshgrid(*([torch.linspace(0.,1.,N_samples)]*3))[::-1], -1)
    pts_sh = pts.shape
    pts_flat = pts.reshape(-1, 3)
    num_pts = pts_flat.shape[0]
    rbatch = num_pts // opt.rbatches if opt.rbatch == 0 else opt.rbatch
    rets = []
    with torch.no_grad():
        for i in tqdm(range(0, num_pts, rbatch)):
            rets.append(model({'coords': pts_flat[i:i+rbatch,:].cuda()})['model_out'])
    alpha = torch.cat(rets, 0).reshape(*pts_sh[:-1]).cpu()
    mask = torch.logical_or(torch.any(pts < c0, -1), torch.any(pts > c1, -1))
    alpha = torch.where(mask, torch.FloatTensor([0.]), alpha)
    print('fraction occupied', torch.mean((alpha > th).float()))
    vertices, triangles = mcubes.marching_cubes(alpha.numpy(), th)
    print('done', vertices.shape, triangles.shape)
    mesh = trimesh.Trimesh(vertices / N_samples - .5, triangles)
    ply_path = os.path.join(root_path, f"mesh_{N_samples}.ply")
    mesh.export(ply_path)
    print(f"export {ply_path}")
    os.system(f"du {ply_path} -sh")