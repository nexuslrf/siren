'''
Ray rendering related helper function.
Copied from Fourier-Feat's JAX code.
'''
import torch
import numpy as np

def get_rays(H, W, focal, c2w):
    j, i = torch.meshgrid(torch.arange(H), torch.arange(W))
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # rays_d = torch.sum(dirs[..., torch.newaxis, :] * c2w[:3,:3], -1)
    rays_d = torch.sum(torch.unsqueeze(dirs, -2) * c2w[:3,:3], -1)
    rays_o = torch.broadcast_to(c2w[:3,-1], rays_d.shape)
    return torch.stack([rays_o, rays_d], 0)

trans_t = lambda t : torch.tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=torch.float32)

rot_phi = lambda phi : torch.tensor([
    [1,0,0,0],
    [0,torch.cos(phi),-torch.sin(phi),0],
    [0,torch.sin(phi), torch.cos(phi),0],
    [0,0,0,1],
], dtype=torch.float32)

rot_theta = lambda th : torch.tensor([
    [torch.cos(th),0,-torch.sin(th),0],
    [0,1,0,0],
    [torch.sin(th),0, torch.cos(th),0],
    [0,0,0,1],
], dtype=torch.float32)

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(torch.tensor(phi/180.*np.pi)) @ c2w
    c2w = rot_theta(torch.tensor(theta/180.*np.pi)) @ c2w
    # c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

def render_rays(model, mesh, rays, corners, near, far, 
        N_samples, N_samples_2, clip, pts_trans_fn, fine_pass=True):

    rays_o, rays_d = rays[0].cuda(), rays[1].cuda()
    c0, c1 = (torch.as_tensor(c).cuda() for c in corners)
    
    th = .5
    
    # Compute 3D query points
    z_vals = torch.linspace(near, far, N_samples).cuda()
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    pts = pts_trans_fn(pts)
    # Run network
    with torch.no_grad():
        alpha = model({'coords': pts})['model_out'].sigmoid().squeeze(-1)
    if clip:
        mask = torch.logical_or(torch.any(pts < c0, -1), torch.any(pts > c1, -1))
        alpha = torch.where(mask, torch.FloatTensor([0.]).cuda(), alpha)

    alpha = (alpha > th).float()

    trans = 1.-alpha + 1e-10
    trans = torch.cat([torch.ones_like(trans[...,:1]).cuda(), trans[...,:-1]], -1)  
    weights = alpha * torch.cumprod(trans, -1)
    
    depth_map = torch.sum(weights * z_vals, -1) 
    acc_map = torch.sum(weights, -1)

    if fine_pass:
        # Second pass to refine isosurface
        z_vals = torch.linspace(-1., 1., N_samples_2).cuda() * .01 + depth_map[...,None]
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        pts = pts_trans_fn(pts)
        # Run network
        with torch.no_grad():
            alpha = model({'coords': pts.cuda()})['model_out'].sigmoid().squeeze(-1)
        if clip:
            # alpha = np.where(np.any(np.abs(pts) > 1, -1), 0., alpha)
            mask = torch.logical_or(torch.any(pts < c0, -1), torch.any(pts > c1, -1))
            alpha = torch.where(mask, torch.FloatTensor([0.]).cuda(), alpha)

        alpha = (alpha > th).float()

        trans = 1.-alpha + 1e-10
        trans = torch.cat([torch.ones_like(trans[...,:1]).cuda(), trans[...,:-1]], -1)  
        weights = alpha * torch.cumprod(trans, -1)
        
        depth_map = torch.sum(weights * z_vals, -1) 
        acc_map = torch.sum(weights, -1)

    return depth_map, acc_map

def make_normals(rays, depth_map):
    rays_o, rays_d = rays[0].cuda(), rays[1].cuda()
    pts = rays_o + rays_d * depth_map[...,None]
    dx = pts - torch.roll(pts, -1, 0)
    dy = pts - torch.roll(pts, -1, 1)
    normal_map = torch.cross(dx, dy)
    normal_map = normal_map / torch.norm(normal_map, dim=-1, keepdim=True).clamp(1e-5)
    return normal_map

def render_mesh_normals(mesh, rays):
    origins, dirs = rays.reshape([2,-1,3])
    origins = origins * .5 + .5
    dirs = dirs * .5
    z = mesh.ray.intersects_first(origins, dirs)
    pic = np.zeros([origins.shape[0],3]) 
    pic[z!=-1] = mesh.face_normals[z[z!=-1]]
    pic = np.reshape(pic, rays.shape[1:])
    return pic