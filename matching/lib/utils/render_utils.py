

import torch
from torch.nn import functional as F
import numpy as np
import cv2

# # ------------------------ data ----------------------------- ##

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

# # ------------------------ general ----------------------------- ##

def split_input(model_input, vis_idx):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    total_pixels = 256*128
    n_pixels = 8192
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'][vis_idx:vis_idx+1], 1, indx)
        data['object_mask'] = torch.index_select(model_input['object_mask'][vis_idx:vis_idx+1], 1, indx)
        data['rgb'] = torch.index_select(model_input['rgb'][vis_idx:vis_idx+1], 1, indx)
        data['intrinsics'] = model_input['intrinsics'][vis_idx:vis_idx+1]
        data['pose'] = model_input['pose'][vis_idx:vis_idx + 1]
        data['theta'] = model_input['theta'][vis_idx:vis_idx + 1]
        data['shape_id_feat'] = model_input['shape_id_feat'][vis_idx:vis_idx + 1]
        data['shape_cloth_feat'] = model_input['shape_cloth_feat'][vis_idx:vis_idx + 1]
        data['tex_feat'] = model_input['tex_feat'][vis_idx:vis_idx + 1]

        split.append(data)
    return split

def merge_output(res):
    ''' Merge the split output. '''
    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        model_outputs[entry] = torch.cat([r[entry] for r in res], dim=1)

    return model_outputs


# # ------------------------ Rendering  ----------------------------- ##
def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).cuda()
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

def get_camera_params(uv, pose, intrinsics):
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:,:4])
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc


def lift(x, y, z, intrinsics):
    # parse intrinsics
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)

def get_box_intersection(cam_proj, uv):

    bz, num_pixels, _ = uv.shape
    boundary_points = np.asarray([[-0.8773,-1.2935,-0.5528,1],[-0.8773,-1.2935,0.8551,1],[-0.8773,0.9773,-0.5528,1],[-0.8773,0.9773,0.8551,1],
                             [0.9087,-1.2935,-0.5528,1],[0.9087,-1.2935,0.8551,1],[0.9087,0.9773,-0.5528,1],[0.9087,0.9773,0.8551,1]])  #8*4
    boundary_points = torch.FloatTensor(boundary_points).cuda().unsqueeze(0).repeat(bz,1,1).permute(0,2,1)
    proj_points = torch.bmm(cam_proj, boundary_points)
    z_max = torch.max(proj_points[:, 2, :], dim=-1, keepdim=True)[0]
    z_min = torch.min(proj_points[:, 2, :], dim=-1, keepdim=True)[0]
    z_max = z_max.unsqueeze(1).repeat(1,num_pixels,1)
    z_min = z_min.unsqueeze(1).repeat(1, num_pixels, 1)

    XYZ_img_max = torch.cat((uv, z_max, torch.ones((uv.shape[0], uv.shape[1], 1)).cuda()), dim=-1)
    XYZ_img_max[:,:,:2] = XYZ_img_max[:,:,:2]*XYZ_img_max[:,:,2:3]
    XYZ_img_min = torch.cat((uv, z_min, torch.ones((uv.shape[0], uv.shape[1], 1)).cuda()), dim=-1)
    XYZ_img_min[:, :, :2] = XYZ_img_min[:, :, :2] * XYZ_img_min[:, :, 2:3]

    inv_cam_proj = torch.pinverse(cam_proj)
    points_max = torch.bmm(inv_cam_proj, XYZ_img_max.permute(0, 2, 1))
    points_min = torch.bmm(inv_cam_proj, XYZ_img_min.permute(0, 2, 1))

    box_intersections = torch.cat((points_min[:,:3,:].permute(0,2,1).unsqueeze(2), points_max[:,:3,:].permute(0,2,1).unsqueeze(2)), dim=2)

    return box_intersections

def get_sphere_intersection(cam_loc, ray_directions, r = 1.35):
    # Input: n_images x 4 x 4 ; n_images x n_rays x 3
    # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays

    n_imgs, n_pix, _ = ray_directions.shape

    cam_loc = cam_loc.unsqueeze(-1)
    ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze()
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2,1) ** 2 - r ** 2)

    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0

    sphere_intersections = torch.zeros(n_imgs * n_pix, 2).cuda().float()
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.Tensor([-1, 1]).cuda().float()
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)

    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    sphere_intersections = sphere_intersections.clamp_min(0.0)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)

    return sphere_intersections, mask_intersect

def get_depth(points, pose):
    ''' Retruns depth from 3D points according to camera pose '''
    batch_size, num_samples, _ = points.shape
    if pose.shape[1] == 7:  # In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:, :4])
        pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda().float()
        pose[:, :3, 3] = cam_loc
        pose[:, :3, :3] = R

    points_hom = torch.cat((points, torch.ones((batch_size, num_samples, 1)).cuda()), dim=2)

    # permute for batch matrix product
    points_hom = points_hom.permute(0, 2, 1)

    points_cam = torch.inverse(pose).bmm(points_hom)
    depth = points_cam[:, 2, :][:, :, None]
    return depth


def get_uv(img_height=256, img_width=128):
    
    # 
    uv = np.mgrid[0:img_height, 0:img_width].astype(np.int32)
    uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
    uv = uv.reshape(2, -1).transpose(1, 0)

    return torch.Tensor(uv).float()

def change_sampling_idx(input_data, num_sample_pts):

    total_pixels = 256*128
    sampling_idx = torch.randperm(total_pixels)[:num_sample_pts]
    input_data['rgb'] = input_data['rgb'][:,sampling_idx,:]
    input_data['uv'] = input_data['uv'][:,sampling_idx,:]
    input_data["object_mask"] = input_data["object_mask"][:,sampling_idx,:]

    return input_data