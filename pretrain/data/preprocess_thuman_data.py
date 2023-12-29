import os
import cv2
import torch
import kaolin
import pandas
import imageio
import numpy as np
from tqdm import tqdm

from pytorch3d.structures import Meshes
from pytorch3d.io.obj_io import load_obj
from pytorch3d.ops import sample_points_from_meshes

import sys

sys.path.append('.')

from lib.utils.render import render_pytorch3d, Renderer

from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from SMPL import *

model_path = 'SMPL_NEUTRAL.pkl'
smpl_model = SMPL(model_path=model_path).cuda()
smpl_face = smpl_model.faces
smpl_faces = torch.from_numpy(smpl_face.astype(np.float32)).long().cuda()

from lib.model.smpl import SMPLServer

smpl_server = SMPLServer(gender='neutral').cuda()


class ScanProcessor():

    def __init__(self):

        self.scan_folder = './data/THuman2.0_Release'

        self.smpl_folder = './data/THuman2.0_smpl'

        self.scan_list = sorted(os.listdir(self.scan_folder))

        self.output_folder = './data/THuman2.0_processed_rgb'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.renderer = Renderer(256)

    def process(self, index):

        batch = {}

        scan_name = "%04d" % index

        scan_path = os.path.join(self.scan_folder, scan_name, scan_name + '.obj')

        output_folder = os.path.join(self.output_folder, scan_name)
        if not os.path.exists(output_folder): os.makedirs(output_folder)

        batch['scan_name'] = scan_name

        pickle_path = os.path.join(self.smpl_folder, '%04d_smpl.pkl' % index)
        file = pandas.read_pickle(pickle_path)
        smpl_param = np.concatenate([np.ones((1, 1)),
                                     np.zeros((1, 3)),
                                     file['global_orient'].reshape(1, -1),
                                     file['body_pose'].reshape(1, -1),
                                     file['betas'][:, :10]], axis=1)[0]

        smpl_param_x = np.concatenate([np.ones((1, 1)),
                                       np.zeros((1, 3)),
                                       np.zeros((1, 3)),
                                       file['body_pose'].reshape(1, -1),
                                       file['betas'][:, :10]], axis=1)[0]
        smpl_gdna_output = smpl_server(torch.FloatTensor(smpl_param_x).cuda().unsqueeze(0), absolute=False)
        smpl_verts = smpl_gdna_output['smpl_verts'].squeeze(0)
        batch['smpl_verts'] = smpl_verts.data.cpu().numpy()

        smpl_param_y = np.concatenate([file['global_orient'].reshape(1, -1),
                                       file['body_pose'].reshape(1, -1),
                                       file['betas'][:, :10]], axis=1)[0]
        batch['smpl_params'] = smpl_param

        ##  add my code  ##
        thetas, betas = torch.split(torch.FloatTensor(smpl_param_y).cuda().unsqueeze(0), [72, 10], dim=1)
        _, _, _, _, _, inverse_global_trans = smpl_model.forward(beta=betas, theta=thetas)
        ## ------------- ##

        scan_verts, scan_faces, aux = load_obj(scan_path, device=torch.device("cuda:0"), load_textures=False)
        scan_faces = scan_faces.verts_idx.long()

        scan_verts = scan_verts - torch.tensor(file['transl']).cuda().float().expand(scan_verts.shape[0], -1)
        scan_verts = scan_verts / file['scale'][0]

        ##  add my code  ##
        # scan_verts without global
        scan_verts_homo = torch.cat([scan_verts, torch.ones(scan_verts.shape[0], 1).cuda()], dim=1)
        scan_verts_wo_global = torch.matmul(inverse_global_trans.squeeze(0).cuda(), scan_verts_homo.T)
        scan_verts_wo_global = scan_verts_wo_global[:3, :].T
        scan_verts = scan_verts_wo_global
        ## ------------- ##

        batch['scan_verts'] = scan_verts.data.cpu().numpy()
        batch['scan_faces'] = scan_faces.data.cpu().numpy()

        ## sampling points
        num_verts, num_dim = scan_verts.shape
        random_idx = torch.randint(0, num_verts, [50000, 1], device=scan_verts.device)
        pts_surf = torch.gather(scan_verts, 0, random_idx.expand(-1, num_dim))
        pts_surf += 0.01 * torch.randn(pts_surf.shape, device=scan_verts.device)
        pts_smpl_surf = smpl_verts.unsqueeze(1).repeat(1, 20, 1).view(-1, 3)
        pts_smpl_surf += 0.01 * torch.randn(pts_smpl_surf.shape, device=smpl_verts.device)
        random_idx = torch.randperm(pts_smpl_surf.shape[0])[:50000]
        pts_smpl_surf = pts_smpl_surf[random_idx, :]
        pts_surf = torch.cat((pts_surf, pts_smpl_surf), 0)

        pts_bbox = torch.rand(pts_surf.shape, device=scan_verts.device) * 2 - 1
        pts_bbox[:, 1] = pts_bbox[:, 1] - 0.2  # fix the issue that the pts_bbox does not cover the body feet
        pts_d = torch.cat([pts_surf, pts_bbox], dim=0)
        # sdf
        residues, face_indices, _ = point_to_mesh_distance(pts_d[None], scan_verts[None], scan_faces)
        occ_gt = kaolin.ops.mesh.check_sign(scan_verts[None], scan_faces, pts_d[None]).float().unsqueeze(-1)
        sdf_gt = (occ_gt * -2 + 1) * residues.unsqueeze(-1)
        # smpl sdf
        smpl_residues, smpl_face_indices, _ = point_to_mesh_distance(pts_d[None], smpl_verts[None], smpl_faces)
        occ_smpl_gt = check_sign(smpl_verts[None], smpl_faces, pts_d[None]).float().unsqueeze(-1)
        sdf_smpl_gt = (occ_smpl_gt * -2 + 1) * smpl_residues.unsqueeze(-1)

        batch['pts_d'] = pts_d.data.cpu().numpy()
        batch['sdf_gt'] = sdf_gt[0].data.cpu().numpy()
        batch['sdf_smpl_gt'] = sdf_smpl_gt[0].data.cpu().numpy()

        np.savez(os.path.join(output_folder, 'occupancy.npz'), **batch)

        # get surface normals
        meshes = Meshes(verts=[scan_verts], faces=[scan_faces])
        verts, normals = sample_points_from_meshes(meshes, num_samples=100000, return_textures=False,
                                                   return_normals=True)

        batch_surf = {}
        batch_surf['surface_points'] = verts[0].data.cpu().numpy()
        batch_surf['surface_normals'] = normals[0].data.cpu().numpy()

        np.savez(os.path.join(output_folder, 'surface.npz'), **batch_surf)

        return


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--tot', type=int, default=1)

    args = parser.parse_args()

    processor = ScanProcessor()

    task = split(list(range(len(processor.scan_list))), args.tot)[args.id]
    batch_list = []

    for i in tqdm(task):
        batch = processor.process(i)
