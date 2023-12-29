
import pandas
import imageio
import numpy as np
from tqdm import tqdm
import glob

from pytorch3d.structures import Meshes
from pytorch3d.io.obj_io import load_obj
from pytorch3d.ops import sample_points_from_meshes

from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance
import scipy.io
from SMPL import*
import os

model_path = 'SMPL_NEUTRAL.pkl'
smpl_model = SMPL(model_path=model_path).cuda()
smpl_face = smpl_model.faces
#scipy.io.savemat('smpl_faces.mat', {'smpl_faces':smpl_faces})
smpl_faces = torch.from_numpy(smpl_face.astype(np.float32)).long().cuda()
J_template = smpl_model.J_template.data.cpu().numpy()
parents = smpl_model.parents
weights = smpl_model.weight.squeeze(0).data.cpu().numpy()
scipy.io.savemat('smpl_model_data.mat', {'J_template':J_template, 'parents':parents, 'weights':weights, 'smpl_faces':smpl_face})


class ScanProcessor():

    def __init__(self):

        self.scan_folder =  'CAPE/select_frames'
        self.smpl_folder =  'CAPE/minimal_body_shape'
        self.output_folder = 'CAPE/processed'

        self.scan_list = sorted(glob.glob(self.scan_folder+'/*.mat'))
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def process(self):

        for k in range(len(self.scan_list)):
            print(k)
            mat_file = scipy.io.loadmat(self.scan_list[k])
            transl = mat_file['transl']
            theta = mat_file['theta']
            vv_cano = mat_file['v_cano']
            vv_posed = mat_file['v_posed']
            vv_posed = vv_posed - transl
            mat_file2 = scipy.io.loadmat(self.smpl_folder+'/'+self.scan_list[k].split('/')[-1][:5]+'.mat')
            minimal_cano = mat_file2['minimal_cano']

            theta = torch.FloatTensor(theta).cuda()
            scan_verts = torch.FloatTensor(vv_posed).cuda()
            minimal_cano = torch.FloatTensor(minimal_cano).cuda().unsqueeze(0)

            theta_wo_global = theta.clone()
            theta_wo_global[:,:3] = 0

            v_posed, verts, delta_J, B, W, inverse_global_trans = smpl_model.forward(v_shaped=minimal_cano, theta=theta)
            _, verts_wo_global, _, B_wo_global, _, _ = smpl_model.forward(v_shaped=minimal_cano, theta=theta_wo_global)

            smpl_verts = verts_wo_global.squeeze(0)
            #smpl_verts_cano = minimal_cano.squeeze(0) # seems wrong, we should treat v_posed as the supervision of our id_feature learning
            smpl_verts_cano = v_posed.squeeze(0) 

            # scan_verts without global
            scan_verts_homo = torch.cat([scan_verts, torch.ones(scan_verts.shape[0], 1).cuda()], dim = 1)
            scan_verts_wo_global = torch.matmul(inverse_global_trans.squeeze(0).cuda(), scan_verts_homo.T)
            scan_verts_wo_global = scan_verts_wo_global[:3,:].T
            scan_verts = scan_verts_wo_global


            # sample poitns
            num_verts, num_dim = scan_verts.shape
            pts_surf = scan_verts.unsqueeze(1).repeat(1,20,1).view(-1,3)
            random_idx = torch.randperm(pts_surf.shape[0])[:100000]
            pts_surf = pts_surf[random_idx,:]
            pts_surf += 0.01 * torch.randn(pts_surf.shape, device=scan_verts.device)
            pts_bbox = torch.rand(pts_surf.shape, device=scan_verts.device) * 2 - 1
            pts_bbox[:,1]  = pts_bbox[:,1] - 0.2
            pts_d = torch.cat([pts_surf, pts_bbox],dim=0)


            #
            occ_gt = check_sign(scan_verts[None], smpl_faces, pts_d[None]).float().unsqueeze(-1)
            occ_smpl_gt = check_sign(smpl_verts[None], smpl_faces, pts_d[None]).float().unsqueeze(-1)
            residues, face_indices, _ = point_to_mesh_distance(pts_d[None], scan_verts[None], smpl_faces)
            sdf_value = (occ_gt*-2+1)*residues.unsqueeze(-1)
            smpl_residues, smpl_face_indices, _ = point_to_mesh_distance(pts_d[None], smpl_verts[None], smpl_faces)
            sdf_value = (occ_gt*-2+1)*residues.unsqueeze(-1)
            sdf_smpl_value = (occ_smpl_gt*-2+1)*smpl_residues.unsqueeze(-1)  


            pts = pts_d.data.cpu().numpy()
            sdf = sdf_value[0].data.cpu().numpy()
            smpl_sdf = sdf_smpl_value[0].data.cpu().numpy()
            scan_verts = scan_verts.data.cpu().numpy()
            smpl_verts = smpl_verts.data.cpu().numpy()
            smpl_verts_cano = smpl_verts_cano.data.cpu().numpy()
            delta_J = delta_J.squeeze(0).data.cpu().numpy()
            B = B_wo_global.squeeze(0).data.cpu().numpy()
            smpl_params = torch.cat((torch.zeros(1,10).cuda(), theta), dim=1)
            smpl_params = smpl_params.data.cpu().numpy()

            batch = {}
            batch['pts_d'] = pts
            batch['sdf_gt'] = sdf
            batch['sdf_smpl_gt'] = smpl_sdf
            np.savez(os.path.join(output_folder, 'occupancy.npz'), **batch)

            scan_verts = torch.FloatTensor(scan_verts).cuda()
            scan_faces = torch.from_numpy(smpl_face.astype(np.float32)).long().cuda()
            meshes = Meshes(verts=[scan_verts], faces=[scan_faces])
            verts, normals = sample_points_from_meshes(meshes, num_samples=100000, return_textures=False, return_normals=True)
            batch_surf = {}
            batch_surf['surface_points'] = verts[0].data.cpu().numpy()
            batch_surf['surface_normals'] = normals[0].data.cpu().numpy()
            np.savez(os.path.join(output_folder, 'surface.npz'), **batch_surf)


        return 


if __name__ == '__main__':


    processor = ScanProcessor()
    processor.process()
    