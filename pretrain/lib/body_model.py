import os
import hydra
import torch
import torch.nn as nn
import wandb
import imageio
import numpy as np
import torch.nn.functional as F
from lib.model.smpl import SMPLServer
from lib.model.mesh import generate_mesh, generate_mesh_test
from lib.model.sample import PointOnBones
from lib.model.generator import Generator
from lib.model.network import ImplicitNetwork
from lib.utils.render import render_mesh_dict, weights2colors
from lib.model.deformer import skinning


class BaseModel(nn.Module):

    def __init__(self, opt, meta_info, data_processor=None):
        super().__init__()

        self.opt = opt

        self.naked_shape_network = ImplicitNetwork(**opt.naked_network)
        print(self.naked_shape_network)
        self.clothed_shape_network = ImplicitNetwork(**opt.clothed_network)
        print(self.clothed_shape_network)
        self.texture_network = ImplicitNetwork(**opt.texture_network)
        print(self.texture_network)

        self.deformer = hydra.utils.instantiate(opt.deformer, opt.deformer)
        print(self.deformer)

        self.generator = Generator(opt.dim_naked_shape)
        print(self.generator)

        self.smpl_server = SMPLServer(gender='neutral')

        self.sampler_bone = PointOnBones(self.smpl_server.bone_ids)

        self.z_naked_shapes_mean = torch.nn.Embedding(meta_info.n_samples, opt.dim_naked_shape)
        self.z_naked_shapes_var = torch.nn.Embedding(meta_info.n_samples, opt.dim_naked_shape)
        torch.nn.init.xavier_normal(self.z_naked_shapes_mean.weight)
        torch.nn.init.xavier_normal(self.z_naked_shapes_var.weight)

        self.z_clothed_shapes_mean = torch.nn.Embedding(meta_info.n_samples, opt.dim_clothed_shape)
        self.z_clothed_shapes_var = torch.nn.Embedding(meta_info.n_samples, opt.dim_clothed_shape)
        torch.nn.init.xavier_normal(self.z_clothed_shapes_mean.weight)
        torch.nn.init.xavier_normal(self.z_clothed_shapes_var.weight)

        ##
        self.fc_class_id = nn.Linear(opt.dim_naked_shape, meta_info.n_identities)
        self.fc_class_cloth = nn.Linear(opt.dim_clothed_shape, meta_info.n_clothes)
        print(self.fc_class_id)
        print(self.fc_class_cloth)

        self.data_processor = data_processor

    def reparameterize(self, mu, logvar):
        """
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, pts_d, smpl_tfs, cond, canonical=False, canonical_shape=False, eval_mode=True, mask=None):

        n_batch, n_points, n_dim = pts_d.shape

        outputs = {}

        if mask is None:
            mask = torch.ones((n_batch, n_points), device=pts_d.device, dtype=torch.bool)
        if not mask.any():
            return {'occ': -1000 * torch.ones((n_batch, n_points, 1), device=pts_d.device)}

        if canonical_shape:
            pts_c = pts_d
            pts_c_clothed = pts_d
            occ_pd, feat_pd = self.naked_shape_network(
                pts_c,
                cond={'latent': cond['latent']},
                mask=mask,
                val_pad=-1000,
                return_feat=True,
                spatial_feat=True,
                normalize=True)
            occ_clothed, clothed_feat_pd = self.clothed_shape_network(pts_c_clothed,
                                                                      cond={'latent': cond['z_clothed_shape']},
                                                                      mask=mask,
                                                                      val_pad=-1000,
                                                                      input_feat=feat_pd,
                                                                      return_feat=True,
                                                                      normalize=True)

        elif canonical:
            pts_c = self.deformer.query_cano(pts_d,
                                             {'betas': cond['z_naked_shape']},
                                             mask=mask)
            pts_c_clothed = self.deformer.query_cano(pts_d,
                                                     {'betas': cond['z_naked_shape']},
                                                     mask=mask)
            occ_pd, feat_pd = self.naked_shape_network(
                pts_c,
                cond={'latent': cond['latent']},
                mask=mask,
                val_pad=-1000,
                return_feat=True,
                spatial_feat=True,
                normalize=True)
            occ_clothed, clothed_feat_pd = self.clothed_shape_network(pts_c_clothed,
                                                                      cond={'latent': cond['z_clothed_shape']},
                                                                      mask=mask,
                                                                      val_pad=-1000,
                                                                      input_feat=feat_pd,
                                                                      return_feat=True,
                                                                      normalize=True)

        else:
            pts_c, others = self.deformer.forward(pts_d,
                                                  {'betas': cond['z_naked_shape'],
                                                   'latent': cond['lbs']},
                                                  smpl_tfs,
                                                  mask=mask,
                                                  eval_mode=eval_mode)
            pts_c_clothed, others_cloth = self.deformer.forward(pts_d,
                                                                {'betas': cond['z_naked_shape'],
                                                                 'latent': cond['lbs_cloth']},
                                                                smpl_tfs,
                                                                mask=mask,
                                                                eval_mode=eval_mode)
            occ_pd, feat_pd = self.naked_shape_network(
                pts_c.reshape((n_batch, -1, n_dim)),
                cond={'latent': cond['latent']},
                mask=others['valid_ids'].reshape((n_batch, -1)),
                val_pad=-1000,
                return_feat=True,
                spatial_feat=True,
                normalize=True)
            occ_clothed, clothed_feat_pd = self.clothed_shape_network(pts_c_clothed.reshape((n_batch, -1, n_dim)),
                                                                      cond={'latent': cond['z_clothed_shape']},
                                                                      mask=others_cloth['valid_ids'].reshape(
                                                                          (n_batch, -1)),
                                                                      val_pad=-1000,
                                                                      input_feat=feat_pd,
                                                                      return_feat=True,
                                                                      normalize=True)

            occ_pd = occ_pd.reshape(n_batch, n_points, -1, 1)
            occ_clothed = occ_clothed.reshape(n_batch, n_points, -1, 1)
            clothed_feat_pd = clothed_feat_pd.reshape(n_batch, n_points, -1, clothed_feat_pd.shape[-1])

            occ_pd, idx_c = occ_pd.max(dim=2)
            occ_clothed, idx_c_clothed = occ_clothed.max(dim=2)

            pts_c_clothed = torch.gather(pts_c_clothed, 2, idx_c_clothed.unsqueeze(-1).expand(-1, -1, 1,
                                                                                              pts_c_clothed.shape[
                                                                                                  -1])).squeeze(2)
            pts_c = torch.gather(pts_c, 2, idx_c.unsqueeze(-1).expand(-1, -1, 1, pts_c.shape[-1])).squeeze(2)


        outputs['occ'] = occ_pd
        outputs['occ_clothed'] = occ_clothed
        outputs['pts_c'] = pts_c
        outputs['pts_c_clothed'] = pts_c_clothed
        outputs['weights'] = self.deformer.query_weights(pts_c,
                                                         cond={
                                                             'betas': cond['z_naked_shape'],
                                                             'latent': cond['lbs']
                                                         })
        outputs['weights_clothed'] = self.deformer.query_weights(pts_c_clothed,
                                                                 cond={
                                                                     'betas': cond['z_naked_shape'],
                                                                     'latent': cond['lbs_cloth']
                                                                 })
        return outputs

    def prepare_cond(self, batch):

        cond = {}
        cond['thetas'] = batch['smpl_params'][:, 7:-10] / np.pi
        cond['z_naked_shape'] = batch['z_naked_shape']

        cond['latent'] = self.generator(batch['z_naked_shape'])  # [1,64,16,64,64]
        cond['lbs'] = batch['z_naked_shape']
        cond['lbs_cloth'] = batch['z_clothed_shape']
        cond['z_clothed_shape'] = batch['z_clothed_shape']

        return cond

    def training_step_single(self, current_epoch, batch):

        cond = self.prepare_cond(batch)

        loss = 0

        reg_naked_shape = torch.mean(-0.5 * torch.sum(
            1 + batch['z_naked_shape_var'] - batch['z_naked_shape_mean'] ** 2 - batch['z_naked_shape_var'].exp(),
            dim=1), dim=0)
        reg_clothed_shape = torch.mean(-0.5 * torch.sum(
            1 + batch['z_clothed_shape_var'] - batch['z_clothed_shape_mean'] ** 2 - batch['z_clothed_shape_var'].exp(),
            dim=1), dim=0)

        wandb.log({'reg_naked_shape': reg_naked_shape})
        wandb.log({'reg_clothed_shape': reg_clothed_shape})
        if current_epoch < self.opt.nepochs_pretrain_coarse:
            loss = loss + self.opt.lambda_reg * (reg_clothed_shape)
        else:
            loss = loss + self.opt.lambda_reg * (0.5 * reg_naked_shape + reg_clothed_shape)

        outputs = self.forward(batch['pts_d'], batch['smpl_tfs'], cond, eval_mode=False)
        loss_naked_occ = F.binary_cross_entropy_with_logits(outputs['occ'], batch['occ_naked_gt'])
        loss_clothed_occ = F.binary_cross_entropy_with_logits(outputs['occ_clothed'], batch['occ_clothed_gt'])
        wandb.log({'loss_naked_occ': loss_naked_occ})
        wandb.log({'loss_clothed_occ': loss_clothed_occ})
        if current_epoch < self.opt.nepochs_pretrain_coarse:
            loss = loss + loss_clothed_occ
        else:
            loss = loss + (0.5 * loss_naked_occ + 0.5 * loss_clothed_occ)

        num_batch = batch['pts_d'].shape[0]
        if current_epoch < self.opt.nepochs_pretrain:
            # Bone occupancy loss
            if self.opt.lambda_bone_occ > 0:
                pts_c, _, occ_gt, _ = self.sampler_bone.get_points(
                    self.smpl_server.joints_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1))
                if self.opt.pretrain_bone:
                    outputs = self.forward(pts_c, None, cond, canonical=True)
                    loss_bone_occ = F.binary_cross_entropy_with_logits(outputs['occ'], occ_gt.unsqueeze(-1))
                    loss_bone_occ_clothed = F.binary_cross_entropy_with_logits(outputs['occ_clothed'],
                                                                               occ_gt.unsqueeze(-1))
                else:
                    outputs = self.forward(batch['smpl_verts_cano'], None, cond, canonical=True)
                    loss_bone_occ = F.binary_cross_entropy(outputs['occ'],
                                                           torch.ones_like(outputs['occ']).type_as(outputs['occ']))
                    loss_bone_occ_clothed = F.binary_cross_entropy(outputs['occ_clothed'],
                                                                   torch.ones_like(outputs['occ_clothed']).type_as(
                                                                       outputs['occ']))

                wandb.log({'loss_bone_occ': loss_bone_occ})
                wandb.log({'loss_bone_occ_clothed': loss_bone_occ_clothed})
                if current_epoch < self.opt.nepochs_pretrain_coarse:
                    loss = loss + self.opt.lambda_bone_occ * (loss_bone_occ_clothed)
                else:
                    loss = loss + self.opt.lambda_bone_occ * (0.5 * loss_bone_occ + loss_bone_occ_clothed)

            # Joint weight loss
            if self.opt.lambda_bone_w > 0:
                pts_c, w_gt, _ = self.sampler_bone.get_joints(
                    self.smpl_server.joints_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1))
                if self.opt.pretrain_bone:
                    w_pd = self.deformer.query_weights(pts_c,
                                                       {'latent': cond['lbs'], 'betas': cond['z_naked_shape'] * 0})
                    w_pd_cloth = self.deformer.query_weights(pts_c, {'latent': cond['lbs_cloth'],
                                                                     'betas': cond['z_naked_shape'] * 0})
                    loss_bone_w = F.mse_loss(w_pd, w_gt)
                    loss_bone_w_cloth = F.mse_loss(w_pd_cloth, w_gt)
                else:
                    w_pd = self.deformer.query_weights(batch['smpl_verts_cano'],
                                                       {'latent': cond['lbs'], 'betas': cond['z_naked_shape'] * 0})
                    w_pd_cloth = self.deformer.query_weights(batch['smpl_verts_cano'], {'latent': cond['lbs_cloth'],
                                                                                        'betas': cond[
                                                                                                     'z_naked_shape'] * 0})
                    loss_bone_w = F.mse_loss(w_pd, batch['smpl_weights_cano'])
                    loss_bone_w_cloth = F.mse_loss(w_pd_cloth, batch['smpl_weights_cano'])
                wandb.log({'loss_bone_w': loss_bone_w})
                wandb.log({'loss_bone_w_cloth': loss_bone_w_cloth})

                loss = loss + self.opt.lambda_bone_w * (loss_bone_w + loss_bone_w_cloth)

        # Displacement loss
        pts_c_gt = self.smpl_server.verts_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1)
        pts_c = self.deformer.query_cano(batch['smpl_verts_cano'], {'betas': cond['z_naked_shape']})
        pts_c_cloth = self.deformer.query_cano(batch['smpl_verts_cano'], {'betas': cond['z_naked_shape']})

        loss_disp = F.mse_loss(pts_c, pts_c_gt)
        loss_disp_cloth = F.mse_loss(pts_c_cloth, pts_c_gt)
        wandb.log({'loss_disp': loss_disp})
        wandb.log({'loss_disp_cloth': loss_disp_cloth})

        loss = loss + self.opt.lambda_disp * (loss_disp + loss_disp_cloth)

        # Latent loss
        loss_latent_class = F.cross_entropy(self.fc_class_id(batch['z_naked_shape_mean']),
                                            batch['id_index']) + F.cross_entropy(
            self.fc_class_cloth(batch['z_clothed_shape_mean']), batch['cloth_index'])
        wandb.log({'loss_latent_class': loss_latent_class})

        loss = loss + loss_latent_class
        wandb.log({'loss': loss})

        return loss

    def training_step(self, current_epoch, batch):

        if self.data_processor is not None:
            batch = self.data_processor.process(batch, self.smpl_server)

        batch['z_naked_shape'] = self.reparameterize(self.z_naked_shapes_mean(batch['index']),
                                                     self.z_naked_shapes_var(batch['index']))
        batch['z_clothed_shape'] = self.reparameterize(self.z_clothed_shapes_mean(batch['index']),
                                                       self.z_clothed_shapes_var(batch['index']))
        batch['z_naked_shape_mean'], batch['z_naked_shape_var'] = self.z_naked_shapes_mean(
            batch['index']), self.z_naked_shapes_var(batch['index'])
        batch['z_clothed_shape_mean'], batch['z_clothed_shape_var'] = self.z_clothed_shapes_mean(
            batch['index']), self.z_clothed_shapes_var(batch['index'])

        wandb.log({"z_naked_shape": wandb.Histogram(
            np_histogram=np.histogram(batch['z_naked_shape_mean'].data.cpu().numpy()))})
        wandb.log({"z_clothed_shape": wandb.Histogram(
            np_histogram=np.histogram(batch['z_clothed_shape_mean'].data.cpu().numpy()))})

        loss = self.training_step_single(current_epoch, batch)
        return loss

    def validation_step(self, current_epoch, batch):

        # Data prep
        if self.data_processor is not None:
            batch = self.data_processor.process(batch, self.smpl_server)

        batch['z_naked_shape'] = self.reparameterize(self.z_naked_shapes_mean(batch['index']),
                                                     self.z_naked_shapes_var(batch['index']))
        batch['z_clothed_shape'] = self.reparameterize(self.z_clothed_shapes_mean(batch['index']),
                                                       self.z_clothed_shapes_var(batch['index']))

        with torch.no_grad():
            self.plot(current_epoch, batch)

    def test_step(self, batch):

        # Data prep
        if self.data_processor is not None:
            batch = self.data_processor.process(batch, self.smpl_server)

        batch['z_naked_shape'] = self.z_naked_shapes_mean(batch['index'])
        batch['z_clothed_shape'] = self.z_clothed_shapes_mean(batch['index'])

        with torch.no_grad():
            surf_pred_cano, surf_pred_cano_clothed, surf_pred_def, surf_pred_def_clothed = self.plot_test(batch)

        return surf_pred_cano, surf_pred_cano_clothed, surf_pred_def, surf_pred_def_clothed

    def extract_mesh(self, current_epoch, smpl_verts, smpl_tfs, cond, res_up=3):

        def occ_func(current_epoch, nepochs_pretrain_coarse, pts_c):
            outputs = self.forward(pts_c, smpl_tfs, cond, canonical=True)
            if current_epoch < nepochs_pretrain_coarse:
                return outputs['occ_clothed'].reshape(-1, 1), outputs['occ_clothed'].reshape(-1, 1)
            else:
                return outputs['occ'].reshape(-1, 1), outputs['occ_clothed'].reshape(-1, 1)

        mesh1, mesh2 = generate_mesh(occ_func, current_epoch, self.opt.nepochs_pretrain_coarse, smpl_verts.squeeze(0),
                                     res_up=res_up)
        mesh1 = {'verts': torch.tensor(mesh1.vertices).type_as(smpl_verts),
                 'faces': torch.tensor(mesh1.faces, device=smpl_verts.device)}
        mesh2 = {'verts': torch.tensor(mesh2.vertices).type_as(smpl_verts),
                 'faces': torch.tensor(mesh2.faces, device=smpl_verts.device)}

        verts1 = mesh1['verts'].unsqueeze(0)
        verts2 = mesh2['verts'].unsqueeze(0)

        outputs1 = self.forward(verts1, smpl_tfs, cond, canonical=True)
        outputs2 = self.forward(verts2, smpl_tfs, cond, canonical=True)

        mesh1['weights'] = outputs1['weights'][0].detach()  # .clamp(0,1)[0]
        mesh1['weights_color'] = torch.tensor(weights2colors(mesh1['weights'].data.cpu().numpy()),
                                              device=smpl_verts.device).float().clamp(0, 1)
        mesh1['pts_c'] = outputs1['pts_c'][0].detach()
        mesh2['weights'] = outputs2['weights'][0].detach()  # .clamp(0,1)[0]
        mesh2['weights_color'] = torch.tensor(weights2colors(mesh2['weights'].data.cpu().numpy()),
                                              device=smpl_verts.device).float().clamp(0, 1)
        mesh2['pts_c'] = outputs2['pts_c'][0].detach()

        return mesh1, mesh2

    def deform_mesh(self, mesh, smpl_tfs):
        import copy
        mesh = copy.deepcopy(mesh)

        smpl_tfs = smpl_tfs.expand(mesh['verts'].shape[0], -1, -1, -1)
        mesh['verts'] = skinning(mesh['verts'], mesh['weights'], smpl_tfs)

        if 'norm' in mesh:
            mesh['norm'] = skinning(mesh['norm'], mesh['weights'], smpl_tfs, normal=True)
            mesh['norm'] = mesh['norm'] / torch.linalg.norm(mesh['norm'], dim=-1, keepdim=True)

        return mesh

    def plot(self, current_epoch, batch):

        # Plot pred surfaces
        for key in batch:
            if type(batch[key]) is list:
                batch[key] = batch[key][0]
            else:
                batch[key] = batch[key][[0]]

        cond = self.prepare_cond(batch)

        surf_pred_cano, surf_pred_cano_clothed = self.extract_mesh(current_epoch, batch['smpl_verts_cano'],
                                                                   batch['smpl_tfs'], cond, res_up=3)
        surf_pred_def = self.deform_mesh(surf_pred_cano, batch['smpl_tfs'])
        surf_pred_def_clothed = self.deform_mesh(surf_pred_cano_clothed, batch['smpl_tfs'])

        img_list = []
        img_list.append(render_mesh_dict(surf_pred_cano, mode='npw'))
        img_list.append(render_mesh_dict(surf_pred_def, mode='npw'))
        img_list.append(render_mesh_dict(surf_pred_cano_clothed, mode='npw'))  # npwt
        img_list.append(render_mesh_dict(surf_pred_def_clothed, mode='npw'))

        img_all = np.concatenate(img_list, axis=1)
        wandb.log({"vis": [wandb.Image(img_all)]})

        save_path = 'medias'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        imageio.imsave(os.path.join(save_path, '%04d.png' % current_epoch), img_all)

    def plot_test(self, batch):

        # Plot pred surfaces
        for key in batch:
            if type(batch[key]) is list:
                batch[key] = batch[key][0]
            else:
                batch[key] = batch[key][[0]]

        cond = self.prepare_cond(batch)

        surf_pred_cano, surf_pred_cano_clothed = self.extract_mesh_test(batch['smpl_verts_cano'],
                                                                        batch['smpl_tfs'], cond, res_up=3)
        surf_pred_def = self.deform_mesh(surf_pred_cano, batch['smpl_tfs'])
        surf_pred_def_clothed = self.deform_mesh(surf_pred_cano_clothed, batch['smpl_tfs'])

        return surf_pred_cano, surf_pred_cano_clothed, surf_pred_def, surf_pred_def_clothed

    def extract_mesh_test(self, smpl_verts, smpl_tfs, cond, res_up=3):

        def occ_func(pts_c, cloth_flag):
            outputs = self.forward(pts_c, smpl_tfs, cond, canonical=True)
            # outputs = self.forward(pts_c, smpl_tfs, cond)
            if cloth_flag:
                return outputs['occ_clothed'].reshape(-1, 1), outputs['occ_clothed'].reshape(-1, 1)
            else:
                return outputs['occ'].reshape(-1, 1), outputs['occ_clothed'].reshape(-1, 1)

        mesh1, mesh2 = generate_mesh_test(occ_func, smpl_verts.squeeze(0), res_up=res_up)
        mesh1 = {'verts': torch.tensor(mesh1.vertices).type_as(smpl_verts),
                 'faces': torch.tensor(mesh1.faces, device=smpl_verts.device)}
        mesh2 = {'verts': torch.tensor(mesh2.vertices).type_as(smpl_verts),
                 'faces': torch.tensor(mesh2.faces, device=smpl_verts.device)}

        verts1 = mesh1['verts'].unsqueeze(0)
        verts2 = mesh2['verts'].unsqueeze(0)
        outputs1 = self.forward(verts1, smpl_tfs, cond, canonical=True)
        outputs2 = self.forward(verts2, smpl_tfs, cond, canonical=True)

        mesh1['weights'] = outputs1['weights'][0].detach()  # .clamp(0,1)[0]
        mesh2['weights'] = outputs2['weights'][0].detach()  # .clamp(0,1)[0]

        return mesh1, mesh2