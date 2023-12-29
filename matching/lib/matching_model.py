import os
import hydra
import torch
import torch.nn as nn
import wandb
import imageio
import numpy as np
import torch.nn.functional as F
from lib.model.smpl import SMPLServer
from lib.model.mesh import generate_mesh_from_img
from lib.model.generator import Generator
from lib.model.network import ImplicitNetwork, TextureNetwork, init_weights
from lib.utils.render import render_mesh_dict, weights2colors
from lib.model.deformer import skinning
from lib.model.img_encoder import ImageEncoder
from lib.model.loss import Classifier, TripletLoss, CrossEntropyWithLabelSmooth, RGBLoss
from lib.utils.render_utils import get_uv, get_box_intersection
from lib.utils.eval_metrics import evaluate

class MatchingModel(nn.Module):

    def __init__(self, opt, opt_render, data_processor=None):
        super().__init__()

        self.opt = opt
        self.num_ray_steps = opt_render.num_ray_steps
        self.num_sample_pixels = opt_render.num_sample_pixels
        self.occ_thresh = 0

        self.naked_shape_network = ImplicitNetwork(**opt.naked_network)
        print(self.naked_shape_network)
        self.clothed_shape_network = ImplicitNetwork(**opt.clothed_network)
        print(self.clothed_shape_network)
        self.texture_network = TextureNetwork()
        print(self.texture_network)
        #self.texture_network.apply(init_weights)

        self.deformer = hydra.utils.instantiate(opt.deformer, opt.deformer)
        print(self.deformer)

        self.generator = Generator(opt.dim_naked_shape)
        print(self.generator)

        self.smpl_server = SMPLServer(gender='neutral')

        # image encoder
        self.encoder = ImageEncoder(feat_dim=4096, dim_naked_shape=opt.dim_naked_shape, dim_clohted_shape=opt.dim_clothed_shape, dim_texture=opt.dim_texture)
        print(self.encoder)

        # classifier
        self.classifier = Classifier(feature_dim=4096, num_classes=opt.num_classes)
        self.criterion_pair = TripletLoss(margin=0.3)
        self.criterion_cla = CrossEntropyWithLabelSmooth()

        self.rgb_loss = RGBLoss()

        # rendering
        self.uv = get_uv(img_height=opt_render.rendering_img_height, img_width=opt_render.rendering_img_width)

        self.data_processor = data_processor

    def forward_weight(self, pts_d, cond, mask=None):

        n_batch, n_points, n_dim = pts_d.shape

        if mask is None:
            mask = torch.ones( (n_batch, n_points), device=pts_d.device, dtype=torch.bool)
        if not mask.any():
            return {'occ': -1000*torch.ones( (n_batch, n_points, 1), device=pts_d.device)}

        pts_c_clothed = self.deformer.query_cano(pts_d,
                                         {'betas': cond['z_naked_shape']},
                                        mask=mask)

        weights = self.deformer.query_weights(pts_c_clothed,cond={'betas': cond['z_naked_shape'],'latent': cond['lbs_cloth']})
        return weights

    def forward_shape(self, pts_d, cond, smpl_tfs, smpl_verts, canonical=False, mask=None, only_near_smpl=False):

        n_batch, n_points, n_dim = pts_d.shape

        if mask is None:
            mask = torch.ones( (n_batch, n_points), device=pts_d.device, dtype=torch.bool)

        # Filter based on SMPL
        if only_near_smpl:
            from kaolin.metrics.pointcloud import sided_distance
            distance, _ = sided_distance(pts_d, smpl_verts[:, ::10])
            mask = mask & (distance < 0.1 * 0.1)

        if not mask.any():
            return -1000 * torch.ones((n_batch, n_points, 1), device=pts_d.device), torch.ones((n_batch, n_points, 3), device=pts_d.device), -1000 * torch.ones((n_batch, n_points, 256), device=pts_d.device)

        if canonical:
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
                                                                      return_feat=True)
        else:
            pts_c, others = self.deformer.forward(pts_d,
                                        {'betas': cond['z_naked_shape'],
                                        'latent': cond['lbs']},
                                        smpl_tfs,
                                        mask=mask,
                                        eval_mode=True)
            pts_c_clothed, others_cloth = self.deformer.forward(pts_d,
                                        {'betas': cond['z_naked_shape'],
                                        'latent': cond['lbs_cloth']},
                                        smpl_tfs,
                                        mask=mask,
                                        eval_mode=True)
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
                              mask=others_cloth['valid_ids'].reshape((n_batch, -1)),
                              val_pad=-1000,
                              input_feat=feat_pd,
                              return_feat=True)

            occ_clothed = occ_clothed.reshape(n_batch, n_points, -1, 1)
            clothed_feat_pd = clothed_feat_pd.reshape(n_batch, n_points, -1, clothed_feat_pd.shape[-1])
            occ_clothed, idx_c_clothed = occ_clothed.max(dim=2)
            pts_c_clothed = torch.gather(pts_c_clothed, 2, idx_c_clothed.unsqueeze(-1).expand(-1, -1, 1, pts_c_clothed.shape[-1])).squeeze(2)
            clothed_feat_pd = torch.gather(clothed_feat_pd, 2, idx_c_clothed.unsqueeze(-1).expand(-1, -1, 1, clothed_feat_pd.shape[-1])).squeeze(2)
            #mask = torch.gather(others_cloth['valid_ids'], 2, idx_c_clothed).squeeze(2)

        return occ_clothed, pts_c_clothed, clothed_feat_pd


    def forward_texture(self, pts_c_clothed, clothed_feat_pd, cond, mask=None):

        n_batch, n_points, n_dim = pts_c_clothed.shape

        if mask is None:
            mask = torch.ones( (n_batch, n_points), device=pts_c_clothed.device, dtype=torch.bool)

        texture = self.texture_network(pts_c_clothed, torch.cat((cond['z_clothed_shape'],cond['z_texture']),dim=1), mask=mask)

        return texture


    def sampling_pixels(self, rgb, mask, uv):

        batch, num_pixels, _ = rgb.shape
        rand_idx = np.random.permutation(num_pixels)[:self.num_sample_pixels]

        rgb = rgb[:,rand_idx,:]
        mask = mask[:, rand_idx]
        uv = uv[:, rand_idx, :]

        return rgb, mask, uv


    @torch.no_grad()
    def img_rendering_precompute(self, batch, cond):

        batch_size, num_sample_pixels, _ = batch['rgb'].shape
        rgb, object_mask, uv = batch['rgb'].clone(), batch['mask'].clone().squeeze(-1), self.uv.clone().unsqueeze(0).repeat(batch_size,1,1).cuda()

        # box intersection
        box_intersections = get_box_intersection(batch["cam_proj"], uv)
        ray_dirs = F.normalize(box_intersections[:,:,1,:]-box_intersections[:,:,0,:], dim=2)
        intersect_dist = torch.sqrt(torch.sum((box_intersections[:, :, 0, :] - box_intersections[:, :, 1, :]) ** 2, -1))
        intervals_dist = torch.linspace(0, 1, steps=self.num_ray_steps).cuda().view(1, 1, -1).repeat(batch_size, num_sample_pixels, 1)
        intersect_dist = intersect_dist.unsqueeze(-1) * intervals_dist
        points = box_intersections[:, :, [0], :] + intersect_dist.unsqueeze(-1) * ray_dirs.unsqueeze(2)

        # computer occupancy value or SDF
        points_all = points.view(batch_size, num_sample_pixels * self.num_ray_steps, 3)
        occ_all = []
        occ_feat_all = []
        pts_c_clothed_all = []
        for pnts in torch.split(points_all, 128*256*10, dim=1):
            with torch.no_grad():
                occ, pts_c_clothed, occ_feat  = self.forward_shape(pnts, cond, batch['smpl_tfs'], batch['smpl_verts'], only_near_smpl=True)
                occ_all.append(occ)
                occ_feat_all.append(occ_feat)
                pts_c_clothed_all.append(pts_c_clothed)
        occ_all = torch.cat(occ_all, 1).reshape(batch_size, num_sample_pixels , self.num_ray_steps)
        occ_feat_all = torch.cat(occ_feat_all, 1).reshape(batch_size, num_sample_pixels, self.num_ray_steps, torch.cat(occ_feat_all, 1).shape[-1])
        pts_c_clothed_all = torch.cat(pts_c_clothed_all, 1).reshape(batch_size, num_sample_pixels, self.num_ray_steps, 3)

        tmp = torch.sign(occ_all) * torch.arange(self.num_ray_steps, 0, -1).cuda().float().reshape((1, self.num_ray_steps))  # Force argmax to return the first >0 value
        sampler_pts_ind = torch.argmax(tmp, -1)
        sampler_pts_c_clothed = torch.gather(pts_c_clothed_all, 2, sampler_pts_ind.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)).squeeze(2)
        sampler_occ = torch.gather(occ_all, 2, sampler_pts_ind.unsqueeze(-1)).squeeze(2)
        sampler_occ_feat =  torch.gather(occ_feat_all, 2, sampler_pts_ind.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,occ_feat_all.shape[-1])).squeeze(2)

        sampler_mask = (sampler_occ > self.occ_thresh)

        return sampler_pts_c_clothed, sampler_occ_feat, sampler_mask

    @torch.no_grad()
    def img_rendering_test(self, batch, cond):

        batch_size, num_sample_pixels, _ = batch['rgb'].shape
        rgb, object_mask, uv = batch['rgb'].clone(), batch['mask'].clone().squeeze(-1), self.uv.clone().unsqueeze(0).repeat(batch_size,1,1).cuda()

        # box intersection
        box_intersections = get_box_intersection(batch["cam_proj"], uv)
        ray_dirs = F.normalize(box_intersections[:,:,1,:]-box_intersections[:,:,0,:], dim=2)
        intersect_dist = torch.sqrt(torch.sum((box_intersections[:, :, 0, :] - box_intersections[:, :, 1, :]) ** 2, -1))
        intervals_dist = torch.linspace(0, 1, steps=self.num_ray_steps).cuda().view(1, 1, -1).repeat(batch_size, num_sample_pixels, 1)
        intersect_dist = intersect_dist.unsqueeze(-1) * intervals_dist
        points = box_intersections[:, :, [0], :] + intersect_dist.unsqueeze(-1) * ray_dirs.unsqueeze(2)

        # computer occupancy value or SDF
        points_all = points.view(batch_size, num_sample_pixels * self.num_ray_steps, 3)
        occ_all = []
        occ_feat_all = []
        pts_c_clothed_all = []
        for pnts in torch.split(points_all, 256*128, dim=1):
            with torch.no_grad():
                occ, pts_c_clothed, occ_feat  = self.forward_shape(pnts, cond, batch['smpl_tfs'], batch['smpl_verts'], only_near_smpl=True)
                occ_all.append(occ)
                occ_feat_all.append(occ_feat)
                pts_c_clothed_all.append(pts_c_clothed)
        occ_all = torch.cat(occ_all, 1).reshape(batch_size, num_sample_pixels , self.num_ray_steps)
        occ_feat_all = torch.cat(occ_feat_all, 1).reshape(batch_size, num_sample_pixels, self.num_ray_steps, torch.cat(occ_feat_all, 1).shape[-1])
        pts_c_clothed_all = torch.cat(pts_c_clothed_all, 1).reshape(batch_size, num_sample_pixels, self.num_ray_steps, 3)

        tmp = torch.sign(occ_all) * torch.arange(self.num_ray_steps, 0, -1).cuda().float().reshape((1, self.num_ray_steps))  # Force argmax to return the first >0 value
        sampler_pts_ind = torch.argmax(tmp, -1)
        sampler_pts_c_clothed = torch.gather(pts_c_clothed_all, 2, sampler_pts_ind.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)).squeeze(2)
        sampler_occ = torch.gather(occ_all, 2, sampler_pts_ind.unsqueeze(-1)).squeeze(2)
        sampler_occ_feat =  torch.gather(occ_feat_all, 2, sampler_pts_ind.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,occ_feat_all.shape[-1])).squeeze(2)

        sampler_mask = (sampler_occ > self.occ_thresh)

        # infer texture
        esti_rgb = self.forward_texture(sampler_pts_c_clothed, sampler_occ_feat, cond, mask=sampler_mask)

        return rgb, esti_rgb, object_mask, sampler_mask

    def img_rendering(self, batch, cond, data_flag):

        rgb = batch['rgb'].clone()[data_flag==1]
        mask = batch['mask'].clone().squeeze(-1)[data_flag==1]
        batch_size, _, _ = rgb.shape
        t_uv = self.uv.clone().unsqueeze(0).repeat(batch_size,1,1).cuda()

        rgb, object_mask, uv = self.sampling_pixels(rgb, mask, t_uv)

        # box intersection
        box_intersections = get_box_intersection(batch["cam_proj"][data_flag==1], uv)
        ray_dirs = F.normalize(box_intersections[:,:,1,:]-box_intersections[:,:,0,:], dim=2)
        intersect_dist = torch.sqrt(torch.sum((box_intersections[:, :, 0, :] - box_intersections[:, :, 1, :]) ** 2, -1))
        intervals_dist = torch.linspace(0, 1, steps=self.num_ray_steps).cuda().view(1, 1, -1).repeat(batch_size, self.num_sample_pixels, 1)
        intersect_dist = intersect_dist.unsqueeze(-1) * intervals_dist
        points = box_intersections[:, :, [0], :] + intersect_dist.unsqueeze(-1) * ray_dirs.unsqueeze(2)

        # computer occupancy value or SDF
        points_all = points.view(batch_size, self.num_sample_pixels * self.num_ray_steps, 3)
        occ_all = []
        occ_feat_all = []
        pts_c_clothed_all = []
        for pnts in torch.split(points_all, 25600, dim=1):
            with torch.no_grad():
                occ, pts_c_clothed, occ_feat  = self.forward_shape(pnts, cond, batch['smpl_tfs'][data_flag==1], batch['smpl_verts'][data_flag==1], only_near_smpl=True)
                occ_all.append(occ)
                occ_feat_all.append(occ_feat)
                pts_c_clothed_all.append(pts_c_clothed)
        occ_all = torch.cat(occ_all, 1).reshape(batch_size, self.num_sample_pixels , self.num_ray_steps)
        occ_feat_all = torch.cat(occ_feat_all, 1).reshape(batch_size, self.num_sample_pixels, self.num_ray_steps, torch.cat(occ_feat_all, 1).shape[-1])
        pts_c_clothed_all = torch.cat(pts_c_clothed_all, 1).reshape(batch_size, self.num_sample_pixels, self.num_ray_steps, 3)

        tmp = torch.sign(occ_all) * torch.arange(self.num_ray_steps, 0, -1).cuda().float().reshape((1, self.num_ray_steps))  # Force argmax to return the first >0 value
        sampler_pts_ind = torch.argmax(tmp, -1)
        sampler_pts_c_clothed = torch.gather(pts_c_clothed_all, 2, sampler_pts_ind.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)).squeeze(2)
        sampler_occ = torch.gather(occ_all, 2, sampler_pts_ind.unsqueeze(-1)).squeeze(2)
        sampler_occ_feat =  torch.gather(occ_feat_all, 2, sampler_pts_ind.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,occ_feat_all.shape[-1])).squeeze(2)

        sampler_mask = (sampler_occ > self.occ_thresh)

        # infer texture
        esti_rgb = self.forward_texture(sampler_pts_c_clothed, sampler_occ_feat, cond, mask=sampler_mask)
        loss_rgb, loss_mask = self.rgb_loss(rgb, esti_rgb, object_mask, sampler_mask, sampler_occ)

        return loss_rgb, loss_mask


    def img_rendering_syn(self, batch, cond, data_flag):

        rgb = batch['rgb'].clone()[data_flag==0]
        mask = batch['mask'].clone().squeeze(-1)[data_flag==0]
        batch_size, _, _ = rgb.shape
        t_uv = self.uv.clone().unsqueeze(0).repeat(batch_size,1,1).cuda()

        pts_c_clothed = batch['pts_c'][data_flag==0]
        pts_mask = batch['pts_mask'][data_flag==0]
        occ_feat = batch['occ_feat'][data_flag==0]
        pts_mask = pts_mask==1

        #print(pts_c_clothed, occ_feat, cond['z_texture'])
        esti_rgb = self.forward_texture(pts_c_clothed, occ_feat, cond, mask=pts_mask)
        loss_rgb = self.rgb_loss(rgb, esti_rgb, mask, pts_mask, None, True)

        return loss_rgb

    def training_step_single(self, current_epoch, batch):

        id_feat, esti_z_naked_shape, esti_z_clothed_shape, esti_z_texture = self.encoder(False, batch['input_img'])

        wandb.log({"z_naked_shape": wandb.Histogram(np_histogram=np.histogram(esti_z_naked_shape.data.cpu().numpy()))})
        wandb.log({"z_clothed_shape": wandb.Histogram(np_histogram=np.histogram(esti_z_clothed_shape.data.cpu().numpy()))})
        wandb.log({"z_texture": wandb.Histogram(np_histogram=np.histogram(esti_z_texture.data.cpu().numpy()))})

        # syn latent loss
        loss_latent_naked = F.mse_loss(esti_z_naked_shape[batch['latent_flag']==1], batch['z_naked_shape'][batch['latent_flag']==1]) 
        loss_latent_clothed = F.mse_loss(esti_z_clothed_shape[batch['latent_flag']==1], batch['z_clothed_shape'][batch['latent_flag']==1]) 
        loss_latent = loss_latent_naked + loss_latent_clothed
        #print(batch['cam_id'][batch['latent_flag']==1])
        wandb.log({'loss_latent_naked': loss_latent_naked})
        wandb.log({'loss_latent_clothed': loss_latent_clothed})
        wandb.log({'loss_latent': loss_latent})

        # classifier
        logits = self.classifier(id_feat)
        loss_cla = self.criterion_cla(logits[batch['label_flag']==1], batch['label'][batch['label_flag']==1])
        loss_pair = self.criterion_pair(id_feat[batch['label_flag']==1], batch['label'][batch['label_flag']==1])
        loss_classifer = loss_cla + loss_pair
        #print(batch['cam_id'][batch['label_flag'] == 1])
        wandb.log({'loss_cla': loss_cla})
        wandb.log({'loss_pair': loss_pair})
        wandb.log({'loss_classifer': loss_classifer})

        # reconstruction loss (rgb_loss, mask_loss)
        loss_rgb_real, loss_mask_real = self.img_rendering(batch, self.prepare_cond(esti_z_naked_shape[batch['label_flag'] == 1], esti_z_clothed_shape[batch['label_flag'] == 1], esti_z_texture[batch['label_flag'] == 1]), batch['label_flag'])
        loss_rgb_syn = self.img_rendering_syn(batch, self.prepare_cond(esti_z_naked_shape[batch['label_flag'] == 1], esti_z_clothed_shape[batch['label_flag'] == 0], esti_z_texture[batch['label_flag'] == 0]), batch['label_flag'])
        loss_rgb = loss_rgb_real + loss_rgb_syn

        wandb.log({'loss_rgb': loss_rgb})
        wandb.log({'loss_rgb_real': loss_rgb_real})
        wandb.log({'loss_rgb_syn': loss_rgb_syn})
        wandb.log({'loss_mask': loss_mask_real})

        loss = loss_latent + loss_classifer + loss_rgb + loss_mask_real
        wandb.log({'loss': loss})

        return loss

    def training_step_single_debug(self, current_epoch, batch):

        print(batch['sub_name'], batch['image_list'])
        id_feat, esti_z_naked_shape, esti_z_clothed_shape, esti_z_texture = self.encoder(False, batch['input_img'])

        wandb.log({"z_naked_shape": wandb.Histogram(np_histogram=np.histogram(esti_z_naked_shape.data.cpu().numpy()))})
        wandb.log({"z_clothed_shape": wandb.Histogram(np_histogram=np.histogram(esti_z_clothed_shape.data.cpu().numpy()))})
        wandb.log({"z_texture": wandb.Histogram(np_histogram=np.histogram(esti_z_texture.data.cpu().numpy()))})

        # syn latent loss
        loss_latent = F.mse_loss(esti_z_naked_shape[batch['latent_flag']], batch['z_naked_shape'][batch['latent_flag']]) \
                      + F.mse_loss(esti_z_clothed_shape[batch['latent_flag']], batch['z_clothed_shape'][batch['latent_flag']])
        wandb.log({'loss_latent': loss_latent})

        # classifier
        logits = self.classifier(id_feat)
        loss_cla = self.criterion_cla(logits, batch['label'])
        loss_pair = 0 #self.criterion_pair(id_feat, batch['label'])
        loss_classifer = loss_cla + loss_pair
        wandb.log({'loss_cla': loss_cla})
        wandb.log({'loss_pair': loss_pair})
        wandb.log({'loss_classifer': loss_classifer})

        # reconstruction loss (rgb_loss, mask_loss)
        loss_rgb, loss_mask = self.img_rendering(batch, self.prepare_cond(batch['z_naked_shape'], batch['z_clothed_shape'], esti_z_texture))
        wandb.log({'loss_rgb': loss_rgb})
        wandb.log({'loss_mask': loss_mask})

        loss = loss_latent + loss_classifer + loss_rgb + loss_mask
        wandb.log({'loss': loss})

        return loss

    def prepare_cond(self, esti_z_naked_shape, esti_z_clothed_shape, esti_z_texture):

        cond = {}
        cond['z_naked_shape'] = esti_z_naked_shape
        cond['latent'] = self.generator(esti_z_naked_shape) #[1,64,16,64,64]
        cond['lbs'] = esti_z_naked_shape
        cond['lbs_cloth'] = esti_z_clothed_shape
        cond['z_clothed_shape'] = esti_z_clothed_shape
        cond['z_texture'] = esti_z_texture

        return cond


    def training_texture_step(self, current_epoch, batch, z_texture):

        batch_size, _, _ = batch['rgb'].shape
        rgb, object_mask, uv = batch['rgb'].clone(), batch['mask'].clone().squeeze(-1), self.uv.clone().unsqueeze(0).repeat(batch_size,1,1).cuda()

        pts_c_clothed = batch['pts_c']
        pts_mask = batch['pts_mask']
        occ_feat = batch['occ_feat']
        pts_mask = pts_mask==1

        esti_rgb = self.forward_texture(pts_c_clothed, occ_feat, cond={'z_texture': z_texture}, mask=pts_mask)
        loss_rgb = self.rgb_loss(rgb, esti_rgb, object_mask, pts_mask, None, True)
        wandb.log({'loss_rgb': loss_rgb})
        wandb.log({"z_texture": wandb.Histogram(np_histogram=np.histogram(z_texture.data.cpu().numpy()))})

        return loss_rgb

    def training_step(self, current_epoch, batch):

        loss = self.training_step_single(current_epoch, batch)
        #loss = self.training_step_single_debug(current_epoch, batch) # GT-latents
        return loss


    @torch.no_grad()
    def test_texture_step(self, current_epoch, batch, z_texture):

        batch_size, _, _ = batch['rgb'].shape
        rgb, object_mask, uv = batch['rgb'].clone(), batch['mask'].clone().squeeze(-1), self.uv.clone().unsqueeze(0).repeat(batch_size,1,1).cuda()

        pts_c_clothed = batch['pts_c']
        pts_mask = batch['pts_mask']
        occ_feat = batch['occ_feat']
        pts_mask = pts_mask==1

        esti_rgb = self.forward_texture(pts_c_clothed, occ_feat, cond={'z_texture': z_texture}, mask=pts_mask)

        return rgb, esti_rgb, object_mask, pts_mask

    @torch.no_grad()
    def matching_val(self, gallery_dataloader, query_dataloader):

        # gallery
        gallery_features = []
        gallery_label = torch.tensor([])
        gallery_camids = torch.tensor([])
        print('----------------------- Extract gallery features -------------------------')
        for data_index, gallery_batch in enumerate(gallery_dataloader):
            id_feat = self.encoder(True, gallery_batch['input_img'].cuda())
            id_feat_flip = self.encoder(True, torch.flip(gallery_batch['input_img'].cuda(), [3]))
            gallery_features.append(F.normalize(id_feat + id_feat_flip, p=2, dim=1))
            gallery_label = torch.cat((gallery_label,gallery_batch['label']), dim=0)
            gallery_camids = torch.cat((gallery_camids, gallery_batch['cam_id']), dim=0)
        gallery_features = torch.cat(gallery_features, 0)

        # query
        query_features = []
        query_label = torch.tensor([])
        query_camids = torch.tensor([])
        print('----------------------- Extract query features -------------------------')
        for data_index, query_batch in enumerate(query_dataloader):
            id_feat = self.encoder(True, query_batch['input_img'].cuda())
            id_feat_flip = self.encoder(True, torch.flip(query_batch['input_img'].cuda(), [3]))
            query_features.append(F.normalize(id_feat + id_feat_flip, p=2, dim=1))
            query_label = torch.cat((query_label,query_batch['label']), dim=0)
            query_camids = torch.cat((query_camids, query_batch['cam_id']), dim=0)
        query_features = torch.cat(query_features, 0)
        torch.cuda.empty_cache()
        gallery_label = gallery_label.numpy()
        query_label = query_label.numpy()
        gallery_camids = gallery_camids.numpy()
        query_camids = query_camids.numpy()

        # distance
        m, n = query_features.size(0), gallery_features.size(0)
        distmat = torch.zeros((m, n))
        # Cosine similarity
        for i in range(m):
            distmat[i] = (- torch.mm(query_features[i:i+1], gallery_features.t())).cpu()
        distmat = distmat.numpy()

        #
        cmc, mAP = evaluate(distmat, query_label, gallery_label, query_camids, gallery_camids)
        print('----------------------- Results -------------------------')
        print('Top1:', cmc[0], 'Top5:', cmc[4], 'Top10:', cmc[9], 'Top20:', cmc[19], 'mAP:', mAP, '\n')
        wandb.log({'Top1': cmc[0]})
        wandb.log({'Top5': cmc[4]})
        wandb.log({'Top10': cmc[9]})
        wandb.log({'Top20': cmc[19]})
        wandb.log({'mAP': mAP})

        return 1

    @torch.no_grad()
    def visualization_texture(self, current_epoch, batch, z_texture):

        for key in batch:
            if type(batch[key]) is list:
                batch[key] = batch[key][0]
            else:
                batch[key] = batch[key][[0]]

        rgb, esti_rgb, mask, esti_mask = self.test_texture_step(current_epoch, batch, z_texture)

        rgb = np.uint8((rgb.view(256,128,3).data.cpu().numpy()*0.5+0.5)*255)
        rgb = np.concatenate((np.ones((256,64,3)).astype(np.uint8)*255, rgb, np.ones((256,64,3)).astype(np.uint8)*255), axis=1)
        esti_rgb = np.uint8((esti_rgb.view(256,128,3).data.cpu().numpy()*0.5+0.5)*255)
        esti_rgb = np.concatenate((np.ones((256,64,3)).astype(np.uint8)*255, esti_rgb, np.ones((256,64,3)).astype(np.uint8)*255), axis=1)
        mask = np.uint8(mask.view(256, 128, 1).data.cpu().numpy() * 255)
        mask = np.concatenate((np.zeros((256,64,3)).astype(np.uint8), np.concatenate((mask,mask,mask),axis=-1), np.zeros((256,64,3)).astype(np.uint8)), axis=1)
        esti_mask = np.uint8(esti_mask.view(256, 128, 1).data.cpu().numpy() * 255)
        esti_mask = np.concatenate((np.zeros((256,64,3)).astype(np.uint8), np.concatenate((esti_mask,esti_mask,esti_mask),axis=-1), np.zeros((256,64,3)).astype(np.uint8)), axis=1)
        image_show =  np.concatenate((np.concatenate((rgb, esti_rgb), axis=1), np.concatenate((mask, esti_mask), axis=1)), axis=0)

        wandb.log({"Texture": [wandb.Image(image_show)]})

        return 1

    @torch.no_grad()
    def visualization(self, current_epoch, batch):

        self.plot(current_epoch, batch)

        return 1

    def deform_mesh(self, mesh, smpl_tfs):
        import copy
        mesh = copy.deepcopy(mesh)

        smpl_tfs = smpl_tfs.expand(mesh['verts'].shape[0],-1,-1,-1)
        mesh['verts'] = skinning(mesh['verts'], mesh['weights'], smpl_tfs)
        
        if 'norm' in mesh:
            mesh['norm']  = skinning( mesh['norm'], mesh['weights'], smpl_tfs, normal=True)
            mesh['norm'] = mesh['norm']/ torch.linalg.norm(mesh['norm'],dim=-1,keepdim=True)
            
        return mesh

    def plot(self, current_epoch, batch):

        with torch.no_grad():
            id_feat, esti_z_naked_shape, esti_z_clothed_shape, esti_z_texture = self.encoder(False, batch['input_img'])

        # Plot pred surfaces
        for key in batch:
            if type(batch[key]) is list:
                batch[key] = batch[key][0]
            else:
                batch[key] = batch[key][[0]]
        cond = self.prepare_cond(esti_z_naked_shape[:1], esti_z_clothed_shape[:1], esti_z_texture[:1])
        surf_pred_cano = self.extract_mesh(batch['smpl_tfs'], batch['smpl_verts'], cond, res_up=3)
        surf_pred_def = self.deform_mesh(surf_pred_cano, batch['smpl_tfs'])

        # Plot pred RGB and mask
        rgb, esti_rgb, mask, esti_mask = self.img_rendering_test(batch, cond)
        rgb = np.uint8((rgb.view(256,128,3).data.cpu().numpy()*0.5+0.5)*255)
        rgb = np.concatenate((np.ones((256,64,3)).astype(np.uint8)*255, rgb, np.ones((256,64,3)).astype(np.uint8)*255), axis=1)
        esti_rgb = np.uint8((esti_rgb.view(256,128,3).data.cpu().numpy()*0.5+0.5)*255)
        esti_rgb = np.concatenate((np.ones((256,64,3)).astype(np.uint8)*255, esti_rgb, np.ones((256,64,3)).astype(np.uint8)*255), axis=1)
        mask = np.uint8(mask.view(256, 128, 1).data.cpu().numpy() * 255)
        mask = np.concatenate((np.zeros((256,64,3)).astype(np.uint8), np.concatenate((mask,mask,mask),axis=-1), np.zeros((256,64,3)).astype(np.uint8)), axis=1)
        esti_mask = np.uint8(esti_mask.view(256, 128, 1).data.cpu().numpy() * 255)
        esti_mask = np.concatenate((np.zeros((256,64,3)).astype(np.uint8), np.concatenate((esti_mask,esti_mask,esti_mask),axis=-1), np.zeros((256,64,3)).astype(np.uint8)), axis=1)
        image_show =  np.concatenate((np.concatenate((rgb, esti_rgb), axis=1), np.concatenate((mask, esti_mask), axis=1)), axis=0)

        #
        img_list = []
        img_list.append(render_mesh_dict(surf_pred_cano,mode='npw')[:,:,:3])
        img_list.append(render_mesh_dict(surf_pred_def, mode='npw')[:,:,:3])
        img_shape_show = np.concatenate(img_list, axis=1)
        img_all = np.concatenate((image_show, img_shape_show), axis=0)
        wandb.log({"vis": [wandb.Image(img_all)]})

        save_path = 'medias'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        imageio.imsave(os.path.join(save_path, '%04d.png' % current_epoch), img_all)

        return 1


    def plot_test(self, batch):

        with torch.no_grad():
            id_feat, esti_z_naked_shape, esti_z_clothed_shape, esti_z_texture = self.encoder(False, batch['input_img'])

        # Plot pred surfaces
        for key in batch:
            if type(batch[key]) is list:
                batch[key] = batch[key][0]
            else:
                batch[key] = batch[key][[0]]
        cond = self.prepare_cond(esti_z_naked_shape[:1], esti_z_clothed_shape[:1], esti_z_texture[:1])
        surf_pred_cano = self.extract_mesh(batch['smpl_tfs'], batch['smpl_verts'], cond, res_up=3)
        surf_pred_def = self.deform_mesh(surf_pred_cano, batch['smpl_tfs'])

        #
        img_list = []
        img_list.append(render_mesh_dict(surf_pred_cano,mode='npw')[:,:,:3])
        img_list.append(render_mesh_dict(surf_pred_def, mode='npw')[:,:,:3])
        img_shape_show = np.concatenate(img_list, axis=1)

        save_path = 'medias'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        return surf_pred_cano, surf_pred_def


    def extract_mesh(self, smpl_tfs, smpl_verts, cond, res_up=3):

        def occ_func(pts_c):
            occ_clothed, _, _ = self.forward_shape(pts_c, cond, smpl_tfs, smpl_verts, canonical=True)
            return occ_clothed.reshape(-1,1)

        mesh = generate_mesh_from_img(occ_func,res_up=res_up)
        mesh = {'verts': torch.tensor(mesh.vertices).type_as(smpl_tfs),
                'faces': torch.tensor(mesh.faces, device=smpl_tfs.device)}
        verts = mesh['verts'].unsqueeze(0)
        weights = self.forward_weight(verts, cond)

        mesh['weights'] = weights[0].detach()#.clamp(0,1)[0]
        mesh['weights_color'] = torch.tensor(weights2colors(mesh['weights'].data.cpu().numpy()),
                                              device=smpl_tfs.device).float().clamp(0, 1)
        return mesh