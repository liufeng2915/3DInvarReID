
import os
import cv2
import torch
import hydra
import glob
import pandas
from PIL import Image
import scipy.io
import numpy as np
from lib.utils.render_utils import load_K_Rt_from_P
from torch.utils import data
import lib.utils.img_transforms as T

## THuman rendering image
class SynImage(data.Dataset):
    def __init__(self, opt, path=None, data_list=None, num_images=4):

        self.dataset_path = hydra.utils.to_absolute_path(path)
        self.scan_info = pandas.read_csv(hydra.utils.to_absolute_path(data_list), dtype=str)

        self.scan_info = self.scan_info[:526]
        self.n_samples = len(self.scan_info)
        self.num_images = num_images

        self.sub_names = []
        self.label = []
        for i in range(len(self.scan_info)):
            self.sub_names.append(self.scan_info.iloc[i]['id'])
            self.label.append(int(self.scan_info.iloc[i]['person_id']))

        self.transform = T.Compose([
            T.Resize((opt.img_height, opt.img_width)),
            T.RandomCroping(p=0.5),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(probability=0.5)
        ])
        self.transform_rgb =  T.Compose([
            T.Resize((opt.rendering_img_height, opt.rendering_img_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])

    def load_mask(self, mask_path):

        mask = cv2.imread(mask_path)
        mask = mask[:,:,0].astype(np.float32)
        mask = mask>127.5
        mask = mask.reshape(-1, 1)
        mask = torch.from_numpy(mask).bool()
        return mask

    def load_latent(self, latent_path):

        mat_file = scipy.io.loadmat(latent_path)
        z_naked_shape = mat_file['z_naked_shape'].astype(np.float32)
        z_clothed_shape = mat_file['z_clothed_shape'].astype(np.float32)

        return z_naked_shape, z_clothed_shape

    def load_pts(self, pts_path):

        mat_file = scipy.io.loadmat(pts_path)
        pts_mask = mat_file['mask'].astype(np.float32)
        occ_feat = mat_file['occ_feat'].astype(np.float32)
        pts_c = mat_file['pts_c_clothed'].astype(np.float32)

        return torch.from_numpy(pts_c), torch.from_numpy(pts_mask), torch.from_numpy(occ_feat)

    def load_tfs(self, tfs_path):
        mat_file = scipy.io.loadmat(tfs_path)
        smpl_tfs = mat_file['smpl_tfs'].astype(np.float32)

        return smpl_tfs

    def load_verts(self, verts_path):
        mat_file = scipy.io.loadmat(verts_path)
        smpl_verts = mat_file['verts'].astype(np.float32)

        return smpl_verts

    def load_cam(self, cam_path):

        cam_file = scipy.io.loadmat(cam_path)
        K = cam_file['K'].astype(np.float32)
        RTs = cam_file['pose'].astype(np.float32)
        P = K @ RTs
        intrinsics, pose = load_K_Rt_from_P(None, P)

        return torch.from_numpy(np.vstack([P, [0,0,0,1]]).astype(np.float32)), torch.from_numpy(intrinsics.astype(np.float32)), torch.from_numpy(pose.astype(np.float32))

    def load_image(self, img_path):

        temp_img = Image.open(img_path)
        img = self.transform(temp_img)
        rgb = self.transform_rgb(temp_img)
        rgb = rgb.view(-1,rgb.shape[1]*rgb.shape[2]).permute(1,0)

        return img, rgb

    def __getitem__(self, index):

        #
        sub_name = self.sub_names[index]
        image_folder = os.path.join(self.dataset_path, sub_name, 'image')
        mask_folder = os.path.join(self.dataset_path, sub_name, 'mask')
        cam_folder = os.path.join(self.dataset_path, sub_name, 'cam')
        latent_path = os.path.join(self.dataset_path, sub_name, 'latent.mat')
        tfs_path = os.path.join(self.dataset_path, sub_name, 'tfs.mat')
        verts_path = os.path.join(self.dataset_path, sub_name, 'smpl_verts.mat')
        precomute_pts_folder = os.path.join('path/precompute/data', sub_name, 'color_data')

        # randomly choose num_images
        rand_idx = np.random.permutation(40)[:self.num_images]+1
        image_list = [os.path.join(image_folder, str(i).zfill(3)+'.png') for i in rand_idx]
        mask_list = [os.path.join(mask_folder, str(i).zfill(3)+'.png') for i in rand_idx]
        cam_list = [os.path.join(cam_folder, str(i).zfill(3)+'.mat') for i in rand_idx]
        pts_list = [os.path.join(precomute_pts_folder, str(i).zfill(3)+'.mat') for i in rand_idx]

        # load images
        images = [self.load_image(i) for i in image_list]
        input_img = torch.cat([images[i][0].unsqueeze(0) for i in range(self.num_images)])
        rgb = torch.cat([images[i][1].unsqueeze(0) for i in range(self.num_images)])

        # load mask
        mask = [self.load_mask(i) for i in mask_list]
        mask = torch.cat([mask[i].unsqueeze(0) for i in range(self.num_images)])

        # load cam
        cam_data =  [self.load_cam(i) for i in cam_list]
        cam_proj = torch.cat([cam_data[i][0].unsqueeze(0) for i in range(self.num_images)])
        intrinsics = torch.cat([cam_data[i][1].unsqueeze(0) for i in range(self.num_images)])
        pose = torch.cat([cam_data[i][2].unsqueeze(0) for i in range(self.num_images)])

        # load latent
        z_naked_shape, z_clothed_shape = self.load_latent(latent_path)
        z_naked_shape = torch.from_numpy(z_naked_shape).repeat(self.num_images,1)
        z_clothed_shape = torch.from_numpy(z_clothed_shape).repeat(self.num_images, 1)

        # load precomute pts
        pts_data =  [self.load_pts(i) for i in pts_list]
        pts_c = torch.cat([pts_data[i][0].unsqueeze(0) for i in range(self.num_images)])
        pts_mask = torch.cat([pts_data[i][1].unsqueeze(0) for i in range(self.num_images)])
        occ_feat = torch.cat([pts_data[i][2].unsqueeze(0) for i in range(self.num_images)])

        # load tfs
        smpl_tfs = self.load_tfs(tfs_path)
        smpl_tfs = torch.from_numpy(smpl_tfs).unsqueeze(0).repeat(self.num_images, 1, 1, 1)

        # load smpl_verts (mean_shape + pose)
        smpl_verts = self.load_verts(verts_path)
        smpl_verts = torch.from_numpy(smpl_verts).unsqueeze(0).repeat(self.num_images, 1, 1)

        # label
        label = self.label[index]
        label = torch.LongTensor([label]*self.num_images)

        batch = {}
        batch['sub_name'] = [sub_name]*self.num_images
        batch['image_list'] = [i.split('/')[-1][:-4] for i in image_list]
        batch['label'] = label
        batch['cam_id'] = torch.LongTensor([-1] * self.num_images)
        batch['input_img'] = input_img
        batch['rgb'] = rgb
        batch['mask'] = mask
        batch['cam_proj'] = cam_proj
        batch['intrinsics'] = intrinsics
        batch['pose'] = pose
        batch['latent_flag'] = torch.LongTensor([1]*self.num_images)
        batch['label_flag'] = torch.LongTensor([0] * self.num_images)
        batch['z_naked_shape'] = z_naked_shape
        batch['z_clothed_shape'] = z_clothed_shape
        batch['smpl_tfs'] = smpl_tfs
        batch['smpl_verts'] = smpl_verts
        batch['pts_c'] = pts_c
        batch['pts_mask'] = pts_mask
        batch['occ_feat'] = occ_feat

        return batch

    def __len__(self):
        return len(self.scan_info)

class CelibReidDataset(data.Dataset):
    def __init__(self, opt, data_path, relabel, data_type):

        self.data_type = data_type
        if data_type == 'gallery':
            self.img_path = os.path.join(data_path, 'gallery/image')
            self.mask_path = os.path.join(data_path, 'gallery/mask')
            self.cam_path = os.path.join(data_path, 'gallery/cam')
            self.tfs_path = os.path.join(data_path, 'gallery/tfs')
            self.verts_path = os.path.join(data_path, 'gallery/smpl_verts')
        elif data_type == 'query':
            self.img_path = os.path.join(data_path, 'query/image')
            self.mask_path = os.path.join(data_path, 'query/mask')
            self.cam_path = os.path.join(data_path, 'query/cam')
            self.tfs_path = os.path.join(data_path, 'query/tfs')
            self.verts_path = os.path.join(data_path, 'query/smpl_verts')

        self.transform = T.Compose([
            T.Resize((opt.img_height, opt.img_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.transform_rgb =  T.Compose([
            T.Resize((opt.rendering_img_height, opt.rendering_img_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])

        self.fnames = []
        self.pids = []
        self.ret = []
        self.preprocess_img_path(relabel=relabel)
        self.num_data = int(len(self.fnames))
        self.num_classes = len(np.unique(self.pids))

    def preprocess_img_path(self, relabel):

        fpaths = sorted(glob.glob(os.path.join(self.img_path, '*.jpg')))
        all_pids = {}
        for fpath in fpaths:
            fname = os.path.basename(fpath)
            pid = int(fname.split('_')[0])
            cam = int(fname.split('_')[1])

            if pid == -1: continue  # junk images are just ignored

            self.fnames.append(fpath)

            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid

            pid = all_pids[pid]
            self.pids.append(pid)
            self.ret.append((fname, pid, cam))

    def load_mask(self, mask_path):

        mask = cv2.imread(mask_path)
        mask = mask[:,:,0].astype(np.float32)
        mask = mask>127.5
        mask = mask.reshape(-1, 1)
        mask = torch.from_numpy(mask).bool()
        return mask

    def load_tfs(self, tfs_path):
        mat_file = scipy.io.loadmat(tfs_path)
        smpl_tfs = mat_file['smpl_tfs'].astype(np.float32)

        return smpl_tfs

    def load_verts(self, verts_path):
        mat_file = scipy.io.loadmat(verts_path)
        smpl_verts = mat_file['verts'].astype(np.float32)

        return smpl_verts

    def load_cam(self, cam_path):

        cam_file = scipy.io.loadmat(cam_path)
        K = cam_file['K'].astype(np.float32)
        RTs = cam_file['pose'].astype(np.float32)
        P = K @ RTs
        intrinsics, pose = load_K_Rt_from_P(None, P)

        return torch.from_numpy(np.vstack([P, [0,0,0,1]]).astype(np.float32)), torch.from_numpy(intrinsics.astype(np.float32)), torch.from_numpy(pose.astype(np.float32))

    def load_image(self, img_path):

        temp_img = Image.open(img_path)
        img = self.transform(temp_img)

        rgb = self.transform_rgb(temp_img)
        rgb = rgb.view(-1,rgb.shape[1]*rgb.shape[2]).permute(1,0)

        return img, rgb

    def __getitem__(self, index):

        # image and identity label
        input_img, rgb = self.load_image(self.fnames[index])
        label = self.pids[index]
        cam_id = self.ret[index][2]
        names = os.path.basename(self.fnames[index])[:-4]

        if self.data_type == 'train':

            # load mask
            mask = self.load_mask(os.path.join(self.mask_path, names + '.png'))
            # load cam
            cam_proj, intrinsics, pose = self.load_cam(os.path.join(self.cam_path, names + '.mat'))
            # load latent
            z_naked_shape = np.zeros((512,1))
            z_clothed_shape = np.zeros((512, 1))
            # load tfs
            smpl_tfs = self.load_tfs(os.path.join(self.tfs_path, names + '.mat'))
            # load smpl_verts (mean_shape + pose)
            smpl_verts = self.load_verts(os.path.join(self.verts_path, names + '.mat'))

            batch = {}
            batch['sub_name'] = names.split('_')[0]
            batch['image_list'] = names
            batch['label'] = label
            batch['cam_id'] = cam_id
            batch['input_img'] = input_img
            batch['rgb'] = rgb
            batch['mask'] = mask
            batch['cam_proj'] = cam_proj
            batch['intrinsics'] = intrinsics
            batch['pose'] = pose
            batch['latent_flag'] = torch.LongTensor([0])
            batch['z_naked_shape'] = z_naked_shape.astype(np.float32)
            batch['z_clothed_shape'] = z_clothed_shape.astype(np.float32)
            batch['smpl_tfs'] = smpl_tfs
            batch['smpl_verts'] = smpl_verts
        else:
            batch = {}
            batch['sub_name'] = names.split('_')[0]
            batch['image_list'] = names
            batch['label'] = label
            batch['cam_id'] = cam_id
            batch['input_img'] = input_img
            batch['rgb'] = rgb

        return batch

    def __len__(self):
        return self.num_data

class CelibReidDataset_Train(data.Dataset):
    def __init__(self, opt, data_path, relabel, num_images=4):

        self.img_path = os.path.join(data_path, 'train/image')
        self.mask_path = os.path.join(data_path, 'train/mask')
        self.cam_path = os.path.join(data_path, 'train/cam')
        self.tfs_path = os.path.join(data_path, 'train/tfs')
        self.verts_path = os.path.join(data_path, 'train/smpl_verts')

        self.transform = T.Compose([
            T.Resize((opt.img_height, opt.img_width)),
            T.RandomCroping(p=0.5),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(probability=0.5)
        ])
        self.transform_rgb =  T.Compose([
            T.Resize((opt.rendering_img_height, opt.rendering_img_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])

        self.fnames = []
        self.pids = []
        self.ret = []
        self.preprocess_img_path(relabel=relabel)
        self.num_data = int(len(self.fnames))
        self.upids = np.unique(self.pids)
        self.num_classes = len(self.upids)
        self.num_images = num_images

    def preprocess_img_path(self, relabel):

        fpaths = sorted(glob.glob(os.path.join(self.img_path, '*.jpg')))
        all_pids = {}
        for fpath in fpaths:
            fname = os.path.basename(fpath)
            pid = int(fname.split('_')[0])
            cam = int(fname.split('_')[1])

            if pid == -1: continue  # junk images are just ignored

            self.fnames.append(fpath)

            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid

            pid = all_pids[pid]
            self.pids.append(pid)
            self.ret.append((fname, pid, cam))

    def load_mask(self, mask_path):

        mask = cv2.imread(mask_path)
        mask = mask[:,:,0].astype(np.float32)
        mask = mask>127.5
        mask = mask.reshape(-1, 1)
        mask = torch.from_numpy(mask).bool()
        return mask

    def load_tfs(self, tfs_path):
        mat_file = scipy.io.loadmat(tfs_path)
        smpl_tfs = mat_file['smpl_tfs'].astype(np.float32)

        return torch.from_numpy(smpl_tfs.astype(np.float32))

    def load_verts(self, verts_path):
        mat_file = scipy.io.loadmat(verts_path)
        smpl_verts = mat_file['verts'].astype(np.float32)

        return torch.from_numpy(smpl_verts.astype(np.float32))

    def load_cam(self, cam_path):

        cam_file = scipy.io.loadmat(cam_path)
        K = cam_file['K'].astype(np.float32)
        RTs = cam_file['pose'].astype(np.float32)
        P = K @ RTs
        intrinsics, pose = load_K_Rt_from_P(None, P)

        return torch.from_numpy(np.vstack([P, [0,0,0,1]]).astype(np.float32)), torch.from_numpy(intrinsics.astype(np.float32)), torch.from_numpy(pose.astype(np.float32))

    def load_image(self, img_path):

        temp_img = Image.open(img_path)
        img = self.transform(temp_img)

        rgb = self.transform_rgb(temp_img)
        rgb = rgb.view(-1,rgb.shape[1]*rgb.shape[2]).permute(1,0)

        return img, rgb

    def __getitem__(self, index):

        pid = self.upids[index]
        img_idx = np.where(pid==self.pids)[0]
        if len(img_idx)<self.num_images:
            img_idx = img_idx[np.random.choice(len(img_idx), self.num_images)]
        else:
            img_idx = img_idx[np.random.permutation(len(img_idx))[:self.num_images]]

        # image and identity label
        images = [self.load_image(self.fnames[i]) for i in img_idx]
        input_img = torch.cat([images[i][0].unsqueeze(0) for i in range(self.num_images)])
        rgb = torch.cat([images[i][1].unsqueeze(0) for i in range(self.num_images)])

        label = np.array(self.pids)[img_idx]
        cam_id = np.array([self.ret[i][2] for i in img_idx])
        names = [os.path.basename(self.fnames[i])[:-4] for i in img_idx]

        # load mask
        mask = [self.load_mask(os.path.join(self.mask_path, i + '.png')) for i in names]
        mask = torch.cat([mask[i].unsqueeze(0) for i in range(self.num_images)])

        # load cam
        cam_data = [self.load_cam(os.path.join(self.cam_path, i + '.mat')) for i in names]
        cam_proj = torch.cat([cam_data[i][0].unsqueeze(0) for i in range(self.num_images)])
        intrinsics = torch.cat([cam_data[i][1].unsqueeze(0) for i in range(self.num_images)])
        pose = torch.cat([cam_data[i][2].unsqueeze(0) for i in range(self.num_images)])

        # load latent
        z_naked_shape = np.zeros((self.num_images,512))
        z_clothed_shape = np.zeros((self.num_images,512))

        # load pts
        pts_c = np.zeros((self.num_images,256*128,3))
        pts_mask = np.zeros((self.num_images,256*128,1))
        occ_feat = np.zeros((self.num_images,256*128,256)) 

        # load tfs
        tfs_data = [self.load_tfs(os.path.join(self.tfs_path, i + '.mat')) for i in names]
        smpl_tfs = torch.cat([tfs_data[i].unsqueeze(0) for i in range(self.num_images)])

        # load smpl_verts (mean_shape + pose)
        verts_data = [self.load_verts(os.path.join(self.verts_path, i + '.mat')) for i in names]
        smpl_verts = torch.cat([verts_data[i].unsqueeze(0) for i in range(self.num_images)])

        batch = {}
        batch['sub_name'] = [i.split('_')[0] for i in names]
        batch['image_list'] = names
        batch['label'] = label
        batch['cam_id'] = cam_id
        batch['input_img'] = input_img
        batch['rgb'] = rgb
        batch['mask'] = mask
        batch['cam_proj'] = cam_proj
        batch['intrinsics'] = intrinsics
        batch['pose'] = pose
        batch['latent_flag'] = torch.LongTensor([0]*self.num_images)
        batch['label_flag'] = torch.LongTensor([1] * self.num_images)
        batch['z_naked_shape'] = z_naked_shape.astype(np.float32)
        batch['z_clothed_shape'] = z_clothed_shape.astype(np.float32)
        batch['smpl_tfs'] = smpl_tfs
        batch['smpl_verts'] = smpl_verts
        batch['pts_c'] = pts_c.astype(np.float32)
        batch['pts_mask'] = pts_mask.astype(np.float32)
        batch['occ_feat'] = occ_feat.astype(np.float32)

        return batch

    def __len__(self):
        return self.num_classes

class DataProcessor():

    def __init__(self, opt):
        self.opt = opt

    def print_tensor(self, batch):

        for key in batch.keys():
            if torch.is_tensor(batch[key]):
                print(key, batch[key].shape, batch[key].dtype)

        return 1

    def to_gpu(self, batch):

        for key in batch.keys():
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].cuda()

        return batch

    def shuffle_tensor(self, batch):

        bz = batch['input_img'].shape[0]
        rand_idx = np.random.permutation(bz)

        for key in batch.keys():
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key][rand_idx]
            else:
                batch[key] = [batch[key][i] for i in rand_idx]
            x = 1

        return batch

    def organize_batch(self, batch):

        #
        batch['sub_name'] = [item for sublist in batch['sub_name'] for item in sublist]
        batch['image_list'] = [item for sublist in batch['image_list'] for item in sublist]
        batch['label'] = batch['label'].view(-1)
        batch['cam_id'] = batch['cam_id'].view(-1)
        batch['input_img'] = batch['input_img'].view(-1, 3, batch['input_img'].shape[3], batch['input_img'].shape[4])
        batch['rgb'] = batch['rgb'].view(-1, batch['rgb'].shape[2], batch['rgb'].shape[3])
        batch['mask'] = batch['mask'].view(-1, batch['mask'].shape[2], batch['mask'].shape[3])
        batch['cam_proj'] = batch['cam_proj'].view(-1, batch['cam_proj'].shape[2], batch['cam_proj'].shape[3])
        batch['intrinsics'] = batch['intrinsics'].view(-1, batch['intrinsics'].shape[2], batch['intrinsics'].shape[3])
        batch['pose'] = batch['pose'].view(-1, batch['pose'].shape[2], batch['pose'].shape[3])
        batch['latent_flag'] = batch['latent_flag'].view(-1)
        batch['label_flag'] = batch['label_flag'].view(-1)
        batch['z_naked_shape'] = batch['z_naked_shape'].view(-1, batch['z_naked_shape'].shape[-1])
        batch['z_clothed_shape'] = batch['z_clothed_shape'].view(-1, batch['z_clothed_shape'].shape[-1])
        batch['smpl_tfs'] = batch['smpl_tfs'].view(-1, batch['smpl_tfs'].shape[2], batch['smpl_tfs'].shape[3], batch['smpl_tfs'].shape[4])
        batch['smpl_verts'] = batch['smpl_verts'].view(-1, batch['smpl_verts'].shape[2], batch['smpl_verts'].shape[3])

        batch['pts_c'] = batch['pts_c'].view(-1, batch['pts_c'].shape[2], batch['pts_c'].shape[3])
        batch['pts_mask'] = batch['pts_mask'].view(-1, batch['pts_mask'].shape[2], batch['pts_mask'].shape[3])
        batch['occ_feat'] = batch['occ_feat'].view(-1, batch['occ_feat'].shape[2], batch['occ_feat'].shape[3])

        return batch

    def combine_batch(self, syn_batch, real_batch):

        #
        syn_batch = self.organize_batch(syn_batch)
        real_batch = self.organize_batch(real_batch)

        # combine
        batch = {}
        batch['sub_name'] = syn_batch['sub_name'] + real_batch['sub_name']
        batch['image_list'] = syn_batch['image_list'] + real_batch['image_list']
        batch['label'] = torch.cat((syn_batch['label'], real_batch['label']), dim=0)
        batch['cam_id'] = torch.cat((syn_batch['cam_id'], real_batch['cam_id']), dim=0)
        batch['input_img'] = torch.cat((syn_batch['input_img'], real_batch['input_img']), dim=0)
        batch['rgb'] = torch.cat((syn_batch['rgb'], real_batch['rgb']), dim=0)
        batch['mask'] = torch.cat((syn_batch['mask'], real_batch['mask']), dim=0)
        batch['cam_proj'] = torch.cat((syn_batch['cam_proj'], real_batch['cam_proj']), dim=0)
        batch['intrinsics'] = torch.cat((syn_batch['intrinsics'], real_batch['intrinsics']), dim=0)
        batch['pose'] = torch.cat((syn_batch['pose'], real_batch['pose']), dim=0)
        batch['latent_flag'] = torch.cat((syn_batch['latent_flag'], real_batch['latent_flag'].squeeze(-1)), dim=0)
        batch['label_flag'] = torch.cat((syn_batch['label_flag'], real_batch['label_flag'].squeeze(-1)), dim=0)
        batch['z_naked_shape'] = torch.cat((syn_batch['z_naked_shape'], real_batch['z_naked_shape']), dim=0)
        batch['z_clothed_shape'] = torch.cat((syn_batch['z_clothed_shape'], real_batch['z_clothed_shape']), dim=0)
        batch['smpl_tfs'] = torch.cat((syn_batch['smpl_tfs'], real_batch['smpl_tfs']), dim=0)
        batch['smpl_verts'] = torch.cat((syn_batch['smpl_verts'], real_batch['smpl_verts']), dim=0)

        batch['pts_c'] =  torch.cat((syn_batch['pts_c'], real_batch['pts_c']), dim=0)
        batch['pts_mask'] =  torch.cat((syn_batch['pts_mask'].squeeze(1), real_batch['pts_mask'].squeeze(-1)), dim=0)
        batch['occ_feat'] = torch.cat((syn_batch['occ_feat'], real_batch['occ_feat']), dim=0)

        #
        #self.print_tensor(batch)

        # shuffle
        batch = self.shuffle_tensor(batch)

        # GPU
        batch = self.to_gpu(batch)

        return batch

class DataModule(torch.nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self):

        self.syn_dataset = SynImage(opt=self.opt, path='./data', data_list='./lib/dataset/data.csv', num_images=self.opt.num_instances)
        if self.opt.reid_data == 'celeb':
            self.real_train_dataset =  CelibReidDataset_Train(opt=self.opt, data_path='../../data/Celeb_reID', relabel=True, num_images=self.opt.num_instances)
            self.real_gallery_dataset =  CelibReidDataset(opt=self.opt, data_path='../../data/Celeb_reID', relabel=True, data_type='gallery')
            self.real_query_dataset =  CelibReidDataset(opt=self.opt, data_path='../../data/Celeb_reID', relabel=True, data_type='query')

    def train_dataloader(self):

        syn_dataloader = torch.utils.data.DataLoader(self.syn_dataset,
                                batch_size=int(self.opt.img_batch_size/self.opt.num_instances),
                                num_workers=self.opt.num_workers,
                                persistent_workers=self.opt.num_workers>0,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=False)
        real_train_dataloader = torch.utils.data.DataLoader(self.real_train_dataset,
                                batch_size=int(self.opt.img_batch_size/self.opt.num_instances),
                                num_workers=self.opt.num_workers,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=False)
        return syn_dataloader, real_train_dataloader

    def val_dataloader(self):
        real_gallery_dataloader = torch.utils.data.DataLoader(self.real_gallery_dataset,
                                batch_size=128,
                                num_workers=self.opt.num_workers,
                                shuffle=False,
                                drop_last=False)
        real_query_dataloader = torch.utils.data.DataLoader(self.real_query_dataset,
                                batch_size=128,
                                num_workers=self.opt.num_workers,
                                shuffle=False,
                                drop_last=False)

        return real_gallery_dataloader, real_query_dataloader