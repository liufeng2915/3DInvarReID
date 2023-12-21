import os
import torch
import hydra
import pandas
import numpy as np

from lib.model.helpers import Dict2Class

class DataSet(torch.utils.data.Dataset):

    def __init__(self, dataset_path, val=False, opt=None):

        self.dataset_path = hydra.utils.to_absolute_path(dataset_path)
        self.opt = opt
        self.val = val
        self.total_points = 100000

        self.scan_info = pandas.read_csv(hydra.utils.to_absolute_path(opt.data_list),dtype=str)

        self.n_samples = len(self.scan_info)

        self.names = []
        self.person_id = []
        self.cloth_id = []
        for i in range(len(self.scan_info)):
            self.names.append(self.scan_info.iloc[i]['id'])
            self.person_id.append(int(self.scan_info.iloc[i]['person_id']))
            self.cloth_id.append(int(self.scan_info.iloc[i]['cloth_id']))
        self.person_id = np.array(self.person_id)
        self.cloth_id = np.array(self.cloth_id)
        self.num_person_id = np.unique(self.person_id).shape[0]
        self.num_cloth_id = np.unique(self.cloth_id).shape[0]

        if val: self.scan_info = self.scan_info[1::30]

    def __getitem__(self, index):

        scan_info = self.scan_info.iloc[index]
        batch = {}
        batch['index'] = index
        batch['id_index'] = self.person_id[index]
        batch['cloth_index'] = self.cloth_id[index]

        f = np.load(os.path.join(self.dataset_path, scan_info['id'], 'occupancy.npz') )
        batch['smpl_params'] = f['smpl_params'].astype(np.float32)
        batch['smpl_params'][4:7] = 0  ## remove global orientation
        batch['smpl_betas'] =  batch['smpl_params'][76:]
        batch['smpl_thetas'] = batch['smpl_params'][4:76]
        batch['scan_name'] = str(f['scan_name'])
        batch['pts_d'] = f['pts_d']
        batch['occ_naked_gt'] = (f['sdf_smpl_gt'] < 0).astype(np.float32)
        batch['occ_clothed_gt'] = (f['sdf_gt'] < 0).astype(np.float32)

        return batch

    def __len__(self):
        return len(self.scan_info)


class DataProcessor():

    def __init__(self, opt):

        self.opt = opt
        self.total_points = 100000

    def process(self, batch, smpl_server):

        num_batch,_,num_dim = batch['pts_d'].shape

        smpl_output = smpl_server(batch['smpl_params'], absolute=False)
        batch.update(smpl_output)

        random_idx = torch.cat([torch.randint(0, self.total_points, [num_batch, self.opt.points_per_frame, 1], device=batch['pts_d'].device), # 1//8 for bbox samples
                                torch.randint(0 ,self.total_points, [num_batch, self.opt.points_per_frame//8, 1], device=batch['pts_d'].device)+self.total_points], # 1 for surface samples
                                1)
        batch['occ_clothed_gt'] = torch.gather(batch['occ_clothed_gt'], 1, random_idx)
        batch['occ_naked_gt'] = torch.gather(batch['occ_naked_gt'], 1, random_idx)
        batch['pts_d'] = torch.gather(batch['pts_d'], 1, random_idx.expand(-1, -1, num_dim))
            
        return batch

    def process_smpl(self, batch, smpl_server):

        smpl_output = smpl_server(batch['smpl_params'], absolute=False)
        
        return smpl_output

class DataModule(torch.nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage=None):

        # if stage == 'fit':
        self.dataset_train = DataSet(dataset_path=self.opt.dataset_path, opt=self.opt)
        self.dataset_val = DataSet(dataset_path=self.opt.dataset_path, opt=self.opt, val=True)
        self.meta_info = {'n_samples': self.dataset_train.n_samples,
                          'n_identities': self.dataset_train.num_person_id,
                          'n_clothes': self.dataset_train.num_cloth_id,
                          'scan_info': self.dataset_train.scan_info,
                          'dataset_path': self.dataset_train.dataset_path}

        self.meta_info = Dict2Class(self.meta_info)

    def train_dataloader(self):

        dataloader = torch.utils.data.DataLoader(self.dataset_train,
                                batch_size=self.opt.batch_size,
                                num_workers=self.opt.num_workers, 
                                persistent_workers=self.opt.num_workers>0,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=False)
        return dataloader

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(self.dataset_val,
                                batch_size=1,
                                num_workers=self.opt.num_workers, 
                                persistent_workers=self.opt.num_workers>0,
                                shuffle=True,
                                drop_last=False,
                                pin_memory=False)
        return dataloader

