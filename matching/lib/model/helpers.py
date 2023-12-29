import numpy as np
import torchvision
import torch

import torch.nn.functional as F
import cv2

def split(a, n):
    k, m = divmod(len(a), n)
    return [ a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n) ]

def vis_images(batch):

    images = []
    for key in batch:
        img = torchvision.utils.make_grid( batch[key], normalize=True, range=(-1,1), nrow=8).permute(1,2,0).data.cpu().numpy()
        images.append(img)
    return np.concatenate(images, axis=0)
    
def select_dict(dict, keys):
    return {key:dict[key] for key in dict if key in keys}

def mask_dict(dict, mask):

    dict_new = {}
    for key in dict:
        dict_new[key] = dict[key][mask]

    return dict_new

def index_dict(dict, start, end):

    for key in dict:
        dict[key] = dict[key][start:end]

    return dict


def grid_sample_feat(feat_map, x):

    n_batch, n_point, _ = x.shape

    if feat_map.ndim == 4:
        x = x[:,:,None,:2]
    elif feat_map.ndim == 5:
        x = x[:,:,None,None,:3]

    feats = F.grid_sample(feat_map, x, align_corners=True, mode='bilinear',padding_mode='zeros')

    return feats.reshape(n_batch, -1, n_point).transpose(1,2)

def expand_cond(cond, x):

    cond = cond[:, None]
    new_shape = list(cond.shape)
    new_shape[0] = x.shape[0]
    new_shape[1] = x.shape[1]
    
    return cond.expand(new_shape)


''' Hierarchical softmax following the kinematic tree of the human body. Imporves convergence speed'''
def hierarchical_softmax(x):
    def softmax(x):
        return torch.nn.functional.softmax(x, dim=-1)

    def sigmoid(x):
        return torch.sigmoid(x)

    n_point, n_dim = x.shape

    prob_all = torch.ones(n_point, 24, device=x.device)

    prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 2, 3]])
    prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))

    prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid(x[:, [4, 5, 6]]))
    prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid(x[:, [4, 5, 6]]))

    prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid(x[:, [7, 8, 9]]))
    prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid(x[:, [7, 8, 9]]))

    prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid(x[:, [10, 11]]))
    prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid(x[:, [10, 11]]))

    prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid(x[:, [24]]) * softmax(x[:, [12, 13, 14]])
    prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid(x[:, [24]]))

    prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid(x[:, [15]]))
    prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid(x[:, [15]]))

    prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid(x[:, [16, 17]]))
    prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid(x[:, [16, 17]]))

    prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid(x[:, [18, 19]]))
    prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid(x[:, [18, 19]]))

    prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid(x[:, [20, 21]]))
    prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid(x[:, [20, 21]]))

    prob_all[:, [22, 23]] = prob_all[:, [20, 21]] * (sigmoid(x[:, [22, 23]]))
    prob_all[:, [20, 21]] = prob_all[:, [20, 21]] * (1 - sigmoid(x[:, [22, 23]]))
    return prob_all

def rectify_pose(pose, rot):
    """
    Rectify AMASS pose in global coord adapted from https://github.com/akanazawa/hmr/issues/50.
 
    Args:
        pose (72,): Pose.
    Returns:
        Rotated pose.
    """
    pose = pose.copy()
    R_rot = cv2.Rodrigues(rot)[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    # new_root = np.linalg.inv(R_abs).dot(R_root)
    new_root = R_rot.dot(R_root)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose


class Dict2Class(object):
    def __init__(self, dict):
        for key in dict:
            setattr(self, key, dict[key])
  
