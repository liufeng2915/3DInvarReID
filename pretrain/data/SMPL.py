
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.autograd import Variable

def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices
    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices
    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ij->bjk', [vertices, J_regressor])



def blend_shapes(betas, shapedirs):
    ''' Calculates the per vertex displacement due to the blend shapes
    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shapedirs: torch.tensor Vx3x(num_betas)
        Blend shapes
    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    blend_shape = torch.matmul(betas, shapedirs).view(-1, int(shapedirs.shape[1]/3), 3)
    return blend_shape


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints
    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32
    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    joints_transformed = torch.matmul(transforms.clone(), joints_homogen)

    rel_transforms = transforms - F.pad(joints_transformed, [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms

class SMPL(nn.Module):
    def __init__(self, model_path, joint_type = 'cocoplus'):
        super(SMPL, self).__init__()

        if joint_type not in ['cocoplus', 'lsp']:
            msg = 'unknow joint type: {}, it must be either "cocoplus" or "lsp"'.format(joint_type)
            sys.exit(msg)

        self.model_path = model_path
        self.joint_type = joint_type
        with open(model_path, 'rb') as reader:
            model = pickle.load(reader, encoding='latin1')
        
        self.faces = model['f']

        np_v_template = np.array(model['v_template'], dtype = np.float)
        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]

        np_shapedirs = np.array(model['shapedirs'], dtype = np.float)[:,:,:10]
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        np_J_regressor = np.array(model['J_regressor'].toarray().transpose(1,0), dtype = np.float)
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())
        tensor_J_template = torch.matmul(self.J_regressor.T, self.v_template)
        self.register_buffer('J_template', tensor_J_template.float())

        np_posedirs = np.array(model['posedirs'], dtype = np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

        np_weights = np.array(model['weights'], dtype = np.float)

        vertex_count = np_weights.shape[0] 
        vertex_component = np_weights.shape[1]

        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))
        self.register_buffer('e3', torch.eye(3).float())
        
        self.cur_device = None


    def forward(self, beta, theta, theta_in_rodrigues=True):
        device, dtype = beta.device, beta.dtype
        self.cur_device = torch.device(device.type, device.index)
        num_batch = beta.shape[0]

        v_shaped = smpl_utils.blend_shapes(beta, self.shapedirs) + self.v_template

        J = smpl_utils.vertices2joints(self.J_regressor, v_shaped)
        delta_J = J - self.J_template.squeeze(0).repeat(num_batch,1,1)

        if theta_in_rodrigues:
            Rs = smpl_utils.batch_rodrigues(theta.view(-1, 3)).view(num_batch,24,3,3)
        else: #theta is already rotations
            Rs = theta.view(-1,24,3,3)

        pose_feature = (Rs[:, 1:, :, :] - torch.eye(3, dtype=dtype, device=device)).view(-1, 207)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped
        J_transformed, A = smpl_utils.batch_rigid_transform(Rs, J, self.parents)
        inverse_global_trans = torch.pinverse(A[:,0,:,:])

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)
        
        #v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2)
        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]
        #joints = smpl_utils.vertices2joints(self.J_regressor, verts)

        return v_posed, verts, delta_J, A, W, inverse_global_trans 