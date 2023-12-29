import torch
import trimesh
import numpy as np
from skimage import measure

from lib.libmise import mise

def generate_mesh(func, current_epoch, nepochs_pretrain_coarse, verts_ori, level_set=0, res_init=32, res_up=3):

    scale = 1.1  # Scale of the padded bbox regarding the tight one.

    verts = verts_ori.data.cpu().numpy()
    gt_bbox = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0)
    gt_center = (gt_bbox[0] + gt_bbox[1]) * 0.5
    gt_scale = (gt_bbox[1] - gt_bbox[0]).max()

    mesh_extractor1 = mise.MISE(res_init, res_up, level_set)
    mesh_extractor2 = mise.MISE(res_init, res_up, level_set)

    points = mesh_extractor1.query()
    # query occupancy grid
    with torch.no_grad():
        while points.shape[0] != 0:
            
            orig_points = points
            points = points.astype(np.float32)
            points = (points / mesh_extractor1.resolution - 0.5) * scale
            points = points * gt_scale + gt_center
            points = torch.tensor(points).type_as(verts_ori)

            values1, _ = func(current_epoch, nepochs_pretrain_coarse, points.unsqueeze(0))
            values1 = values1.data.cpu().numpy().astype(np.float64)[:,0]

            mesh_extractor1.update(orig_points, values1)
            points = mesh_extractor1.query()

    points = mesh_extractor2.query()
    # query occupancy grid
    with torch.no_grad():
        while points.shape[0] != 0:
            orig_points = points
            points = points.astype(np.float32)
            points = (points / mesh_extractor2.resolution - 0.5) * scale
            points = points * gt_scale + gt_center
            points = torch.tensor(points).type_as(verts_ori)

            _, values2 = func(current_epoch, nepochs_pretrain_coarse, points.unsqueeze(0))
            values2 = values2.data.cpu().numpy().astype(np.float64)[:, 0]

            mesh_extractor2.update(orig_points, values2)
            points = mesh_extractor2.query()

    value_grid1 = mesh_extractor1.to_dense()
    value_grid2 = mesh_extractor2.to_dense()
    # value_grid = np.pad(value_grid, 1, "constant", constant_values=-1e6)

    # marching cube marching_cubes, marching_cubes_lewiner
    verts1, faces1, normals1, values1 = measure.marching_cubes(
                                                volume=value_grid1,
                                                gradient_direction='ascent',
                                                level=level_set)
    verts2, faces2, normals2, values2 = measure.marching_cubes(
                                                volume=value_grid2,
                                                gradient_direction='ascent',
                                                level=level_set)
    verts1 = (verts1 / mesh_extractor1.resolution - 0.5) * scale
    verts1 = verts1 * gt_scale + gt_center
    verts2 = (verts2 / mesh_extractor2.resolution - 0.5) * scale
    verts2 = verts2 * gt_scale + gt_center

    meshexport1 = trimesh.Trimesh(verts1, faces1, normals1, vertex_colors=values1)
    meshexport2 = trimesh.Trimesh(verts2, faces2, normals2, vertex_colors=values2)

    # remove disconnect part
    connected_comp1 = meshexport1.split(only_watertight=False)
    max_area = 0
    max_comp = None
    for comp in connected_comp1:
        if comp.area > max_area:
            max_area = comp.area
            max_comp = comp
    meshexport1 = max_comp

    # remove disconnect part
    connected_comp2 = meshexport2.split(only_watertight=False)
    max_area = 0
    max_comp = None
    for comp in connected_comp2:
        if comp.area > max_area:
            max_area = comp.area
            max_comp = comp
    meshexport2 = max_comp

    return meshexport1, meshexport2