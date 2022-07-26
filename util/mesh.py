import torch
import numpy as np

"""
Several functions in this code are based on PyTorch3D.
We slightly modify or wrap the code to provide API for convenience.

If you use these functions, please cite the original work:
- Paper: https://arxiv.org/abs/2007.08501
- Docs: https://pytorch3d.readthedocs.io/en/latest/
- Code: https://github.com/facebookresearch/pytorch3d
"""


def laplacian_smooth(verts, faces, method="uniform", lambd=1.):
    """
    Laplacian smoothing based on pytorch3d.loss.mesh_laplacian_smoothing.
    For the original code please see:
    - https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html
    """
    
    v = verts[0]
    f = faces[0]

    with torch.no_grad():
        if method == "uniform":
            V = v.shape[0]
            edge = torch.cat([f[:,[0,1]],
                              f[:,[1,2]],
                              f[:,[2,0]]], dim=0).T
            L = torch.sparse_coo_tensor(edge, torch.ones_like(edge[0]).float(), (V, V))
            norm_w = 1.0 / torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
            
        elif method == "cot":
            L = laplacian_cot(v, f)
            norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
            idx = norm_w > 0
            norm_w[idx] = 1.0 / norm_w[idx]
            
    v_bar = L.mm(v) * norm_w   # new vertices    
    return ((1-lambd)*v + lambd*v_bar).unsqueeze(0)


def laplacian_cot(verts, faces):
    """
    Laplacian cotangent weights based on pytorch3d.ops.cot_laplacian.
    For the original code please see:
    - https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/laplacian_matrices.html
    
    Note that in previous version (v0.4.0) this function is defined
    in pytorch3d.loss.mesh_laplacian_smoothing.
    """

    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    s = 0.5 * (A + B + C)
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    L += L.t()
    
    return L



def compute_normal(v, f):
    """
    Compute the normal of each vertex based on pytorch3d.structures.meshes.
    For original code please see _compute_vertex_normals function in:
    - https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/structures/meshes.html    
    """
    
    with torch.no_grad():
        n_v = torch.zeros_like(v)   # normals of vertices
        v_f = v[:, f[0]]

        # compute normals of faces
        n_f_0 = torch.cross(v_f[:,:,1]-v_f[:,:,0], v_f[:,:,2]-v_f[:,:,0], dim=2) 
        n_f_1 = torch.cross(v_f[:,:,2]-v_f[:,:,1], v_f[:,:,0]-v_f[:,:,1], dim=2) 
        n_f_2 = torch.cross(v_f[:,:,0]-v_f[:,:,2], v_f[:,:,1]-v_f[:,:,2], dim=2) 

        # sum the faces normals
        n_v = n_v.index_add(1, f[0,:,0], n_f_0)
        n_v = n_v.index_add(1, f[0,:,1], n_f_1)
        n_v = n_v.index_add(1, f[0,:,2], n_f_2)

        n_v = n_v / torch.norm(n_v, dim=-1).unsqueeze(-1) #  + 1e-12)
        
    return n_v


        
# for evaluation

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance

point_face_distance = _PointFaceDistance.apply

def point_to_mesh_dist(pcls, meshes):
    """
    Compute point to mesh distance based on pytorch3d.loss.point_mesh_face_distance.
    For original code please see:
    - https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/point_mesh_distance.html
    """
    
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    return point_to_face.sqrt()


def compute_mesh_distance(v_pred, v_gt, f_pred, f_gt, n_pts=100000, seed=10086):
    """ Compute average symmetric surface distance (ASSD) and Hausdorff distance (HD). """
    
    mesh_pred = Meshes(verts=list(v_pred), faces=list(f_pred))
    mesh_gt = Meshes(verts=list(v_gt), faces=list(f_gt))
    pts_pred = sample_points_from_meshes(mesh_pred, num_samples=n_pts)
    pts_gt = sample_points_from_meshes(mesh_gt, num_samples=n_pts)
    pcl_pred = Pointclouds(pts_pred)
    pcl_gt = Pointclouds(pts_gt)

    x_dist = point_to_mesh_dist(pcl_pred, mesh_gt)
    y_dist = point_to_mesh_dist(pcl_gt, mesh_pred)

    assd = (x_dist.mean().item() + y_dist.mean().item()) / 2

    x_quantile = torch.quantile(x_dist, 0.9).item()
    y_quantile = torch.quantile(y_dist, 0.9).item()
    hd = max(x_quantile, y_quantile)
    
    return assd, hd


from mesh_intersection.bvh_search_tree import BVH

def check_self_intersect(v, f, collisions=8):
    """
    Check mesh self-intersections.
    
    We use the calculate_non_manifold_face_intersection function from
    the Neural Mesh Flow paper. For original code please see:
    - https://github.com/KunalMGupta/NeuralMeshFlow/blob/master/evaluation/tools.py
    """
    
    triangles = v[:, f[0]]
    bvh = BVH(max_collisions=collisions)
    outputs = bvh(triangles)
    outputs = outputs.detach().cpu().numpy().squeeze()
    collisions = outputs[outputs[:, 0] >= 0, :]  # the number of collisions
    
    # ------- old version ------- 
    # This just returns the ratio #collisions / #faces.
    # It will over-estimate the percentage of SIFs.
    # return collisions.shape[0] / f.shape[1] * 100.

    # ------- new version ------- 
    # Find all self-intersected faces using a set without overlapping
    sifs = len(set(collisions.reshape(-1)))
    return sifs / f.shape[1] * 100.


def compute_dice(x, y, dim='2d'):
    """compute dice score of the segmentation"""
    
    if dim == '2d':
        # input size (B, C, H, W)
        dice = (2*(x*y).sum([2,3]) / (x.sum([2,3]) + y.sum([2,3]))).mean(-1)
    elif dim == '3d':
        # input size (B, C, L, W, H)
        dice = (2*(x*y).sum([2,3,4]) / (x.sum([2,3,4]) + y.sum([2,3,4]))).mean(-1)
    return dice.item()