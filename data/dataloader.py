import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
import trimesh
from data.preprocess import process_volume, process_surface
from util.mesh import laplacian_smooth, compute_normal


# ----------------------------
#  for segmentation
# ----------------------------

class SegData():
    def __init__(self, vol, seg):
        self.vol = torch.Tensor(vol)
        self.seg = torch.Tensor(seg)

        vol = []
        seg = []

        
class SegDataset(Dataset):
    def __init__(self, data):
        super(Dataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        brain = self.data[i]
        return brain.vol, brain.seg
    
    
def load_seg_data(config, data_usage='train'):
    """
    data_dir: the directory of your dataset
    data_name: [hcp, adni, dhcp, ...]
    data_usage: [train, valid, test]
    """
    
    data_name = config.data_name
    data_dir = config.data_dir
    data_dir = data_dir + data_usage + '/'

    subject_list = sorted(os.listdir(data_dir))
    data_list = []

    for i in tqdm(range(len(subject_list))):
        subid = subject_list[i]

        if data_name == 'hcp' or data_name == 'adni':
            brain = nib.load(data_dir+subid+'/mri/orig.mgz')
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 255.).astype(np.float32)
            brain_arr = process_volume(brain_arr, data_name)

            seg = nib.load(data_dir+subid+'/mri/ribbon.mgz')
            seg_arr = seg.get_fdata()
            seg_arr = process_volume(seg_arr, data_name)[0]
            seg_left = (seg_arr == 2).astype(int)    # left wm
            seg_right = (seg_arr == 41).astype(int)  # right wm

            seg_arr = np.zeros_like(seg_left, dtype=int)  # final label
            seg_arr += 1 * seg_left
            seg_arr += 2 * seg_right
    
        elif data_name == 'dhcp':
            brain = nib.load(data_dir+subid+'/'+subid+'_T2w.nii.gz')
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 20).astype(np.float32)
            brain_arr = process_volume(brain_arr, data_name)
            
            # wm_label is the generated segmentation by projecting surface into the volume
            seg_arr = np.load(data_dir+subid+'/'+subid+'_wm_label.npy', allow_pickle=True)
            seg_arr = process_volume(seg_arr, data_name)[0]
            
        segdata = SegData(vol=brain_arr, seg=seg_arr)
        # add to data list
        data_list.append(segdata)

    # make dataset
    dataset = SegDataset(data_list)
    
    return dataset



# ----------------------------
#  for surface reconstruction
# ----------------------------

class BrainData():
    """
    v_in: vertices of input surface
    f_in: faces of ground truth surface
    v_gt: vertices of input surface
    f_gt: faces of ground truth surface
    """
    def __init__(self, volume, v_in, v_gt, f_in, f_gt):
        self.v_in = torch.Tensor(v_in)
        self.f_in = torch.LongTensor(f_in)
        self.v_gt = torch.Tensor(v_gt)
        self.f_gt = torch.LongTensor(f_gt)
        self.volume = torch.from_numpy(volume)
        # free the memory
        volume = []
        v_in = []
        f_in = []
        v_gt = []
        f_gt = []
        
        
class BrainDataset(Dataset):
    def __init__(self, data):
        super(Dataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        brain = self.data[i]
        return brain.volume, brain.v_in, brain.v_gt,\
                brain.f_in, brain.f_gt


def load_surf_data(config, data_usage='train'):
    """
    data_dir: the directory of your dataset
    init_dir: the directory of created initial surfaces
    data_name: [hcp, adni, ...]
    data_usage: [train, valid, test]
    surf_type: [wm, gm]
    surf_hemi: [lh, rh]
    n_inflate: num of iterations for WM surface inflation
    rho: inflation scale
    lambd: weight for Laplacian smoothing
    """
    
    data_dir = config.data_dir
    data_dir = data_dir + data_usage + '/'
    data_name = config.data_name
    init_dir = config.init_dir + data_usage + '/'
    surf_type = config.surf_type
    surf_hemi = config.surf_hemi
    device = config.device

    n_inflate = config.n_inflate   # 2
    rho = config.rho    # 0.002
    lambd = config.lambd
    
    subject_list = sorted(os.listdir(data_dir))
    data_list = []

    for i in tqdm(range(len(subject_list))):
        subid = subject_list[i]
        
        # ------- load brain MRI ------- 
        if data_name == 'hcp' or data_name == 'adni':
            brain = nib.load(data_dir+subid+'/mri/orig.mgz')
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 255.).astype(np.float32)
        elif data_name == 'dhcp':
            brain = nib.load(data_dir+subid+'/'+subid+'_T2w.nii.gz')
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 20).astype(np.float16)
        brain_arr = process_volume(brain_arr, data_name)
        
        # ------- wm surface reconstruction ------- 
        if surf_type == 'wm':
            # ------- load input surface ------- 
            # inputs is the initial surface
            mesh_in = trimesh.load(init_dir+'init_'+data_name+'_'+surf_hemi+'_'+subid+'.obj')
            v_in, f_in = mesh_in.vertices, mesh_in.faces
            
            # ------- load gt surface ------- 
            if data_name == 'hcp':
                # depends on your FreeSurfer version
                v_gt, f_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.white.deformed')
            elif data_name == 'adni':
                v_gt, f_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.white')
            elif data_name == 'dhcp':
                if surf_hemi == 'lh':
                    surf_gt = nib.load(data_dir+subid+'/'+subid+'_left_wm.surf.gii')
                    v_gt, f_gt = surf_gt.agg_data('pointset'), surf_gt.agg_data('triangle')
                elif surf_hemi == 'rh':
                    surf_gt = nib.load(data_dir+subid+'/'+subid+'_right_wm.surf.gii')
                    v_gt, f_gt = surf_gt.agg_data('pointset'), surf_gt.agg_data('triangle')
                # apply the affine transformation provided by brain MRI nifti
                v_tmp = np.ones([v_gt.shape[0],4])
                v_tmp[:,:3] = v_gt
                v_gt = v_tmp.dot(np.linalg.inv(brain.affine).T)[:,:3]
            v_gt, f_gt = process_surface(v_gt, f_gt, data_name)
        
        # ------- pial surface reconstruction ------- 
        elif surf_type == 'gm':
            # ------- load input surface ------- 
            # input is wm surface
            if data_name == 'hcp':
                # depends on your FreeSurfer version
                v_in, f_in = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.white.deformed')
            elif data_name == 'adni':
                v_in, f_in = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.white')
            elif data_name == 'dhcp':
                if surf_hemi == 'lh':
                    surf_in = nib.load(data_dir+subid+'/'+subid+'_left_wm.surf.gii')
                    v_in, f_in = surf_in.agg_data('pointset'), surf_in.agg_data('triangle')
                elif surf_hemi == 'rh':
                    surf_in = nib.load(data_dir+subid+'/'+subid+'_right_wm.surf.gii')
                    v_in, f_in = surf_in.agg_data('pointset'), surf_in.agg_data('triangle')
                v_tmp = np.ones([v_in.shape[0],4])
                v_tmp[:,:3] = v_in
                v_in = v_tmp.dot(np.linalg.inv(brain.affine).T)[:,:3]
            v_in, f_in = process_surface(v_in, f_in, data_name)
            
            # ------- inflating and smoothing ------- 
            v_in = torch.Tensor(v_in).unsqueeze(0).to(device)
            f_in = torch.LongTensor(f_in).unsqueeze(0).to(device)
            for i in range(n_inflate):
                v_in = laplacian_smooth(v_in, f_in, lambd=lambd)
                n_in = compute_normal(v_in, f_in)
                v_in += rho * n_in   # inflate along normal direction
            v_in = v_in.cpu().numpy()[0]
            f_in = f_in.cpu().numpy()[0]
            
            # ------- load gt surface ------- 
            if data_name == 'hcp':
                v_gt, f_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.pial.deformed')
            elif data_name == 'adni':
                v_gt, f_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.pial')
            elif data_name == 'dhcp':
                if surf_hemi == 'lh':
                    surf_gt = nib.load(data_dir+subid+'/'+subid+'_left_pial.surf.gii')
                    v_gt, f_gt = surf_gt.agg_data('pointset'), surf_gt.agg_data('triangle')
                elif surf_hemi == 'rh':
                    surf_gt = nib.load(data_dir+subid+'/'+subid+'_right_pial.surf.gii')
                    v_gt, f_gt = surf_gt.agg_data('pointset'), surf_gt.agg_data('triangle')
                v_tmp = np.ones([v_gt.shape[0],4])
                v_tmp[:,:3] = v_gt
                v_gt = v_tmp.dot(np.linalg.inv(brain.affine).T)[:,:3]
            v_gt, f_gt = process_surface(v_gt, f_gt, data_name)
            
        braindata = BrainData(volume=brain_arr, v_in=v_in, v_gt=v_gt,
                              f_in=f_in, f_gt=f_gt)
        # add to data list
        data_list.append(braindata)
        
    # make dataset
    dataset = BrainDataset(data_list)
#     torch.cuda.empty_cache()
    
    return dataset
