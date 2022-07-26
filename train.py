import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from data.dataloader import load_surf_data, load_seg_data
from model.net import CortexODE, Unet
from util.mesh import compute_dice

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

import logging
from torchdiffeq import odeint_adjoint as odeint
from config import load_config


def train_seg(config):
    """training WM segmentation"""
    
    # --------------------------
    # load configuration
    # --------------------------
    model_dir = config.model_dir   # the directory to save the checkpoints
    data_name = config.data_name
    device = config.device
    tag = config.tag
    n_epochs = config.n_epochs
    lr = config.lr

    # start training logging
    logging.basicConfig(filename=model_dir+'model_seg_'+data_name+'_'+tag+'.log',
                        level=logging.INFO, format='%(asctime)s %(message)s')
    
    # --------------------------
    # load dataset
    # --------------------------
    logging.info("load dataset ...")
    trainset = load_seg_data(config, data_usage='train')
    validset = load_seg_data(config, data_usage='valid')

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False)
    
    # --------------------------
    # initialize model
    # --------------------------
    logging.info("initalize model ...")
    segnet = Unet(c_in=1, c_out=3).to(device)
    optimizer = optim.Adam(segnet.parameters(), lr=lr)
    # in case you need to load a checkpoint
    # segnet.load_state_dict(torch.load(model_dir+'model_seg_'+data_name+'_'+tag+'_XXepochs.pt'))
    # segnet.load_state_dict(torch.load('./ckpts/pretrained/adni/model_seg_adni_pretrained.pt'))

    # --------------------------
    # training model
    # --------------------------
    logging.info("start training ...")
    for epoch in tqdm(range(n_epochs+1)):
        avg_loss = []
        for idx, data in enumerate(trainloader):
            volume_in, seg_gt = data

            optimizer.zero_grad()
            volume_in = volume_in.to(device)
            seg_gt = seg_gt.long().to(device)

            seg_out = segnet(volume_in)
            loss = nn.CrossEntropyLoss()(seg_out, seg_gt)
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        logging.info("epoch:{}, loss:{}".format(epoch,np.mean(avg_loss)))

        if epoch % 10 == 0:
            logging.info('-------------validation--------------')
            with torch.no_grad():
                avg_error = []
                avg_dice = []
                for idx, data in enumerate(validloader):
                    volume_in, seg_gt = data
                    volume_in = volume_in.to(device)
                    seg_gt = seg_gt.long().to(device)
                    seg_out = segnet(volume_in)
                    avg_error.append(nn.CrossEntropyLoss()(seg_out, seg_gt).item())
                    
                    # compute dice score
                    seg_out = torch.argmax(seg_out, dim=1)
                    seg_out = F.one_hot(seg_out, num_classes=3).permute(0,4,1,2,3)[:,1:]
                    seg_gt = F.one_hot(seg_gt, num_classes=3).permute(0,4,1,2,3)[:,1:]
                    dice = compute_dice(seg_out, seg_gt, '3d')
                    avg_dice.append(dice)
                logging.info("epoch:{}, validation error:{}".format(epoch, np.mean(avg_error)))
                logging.info("Dice score:{}".format(np.mean(avg_dice)))
                logging.info('-------------------------------------')
        # save model checkpoints
        if epoch % 20 == 0:
            torch.save(segnet.state_dict(),
                       model_dir+'model_seg_'+data_name+'_'+tag+'_'+str(epoch)+'epochs.pt')
    # save final model
    torch.save(segnet.state_dict(),
               model_dir+'model_seg_'+data_name+'_'+tag+'.pt')


def train_surf(config):
    """
    Training CortexODE for cortical surface reconstruction
    using adjoint sensitivity method proposed in neural ODE
    
    For original neural ODE paper please see:
    - Chen et al. Neural ordinary differential equations. NeurIPS, 2018.
      Paper: https://arxiv.org/abs/1806.07366v5
      Code: https://github.com/rtqichen/torchdiffeq
    
    Note: using seminorm in adjoint method can accelerate the training, but it
    will cause exploding gradients for explicit methods in our experiments.

    For seminorm please see:
    - Patrick et al. Hey, that's not an ODE: Faster ODE Adjoints via Seminorms. ICML, 2021.
      Paper: https://arxiv.org/abs/2009.09457
      Code: https://github.com/patrick-kidger/FasterNeuralDiffEq

    Configurations (see config.py):
    model_dir: directory to save your checkpoints
    data_name: [hcp, adni, ...]
    surf_type: [wm, gm]
    surf_hemi: [lh, rh]
    """
    
    # --------------------------
    # load configuration
    # --------------------------
    model_dir = config.model_dir
    data_name = config.data_name
    surf_type = config.surf_type
    surf_hemi = config.surf_hemi
    device = config.device
    tag = config.tag
    
    n_epochs = config.n_epochs
    n_samples = config.n_samples
    lr = config.lr
    
    C = config.dim_h     # hidden dimension of features
    K = config.kernel_size    # kernel / cube size
    Q = config.n_scale    # multi-scale input
    
    step_size = config.step_size    # step size of integration
    solver = config.solver    # ODE solver
    
    # create log file
    logging.basicConfig(filename=model_dir+'/model_'+surf_type+'_'+data_name+'_'+surf_hemi+'_'+tag+'.log',
                        level=logging.INFO, format='%(asctime)s %(message)s')
    
    # --------------------------
    # load dataset
    # --------------------------
    logging.info("load dataset ...")
    trainset = load_surf_data(config, 'train')
    validset = load_surf_data(config, 'valid')

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False)

    # --------------------------
    # initialize models
    # --------------------------
    logging.info("initalize model ...")
    cortexode = CortexODE(dim_in=3, dim_h=C, kernel_size=K, n_scale=Q).to(device)
    optimizer = optim.Adam(cortexode.parameters(), lr=lr)
    T = torch.Tensor([0,1]).to(device)    # integration time interval for ODE

    # --------------------------
    # training
    # --------------------------
    logging.info("start training ...")
    for epoch in tqdm(range(n_epochs+1)):
        avg_loss = []
        for idx, data in enumerate(trainloader):
            volume_in, v_in, v_gt, f_in, f_gt = data

            optimizer.zero_grad()

            volume_in = volume_in.to(device).float()
            v_in = v_in.to(device)
            f_in = f_in.to(device)
            v_gt = v_gt.to(device)
            f_gt = f_gt.to(device)
            
            cortexode.set_data(v_in, volume_in)    # set the input data

            if surf_type == 'wm':    # training with randomly sampled points

                ### integration using seminorm (not recommended)
                # v_out = odeint(cortexode, v_in, t=T, method=solver,
                #                options=dict(step_size=step_size), adjoint_options=dict(norm='seminorm'))[-1]
                
                ### integration without seminorm
                v_out = odeint(cortexode, v_in, t=T, method=solver,
                               options=dict(step_size=step_size))[-1]
                
                mesh_out = Meshes(verts=v_out, faces=f_in)
                mesh_gt = Meshes(verts=v_gt, faces=f_gt)
                v_out = sample_points_from_meshes(mesh_out, n_samples)
                v_gt = sample_points_from_meshes(mesh_gt, n_samples)
                
                # scale by 1e3 since the coordinates are rescaled to [-1,1]
                loss = 1e3 * chamfer_distance(v_out, v_gt)[0]    # chamfer loss
                
            elif surf_type == 'gm':    # training with vertices
                v_out = odeint(cortexode, v_in, t=T, method=solver,
                               options=dict(step_size=step_size))[-1]
                loss = 1e3 * nn.MSELoss()(v_out, v_gt)

            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        logging.info('epoch:{}, loss:{}'.format(epoch, np.mean(avg_loss)))
        
        if epoch % 20 == 0:
            logging.info('-------------validation--------------')
            with torch.no_grad():
                valid_error = []
                for idx, data in enumerate(validloader):
                    volume_in, v_in, v_gt, f_in, f_gt = data

                    optimizer.zero_grad()

                    volume_in = volume_in.to(device).float()
                    v_in = v_in.to(device)
                    f_in = f_in.to(device)
                    v_gt = v_gt.to(device)
                    f_gt = f_gt.to(device)

                    cortexode.set_data(v_in, volume_in)

                    v_out = odeint(cortexode, v_in, t=T, method=solver,
                                   options=dict(step_size=step_size))[-1]
                    valid_error.append(1e3 * chamfer_distance(v_out, v_gt)[0].item())
                        
                logging.info('epoch:{}, validation error:{}'.format(epoch, np.mean(valid_error)))
                logging.info('-------------------------------------')

        # save model checkpoints 
        if epoch % 20 == 0:
            torch.save(cortexode.state_dict(), model_dir+'/model_'+surf_type+'_'+\
                       data_name+'_'+surf_hemi+'_'+tag+'_'+str(epoch)+'epochs.pt')

    # save the final model
    torch.save(cortexode.state_dict(), model_dir+'/model_'+surf_type+'_'+\
               data_name+'_'+surf_hemi+'_'+tag+'.pt')
    

if __name__ == '__main__':
    config = load_config()
    if config.train_type == 'surf':
        train_surf(config)
    elif config.train_type == 'seg':
        train_seg(config)
