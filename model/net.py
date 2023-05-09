import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


 

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        self.deconv4 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.deconv3 = nn.Conv3d(192, 64, kernel_size=3, padding=1)
        self.deconv2 = nn.Conv3d(96, 32, kernel_size=3, padding=1)
        self.deconv1 = nn.Conv3d(48, 16, kernel_size=3, padding=1)
        self.lastconv1 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.lastconv2 = nn.Conv3d(16, out_channels, kernel_size=1)

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), 0.2)
        print("taille x1",x1.shape)
        x2 = F.leaky_relu(self.conv2(x1), 0.2)
        print("taille x2",x2.shape)
        x3 = F.leaky_relu(self.conv3(x2), 0.2)
        print("taille x3",x3.shape)
        x4 = F.leaky_relu(self.conv4(x3), 0.2)
        print("taille x4",x4.shape)
        x  = F.leaky_relu(self.conv5(x4), 0.2)
        print("taille x",x.shape)
        x  = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        print("taille x up ",x.shape)
        
        x = torch.cat([x, x4], dim=1)
        x = F.leaky_relu(self.deconv4(x), 0.2)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        
        x = torch.cat([x, x3], dim=1)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        
        x = torch.cat([x, x2], dim=1)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        
        x = torch.cat([x, x1], dim=1)
        x = F.leaky_relu(self.deconv1(x), 0.2)

        x = F.leaky




class CortexODE(nn.Module):
    """
    The deformation network of CortexODE model.

    dim_in: input dimension
    dim_h (C): hidden dimension
    kernel_size (K): size of convolutional kernels
    n_scale (Q): number of scales of the multi-scale input
    """
    
    def __init__(self, dim_in=3,
                       dim_h=128,
                       kernel_size=5,
                       n_scale=3):
        
        super(CortexODE, self).__init__()


        C = dim_h        # hidden dimension
        K = kernel_size  # kernel size
        Q = n_scale      # number of scales
        
        self.C = C
        self.K = K
        self.Q = Q

        # FC layers
        self.fc1 = nn.Linear(dim_in, C)
        self.fc2 = nn.Linear(C*2, C*4)
        self.fc3 = nn.Linear(C*4, C*2)
        self.fc4 = nn.Linear(C*2, dim_in)
        
        # local convolution
        self.localconv = nn.Conv3d(Q, C, (K, K, K))
        self.localfc = nn.Linear(C, C)
        
        # for cube sampling
        self.initialized = False
        grid = np.linspace(-K//2, K//2, K)
        grid_3d = np.stack(np.meshgrid(grid, grid, grid), axis=0).transpose(2,1,3,0)
        self.x_shift = torch.Tensor(grid_3d.copy()).view(-1,3)
        self.cubes = torch.zeros([1, self.Q, self.K, self.K, self.K])

    def _initialize(self, V):
        # initialize coordinates shift and cubes
        self.x_shift = self.x_shift.to(V.device)
        self.cubes = self.cubes.to(V.device)
        self.initialized == True
        
    def set_data(self, x, V):
        # x: coordinats
        # V: input brain MRI volume
        if not self.initialized:
            self._initialize(V)
            
        # set the shape of the volume
        D1,D2,D3 = V[0,0].shape
        D = max([D1,D2,D3])
        # rescale for grid sampling
        self.rescale = torch.Tensor([D3/D, D2/D, D1/D]).to(V.device)
        self.D = D

        self.m = x.shape[1]    # number of points
        self.neighbors = self.cubes.repeat(self.m,1,1,1,1)    # repeat m cubes
        
        # set multi-scale volume
        self.Vq = [V]
        for q in range(1, self.Q):
            # iteratively downsampling
            self.Vq.append(F.avg_pool3d(self.Vq[-1], 2))

    def forward(self, t, x):
        
        # local feature
        z_local = self.cube_sampling(x)
        z_local = self.localconv(z_local)
        z_local = z_local.view(-1, self.m, self.C)
        z_local = self.localfc(z_local)
        
        # point feature
        z_point = F.leaky_relu(self.fc1(x), 0.2)
        
        # feature fusion
        z = torch.cat([z_point, z_local], 2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        z = F.leaky_relu(self.fc3(z), 0.2)
        dx = self.fc4(z)
        
        return dx
    
    def cube_sampling(self, x):
        # x: coordinates
        with torch.no_grad():
            for q in range(self.Q):
                # make sure the cubes have the same size
                xq = x.unsqueeze(-2) + self.x_shift / self.D * 2 * (2**q)
                xq = xq.contiguous().view(1,-1,3).unsqueeze(-2).unsqueeze(-2)
                xq = xq / self.rescale  # rescale the coordinates
                # sample the q-th cube
                vq = F.grid_sample(self.Vq[q], xq, mode='bilinear', padding_mode='border', align_corners=True)
                # update the cubes
                self.neighbors[:,q] = vq[0,0].view(self.m, self.K, self.K, self.K)
        
        return self.neighbors.clone()