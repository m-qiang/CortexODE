import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.mesh import laplacian_smooth


# segmentation U-Net
class Unet(nn.Module):
    def __init__(self, c_in=1, c_out=2):
        super(Unet, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=c_in, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3,
                               stride=2, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3,
                               stride=2, padding=1)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3,
                               stride=2, padding=1)
        self.conv5 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3,
                               stride=2, padding=1)

        self.deconv4 = nn.Conv3d(in_channels=256, out_channels=64, kernel_size=3,
                               stride=1, padding=1)
        self.deconv3 = nn.Conv3d(in_channels=128, out_channels=32, kernel_size=3,
                               stride=1, padding=1)
        self.deconv2 = nn.Conv3d(in_channels=64, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        self.deconv1 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        
        self.lastconv1 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3,
                                   stride=1, padding=1)
        self.lastconv2 = nn.Conv3d(in_channels=16, out_channels=c_out, kernel_size=3,
                                   stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear')
        
    def forward(self, x):

        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.conv2(x1), 0.2)
        x3 = F.leaky_relu(self.conv3(x2), 0.2)
        x4 = F.leaky_relu(self.conv4(x3), 0.2)
        x  = F.leaky_relu(self.conv5(x4), 0.2)
        x  = self.up(x)
        
        x = torch.cat([x, x4], dim=1)
        x = F.leaky_relu(self.deconv4(x), 0.2)
        x = self.up(x)
        
        x = torch.cat([x, x3], dim=1)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        x = self.up(x)
        
        x = torch.cat([x, x2], dim=1)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        x = self.up(x)
        
        x = torch.cat([x, x1], dim=1)
        x = F.leaky_relu(self.deconv1(x), 0.2)

        x = F.leaky_relu(self.lastconv1(x), 0.2)
        x = self.lastconv2(x)

        return x


class CortexODE(nn.Module):
    def __init__(self, dim_in=3,
                       dim_h=128,
                       kernel_size=5,
                       n_scale=3):
        
        super(CortexODE, self).__init__()
        """
        dim_in (3): input dimension
        dim_h (C): hidden dimension
        kernel_size (K): size of convolutional kernels
        n_scale (Q): number of scales of the multi-scale input
        """

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

        
    def _initialize(self, V):
        # intialize rescale
        D1,D2,D3 = V[0,0].shape
        D = max([D1,D2,D3])
        self.rescale = torch.Tensor([D3/D, D2/D, D1/D]).to(V.device)
        self.D = D
        # initialize coordinates shift
        self.x_shift = self.x_shift.to(V.device)
        self.initialized == True

        
    def forward(self, x, V, solver='euler', step_size=0.2, T=1.0):
        if not self.initialized:
            self._initialize(V)
            
        h = step_size
        N = int(T/h)

        if solver == 'euler':
            # forward Euler method
            for n in range(N):
                dx = self.deform(x, V)
                x = x + h * dx
        
        if solver == 'midpoint':
            # midpoint method
            for n in range(N):
                dx1 = self.deform(x, V)
                dx2 = self.deform(x + h*dx1/2, V)
                x = x + h * dx2
                
        if solver == 'heun':
            # Heun's method
            for n in range(N):
                dx1 = self.deform(x, V)
                dx2 = self.deform(x + h*dx1, V)
                x = x + h * (dx1 + dx2) / 2
                
        if solver == 'rk4':
            # fourth-order RK method
            for n in range(N):
                dx1 = self.deform(x, V)
                dx2 = self.deform(x + h*dx1/2, V)
                dx3 = self.deform(x + h*dx2/2, V)
                dx4 = self.deform(x + h*dx3, V)
                x = x + h * (dx1 + 2*dx2 + 2*dx3 + dx4) / 6
                
        return x


    def deform(self, x, V):
        m = x.shape[1]
        
        # local feature
        z_local = self.cube_sampling(x, V)
        z_local = self.localconv(z_local)
        z_local = z_local.view(-1, m, self.C)
        z_local = self.localfc(z_local)
        
        # point feature
        z_point = F.leaky_relu(self.fc1(x), 0.2)
        
        # feature fusion
        z = torch.cat([z_point, z_local], 2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        z = F.leaky_relu(self.fc3(z), 0.2)
        dx = self.fc4(z)
        
        return dx
    
    
    def cube_sampling(self, x, V):
        """
        x: coordinates
        V: volumetric input
        """
        with torch.no_grad():
            m = x.shape[1]
            
            # initialize all cubes (m,Q,K,K,K)
            self.v = torch.zeros([m, self.Q, 
                                  self.K, self.K, self.K]).to(V.device)
            Vq = V    # set the initial scale
            for q in range(self.Q):
                if q >= 1:
                    Vq = F.avg_pool3d(Vq, 2)  # downsampling

                # make sure the cubes have the same size
                xq = x.unsqueeze(-2) + self.x_shift / self.D * 2 * (2**q)
                xq = xq.contiguous().view(1,-1,3).unsqueeze(-2).unsqueeze(-2)
                xq = xq / self.rescale  # normalize to [-1,1]
                # sample the q-th cube
                vq = F.grid_sample(Vq, xq, mode='bilinear', padding_mode='border', align_corners=True)
                self.v[:,q] = vq[0,0].view(m, self.K, self.K, self.K)
        
        return self.v
