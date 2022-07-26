"""
Topology correction algorithm by Bazin et al.
Please cite the original papers if you use this code:
- Bazin et al. Topology correction using fast marching methods and its application to brain segmentation.
  MICCAI, 2005.
- Bazin et al. Topology correction of segmented medical images using a fast marching algorithm.
  Computer methods and programs in biomedicine, 2007.

The algorithm is re-implemented and accelerated using Python+Numba.

For the original Java implementation please see:
- https://github.com/piloubazin/cbstools-public/blob/master/de/mpg/cbs/core/shape/ShapeTopologyCorrection2.java
Or refer to the Nighres software:
- https://nighres.readthedocs.io/en/latest/shape/topology_correction.html
The look up table file "critical186LUT.raw.gz" is downloaded from Nighres:
- https://nighres.readthedocs.io/en/latest/
"""


from heapq import *
import numpy as np
from numba import njit
import gzip
from scipy.ndimage import binary_dilation


class topology():
    """
    apply topology correction algorithm
    inpt: input volume
    threshold: used to create initial mask. We set threshold=16 for CortexODE.
    """
    def __init__(self):
        bit, lut = tca_init_fill('./util/critical186LUT.raw.gz', threshold=1.0)
        self.bit = bit
        self.lut = lut
    def apply(self, inpt, threshold=1.0):
        mask, init_pts = tca_mask_fill(inpt, threshold)
        output = tca_fill(inpt, mask, init_pts, self.bit, self.lut)
        return output  # , mask


@njit
def bit_map():
    """used for compute key"""
    twobit = np.array([2**k for k in range(26)], dtype=np.float64)
    bit = np.zeros(27, dtype=np.float64)
    bit[:13] = twobit[:13]
    bit[14:] = twobit[13:]
    bit = bit.copy().reshape(3,3,3)
    
    return bit


@njit
def check_topology(img, LUT, bit):
    """check the critical points"""
    res = False
    if img[1,1,1] == 1:  # inside the original object
        res = True
    else:  # check topology
        # load key from pattern: should keep dtypes as the same
        key = 0.
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    key += img[i,j,k] * bit[i,j,k]
        key = int(key)
        if LUT[key]:        
            res = True
        else:
            res = False
        return res


""" tca_fill
This algorithm propagates from background to object.
It fills all holes and is used to fix WM segemntation. 
"""

def tca_mask_fill(levelset, threshold=1.0):
    """intialize processing mask"""
    
    initmask = np.zeros_like(levelset)
    initmask[2:-2, 2:-2, 2:-2] = 1
    initmask *= (levelset <= threshold)
    mask = binary_dilation(initmask, structure=np.ones([3,3,3]))
    init_pts = np.stack(np.where((mask-initmask)==1)).astype(int).T
    
    return mask, init_pts


def tca_init_fill(path, threshold=1.0):
    """
    Initialization for topology correction.
    Step 1. load look up table
    Step 2. load bit map
    Step 3. compile the Numba by a toy example
    """
    # load look up tables
    with gzip.open(path, 'rb') as lut_file:
        LUT = lut_file.read()
        
    # load bit map
    bit = bit_map()
    
    # create a toy example to compile the Numba
    img = (threshold-0.1) * np.ones([10,10,10])
    img[4:6,4:6,4:6] = threshold + 0.1
    mask, init_pts = tca_mask_fill(img, threshold)
    img_fix = tca_fill(img, mask, init_pts, bit, LUT)
    
    return bit, LUT


@njit
def tca_fill(levelset, mask, init_pts, bit, LUT):
    
    """Configuration"""
    minDistance = 1e-5
    UNKNOWN = 10e+10
    nx,ny,nz = levelset.shape
    # connectivity
    C6 = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    
    """Initialize indicators"""
    corrected = np.ones_like(levelset) * UNKNOWN  # gdm functions
    processed = np.zeros_like(levelset).astype(np.float64)
    inheap = np.zeros_like(levelset).astype(np.float64)
    mainval = 1e15
    maskval = -1e15
    maskval = np.max(mask*levelset)
    
    """add neighbors to the heap"""
    heap = []  # max heap
    for x0,y0,z0 in init_pts:
        processed[x0,y0,z0] = 1.
        corrected[x0,y0,z0] = levelset[x0,y0,z0]
        if corrected[x0,y0,z0] < mainval:
            mainval = corrected[x0,y0,z0]

        for dx, dy, dz in C6:
            xn = x0 + dx
            yn = y0 + dy
            zn = z0 + dz
            if mask[xn,yn,zn] and not processed[xn,yn,zn]:
                heap.append((-levelset[xn,yn,zn],(xn,yn,zn)))
                inheap[xn,yn,zn] = 1.
    heapify(heap)
    
    """Run Topology Correction"""
    while len(heap) > 0:

        # pop the heap
        val_, (x, y, z) = heappop(heap)
        val = - val_
        inheap[x, y, z] = 0.

        if processed[x, y, z]:
            continue
        cube = processed[x-1:x+2, y-1:y+2, z-1:z+2]
        non_critical = check_topology(cube, LUT, bit)
        
        if non_critical:
            # all correct: update and find new neighbors
            corrected[x,y,z] = val
            processed[x,y,z]= 1.  #  update the current level
            mainval = val

            # find new neighbors
            for dx, dy, dz in C6:
                xn = x + dx
                yn = y + dy
                zn = z + dz
                if mask[xn,yn,zn] and not processed[xn,yn,zn] and not inheap[xn,yn,zn]:
                    heappush(heap, (-min(levelset[xn,yn,zn], val-minDistance), (xn, yn, zn)))
                    inheap[xn,yn,zn] = True
                    
    corrected += (mainval-corrected) * (1-processed)
    corrected += (maskval-corrected) * (1-mask)
    
    return corrected



""" tca_cut (to be validated)
This algorithm propagates from object to background.
It cuts all handles and is used to fix GM segemntation. 

Note: this function is not fully validated because we only use tca_fill for CortexODE.
"""

def tca_mask_cut(levelset, threshold=1.0):
    """intialize processing mask"""
    initmask = np.zeros_like(levelset)
    initmask[2:-2, 2:-2, 2:-2] = 1
    initmask *= (levelset <= threshold)
    mask = binary_dilation(initmask, structure=np.ones([3,3,3]))
    init_pts = np.stack(np.where(levelset==np.min(levelset))).astype(int).T
    return mask, init_pts


def tca_init_cut(path, threshold=1.0):
    """
    Initialization for topology correction.
    Step 1. load look up table
    Step 2. load bit map
    Step 3. compile the Numba by a toy example
    """
    # load look up tables
    with gzip.open(path, 'rb') as lut_file:
        LUT = lut_file.read()
        
    # load bit map
    bit = bit_map()
    
    # create a toy example to compile the Numba
    img = (threshold+0.1) * np.ones([10,10,10])
    img[4:6,4:6,4:6] = threshold - 0.1
    img[5,5,5] = threshold - 0.2
    mask, init_pts = tca_mask_cut(img, threshold)
    img_fix = tca_cut(img, mask, init_pts, bit, LUT)
    
    return bit, LUT



@njit
def tca_cut(levelset, mask, init_pts, bit, LUT):
    
    """Configuration"""
    minDistance = 1e-5
    UNKNOWN = 10e+10
    nx,ny,nz = levelset.shape
    # connectivity
    C6 = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    
    """Initialize indicators"""
    corrected = np.ones_like(levelset) * UNKNOWN  # gdm functions
    processed = np.zeros_like(levelset).astype(np.float64)
    inheap = np.zeros_like(levelset).astype(np.float64)
    mainval = -1e15
    maskval = -1e15
    maskval = np.max(mask*levelset)
    
    """add neighbors to the heap"""
    heap = []  # max heap
    for x0,y0,z0 in init_pts:
        processed[x0,y0,z0] = 1.
        corrected[x0,y0,z0] = levelset[x0,y0,z0]
        if corrected[x0,y0,z0] > mainval:
            mainval = corrected[x0,y0,z0]

        for dx, dy, dz in C6:
            xn = x0 + dx
            yn = y0 + dy
            zn = z0 + dz
            if mask[xn,yn,zn] and not processed[xn,yn,zn]:
                heap.append((levelset[xn,yn,zn],(xn,yn,zn)))
                inheap[xn,yn,zn] = 1.
    heapify(heap)
    

    """Run Topology Correction"""
    while len(heap) > 0:

        # pop the heap
        val_, (x, y, z) = heappop(heap)
        val = val_
        inheap[x, y, z] = 0.

        if processed[x, y, z]:
            continue
        cube = processed[x-1:x+2, y-1:y+2, z-1:z+2]
        non_critical = check_topology(cube, LUT, bit)
        
        if non_critical:
            # all correct: update and find new neighbors
            corrected[x,y,z] = val
            processed[x,y,z]= 1.  #  update the current level
            mainval = val

            # find new neighbors
            for dx, dy, dz in C6:
                xn = x + dx
                yn = y + dy
                zn = z + dz
                if mask[xn,yn,zn] and not processed[xn,yn,zn] and not inheap[xn,yn,zn]:
                    heappush(heap, (max(levelset[xn,yn,zn], val-minDistance), (xn, yn, zn)))
                    inheap[xn,yn,zn] = True
                    
    corrected += (mainval-corrected) * (1-processed)
    corrected += (maskval-corrected) * (1-mask)
    
    return corrected