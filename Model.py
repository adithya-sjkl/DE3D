import torch
import numpy as np
import einops
import torch.nn as nn
import timm
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parallel_scan(arr:torch.Tensor, op):
    """
    Implements the Hillis-Steele parallel scan algorithm.
    
    Args:
    arr: Input array
    op: Binary associative operator (e.g., np.add for sum, np.maximum for max)
    
    Returns:
    Scan result array
    """
    n = arr.shape[-2]
    output = torch.clone(arr)
    
    # Simulate log2(n) parallel steps
    for d in range(1, int(np.log2(n)) + 1):
        step = 2 ** (d - 1)
        
        # Simulate parallel operations
        temp = torch.clone(output)

        output[:,:,:,:,step:,:] = op(temp[:,:,:,:,:n-step,:], temp[:,:,:,:,step:,:])
    
    return output

class BinaryOperation(torch.nn.Module):
    def __init__(self, Lambda, B):
        super(BinaryOperation,self).__init__()
        self.Lambda = Lambda
        self.B = B
    def forward(self, x:torch.Tensor, y:torch.Tensor):
        Ax = torch.einsum('bchwnl,l->bchwnl', x, self.Lambda)
        Bu = torch.einsum('kl,bchwnl->bchwnk',self.B,y) #matrix mul of B with the last dim of y
        output = Ax + Bu
        return output
    


class ImageLRU(torch.nn.Module):
    def __init__(self, num_slices:int, rmin:float, rmax:float):
        super(ImageLRU, self).__init__()
        self.num_slices = num_slices
        self.rmin = rmin
        self.rmax = rmax
        self.u1 = torch.rand(num_slices).to(device)
        self.u2 = torch.rand(num_slices).to(device)
        self.theta = nn.Parameter(2*np.pi*self.u2)
        self.nu_log = nn.Parameter(torch.log(-0.5*torch.log(self.u1*(self.rmax**2-self.rmin**2)+self.rmin**2)))
        self.Lambda = nn.Parameter(torch.complex(torch.exp(-torch.exp(self.nu_log)) , self.theta))
        self.B = nn.Parameter(torch.randn(self.num_slices, self.num_slices, dtype=torch.complex64))
        self.C = nn.Parameter(torch.randn(self.num_slices, self.num_slices, dtype=torch.complex64))
        self.D = nn.Parameter(torch.randn(self.num_slices, self.num_slices))
        self.op = BinaryOperation(self.Lambda, self.B)

    def forward(self, x:torch.Tensor):
        img_array = einops.rearrange(x,'b c h w (n l) -> b c h w n l',l=self.num_slices)
        complex_array = img_array.to(torch.complex64)
        out = parallel_scan(complex_array, self.op)
        out = torch.real(torch.einsum('kl,bchwnl->bchwnk',self.C,out)) + torch.einsum('kl,bchwnl->bchwnk',self.D,img_array)
        return out


feat_ext = timm.create_model(
    'tf_efficientnet_b4.ns_jft_in1k',
    #'resnet18',
    pretrained=True,
    in_chans = 1
)

class features_2D(nn.Module):
    def __init__ (self, batch_size):
        super(features_2D, self).__init__()
        self.feat_ext = feat_ext
        self.batch_size = batch_size
        
    def forward(self, x:torch.Tensor):
        x = einops.rearrange(x, 'b c h w l -> (b h) c w l')   # we concat the images in the batch along the height
        x = self.feat_ext(x)                           #we choose the layer from which we extract features
        x = einops.rearrange(x, '(b h) o -> b (h o)', b=self.batch_size) #reverse the first step

        return x


class E3D(nn.Module):
    def __init__(self,batch_size,num_slices:int, rmin:float, rmax:float, dropout:float=0.5):
        super(E3D,self).__init__()
        self.imagelru = ImageLRU(num_slices=num_slices, rmin=rmin, rmax=rmax)
        self.features2d = features_2D(batch_size=batch_size)
        self.fc1 = nn.Linear(in_features=7000,out_features=500)
        self.fc2 = nn.Linear(in_features=500,out_features=2)
        #self.fc3 = nn.Linear(in_features=200,out_features=2)
        self.do = nn.Dropout(dropout, inplace=False)
        self.relu = nn.ReLU()
    def forward(self,x:torch.Tensor):
        x = self.imagelru(x)
        x = einops.rearrange(x, 'b c h w n l -> b c h w (n l)')
        x = self.imagelru(x)
        x = x[:,:,:,:,-1,:]
        x = einops.rearrange(x, 'b c h w l -> b c l h w')
        x = self.features2d(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.fc2(x)
        #x = self.relu(x)
        #x = self.do(x)
        #x = self.fc3(x)

        return x