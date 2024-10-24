import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import timm
from torch.functional import Tensor
import typing
import einops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feat_ext = timm.create_model(
    'tf_efficientnet_b0.ns_jft_in1k',
    #'resnet18',
    pretrained=True,
    features_only=True,
    #out_indices = [1,2,3,4],
    in_chans = 1
)

class features_2D(nn.Module):
    def __init__ (self, channels, batch_size):
        super(features_2D, self).__init__()
        self.feat_ext = feat_ext
        self.channels = channels
        self.batch_size = batch_size
        self.channel_red = nn.Conv2d(in_channels=1280,out_channels=channels,kernel_size=1)
        
    def forward(self, x:Tensor):
        x = einops.rearrange(x, 'b c h w l -> (b h) c w l')   # we concat the images in the batch along the height
        x = self.feat_ext(x)                           #we choose the layer from which we extract features
        x = self.channel_red(x)                           #1x1 convs help reduce the number of channels by mixing
        x = einops.rearrange(x, '(b h) c w l -> b h c w l', b=self.batch_size) #reverse the first step

        return x
    
#print('EffB0 trainable parameters in millions:',sum(p.numel() for p in feat_ext.parameters() if p.requires_grad)/1000000)

#print([y.shape for y in feat_ext(x)])

class ConvLSTMCell(nn.Module):
    def __init__(self, intermediate_channels:int, kernel_size:int=3):
        super(ConvLSTMCell, self).__init__()
        """conv_x  has a valid padding by:
        setting padding = kernel_size // 2
        hidden channels for h & c = intermediate_channels
        """
        self.intermediate_channels = intermediate_channels
        self.conv_x = nn.Conv2d(
            2 * intermediate_channels, intermediate_channels *  4,
            kernel_size=kernel_size, padding= kernel_size // 2, bias=True
        )
    def forward(self, x:Tensor, state:typing.Tuple[Tensor, Tensor]) -> typing.Tuple:
        """
        c and h channels = intermediate_channels so  a * c is valid
        if the last dim in c not equal to a then a has been halved
        """
        c, h = state
        h = h.to(device=x.device)
        c = c.to(device=x.device)
        x = torch.cat([x, h], dim=1)
        x = self.conv_x(x)
        a, b, g, d = torch.split(x, self.intermediate_channels, dim=1)
        a = torch.sigmoid(a)
        b = torch.sigmoid(b)
        g = torch.sigmoid(g)
        d = torch.tanh(d)
        c =  a * c +  g * d
        h = b * torch.tanh(c)
        return c, h




class DE3D(nn.Module):
    def __init__(self, channels:int, batch_size:int, feat_res:int=28):
        super(DE3D, self).__init__()
        self.channels = channels
        self.batch_size = batch_size
        self.c = torch.zeros(batch_size,channels,feat_res,feat_res)
        self.h = torch.zeros(batch_size,channels,feat_res,feat_res)
        self.features_2D = features_2D(channels=channels,batch_size=batch_size)
        self.convlstm = ConvLSTMCell(intermediate_channels=channels)
        self.fconv1 = nn.Conv2d(in_channels=channels,out_channels=4,kernel_size=5)
        self.fconv2 = nn.Conv2d(in_channels=4,out_channels=1,kernel_size=5)
        self.fc1 = nn.Linear(in_features=400,out_features=50)
        self.fc2 = nn.Linear(in_features=50,out_features=2)

    def forward(self, x:Tensor):
        x = self.features_2D(x)
        c = self.c
        h = self.h
        _,H,_,_,_ = x.shape
        x = einops.rearrange(x, 'b h c w l -> h b c w l')
        
        #the convlstm goes through the slices along the height for each batch simultaneously 
        for slice_num in range(H):
            slice = x[slice_num,:,:,:,:]
            c,h = self.convlstm(slice,[c,h])
        x = self.fconv1(h)
        x = self.fconv2(x)
        x = einops.rearrange(x,'b c w l -> b (c w l)')
        x = self.fc1(x)
        x = self.fc2(x)
        return x

'''
class DE3D(nn.Module):
    def __init__(self, channels:int, batch_size:int, feat_res:int=28):
        super(DE3D, self).__init__()
        #self.channels = channels
        #self.batch_size = batch_size
        self.fully_connected = nn.Linear(in_features=1200,out_features=2)
        self.c = torch.zeros(batch_size,channels,feat_res,feat_res)
        self.h = torch.zeros(batch_size,channels,feat_res,feat_res)
        self.features = feat_2D_convlstm(channels=channels,batch_size=batch_size,c=self.c,h=self.h)

    def forward(self, x:Tensor):

        x_1 = x
        x_2 = x_1.clone()
        x_3 = x_1.clone()

        x_2 = einops.rearrange(x_2, 'b c h w l -> b c w h l')
        x_3 = einops.rearrange(x_3, 'b c h w l -> b c l w h')

        out_axial = self.features(x_1)
        out_saggital = self.features(x_2)
        out_coronal = self.features(x_3)

        out = torch.cat([out_axial,out_saggital,out_coronal],dim=1)
        out = self.fully_connected(out)

        return out
'''






