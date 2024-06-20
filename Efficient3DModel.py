import torch
import transformers
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch import optim
import timm
from torchsummary import summary
from transformers import AutoModel, AutoTokenizer
from torch.functional import Tensor
import typing
import einops

x = torch.randn(5,1,224,224)

feat_ext = timm.create_model(
    'tf_efficientnet_b0.ns_jft_in1k',
    #'resnet18',
    pretrained=True,
    features_only=True,
    #out_indices = [1,2,3,4],
    in_chans = 1
)

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
        #h = h.to(device=x.device)
        #c = c.to(device=x.device)
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

convLSTM = ConvLSTMCell(16)


# Hyperparameters
int_channels = 10
batch_size = 1

class features_2D(nn.Module):
    def __init__ (self, channels, batch_size):
        super(features_2D, self).__init__()
        self.feat_ext = feat_ext
        self.channels = channels
        self.batch_size = batch_size
        self.channel_red = nn.Conv2d(in_channels=40,out_channels=channels,kernel_size=1)
        
    def forward(self, x:Tensor):
        x = einops.rearrange(x, 'b h w l -> (b h) w l')   # we concat the images in the batch along the height
        x = torch.unsqueeze(x, dim=1)                     #efficientnet (grayscale) needs a channel dimension of size 1
        x = self.feat_ext(x)[2]                           #we choose the layer from which we extract features
        x = self.channel_red(x)                           #1x1 convs help reduce the number of channels by mixing
        x = einops.rearrange(x, '(b h) c w l -> b h c w l', b=self.batch_size) #reverse the first step

        return x


class feat_2D_convlstm(nn.Module):
    def __init__(self, channels:int, batch_size:int, c:Tensor, h:Tensor):
        super(feat_2D_convlstm, self).__init__()
        self.channels = channels
        self.batch_size = batch_size
        self.c = c
        self.h = h
        self.features_2D = features_2D(channels=channels,batch_size=batch_size)
        self.convlstm = ConvLSTMCell(intermediate_channels=int_channels)
        self.fconv1 = nn.Conv2d(in_channels=channels,out_channels=4,kernel_size=5)
        self.fconv2 = nn.Conv2d(in_channels=4,out_channels=1,kernel_size=5)

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
        return x


class DE3D(nn.Module):
    def __init__(self, channels:int, batch_size:int, feat_res):
        super(DE3D, self).__init__()
        #self.channels = channels
        #self.batch_size = batch_size
        self.fully_connected = nn.Linear(in_features=1200,out_features=1)
        self.c = torch.zeros(batch_size,channels,feat_res,feat_res)
        self.h = torch.zeros(batch_size,channels,feat_res,feat_res)
        self.features = feat_2D_convlstm(channels=int_channels,batch_size=batch_size,c=self.c,h=self.h)

    def forward(self, x:Tensor):

        x_1 = x
        x_2 = x_1.clone()
        x_3 = x_1.clone()

        x_2 = einops.rearrange(x_2, 'b h w l -> b w h l')
        x_3 = einops.rearrange(x_3, 'b h w l -> b l w h')

        out_axial = self.features(x_1)
        out_saggital = self.features(x_2)
        out_coronal = self.features(x_3)

        out = torch.cat([out_axial,out_saggital,out_coronal],dim=1)
        out = self.fully_connected(out)




#image_tensor = torch.randn(2,224,224,224)
de3d = DE3D(channels=int_channels,batch_size=batch_size,feat_res=28)



#output = de3d(image_tensor)

#print(summary(de3d,(224,224,224)))



