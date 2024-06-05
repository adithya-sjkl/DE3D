# DE3D
Here we attempt to create a data efficient 3D image classification model using deep learning.

This is to test git.

DE3D:
self.c 
self.h 

self.feat_axial = feat2D_convlstm(channels,batch,self.c,self.h)

-----------

feat2D_convlstm:
__inti__(channels,batch,c,h):
self.c = c
self.h = h

forward(x):
c = self.c
h = self.h

loop:
 c, h = self.convlstm(slice, (self.c,self.h))