import torch
import numpy as np
import einops

def parallel_scan(input_tensor, op):
    """
    Implements the Hillis-Steele parallel scan algorithm.
    
    Args:
    input_tensor: Input tensor of shape (batch_size, sequence_length, ...)
    op: A binary associative operator
    
    Returns:
    Output tensor of the same shape as input_tensor
    """
    batch_size, seq_len, *rest = input_tensor.shape
    out = input_tensor.clone()
    
    for d in range(seq_len.bit_length()):
        step = 2 ** d
        if step < seq_len:
            temp = torch.cat([out[:, step:], torch.zeros_like(out[:, :step])], dim=1)
            out = op(out, temp)
    
    return out

class BinaryOperation(torch.nn.Module):
    def __init__(self, Lambda, B):
        super(BinaryOperation,self).__init__()
        self.Lambda = Lambda
        self.B = B
    def forward(self, x:torch.Tensor, y:torch.Tensor):
        Ax = torch.einsum('bchwl,l->bchwl', x, self.Lambda)
        Bu = torch.einsum('kl,bchwl->bchwk',self.B,y) #matrix mul of B with the last dim of y
        output = Ax + Bu
        return output
    






