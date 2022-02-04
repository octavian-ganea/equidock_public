import dgl.backend as F
import torch

def zerocopy_from_numpy(x):
    return torch.from_numpy(x)
    # return F.zerocopy_from_numpy(x)