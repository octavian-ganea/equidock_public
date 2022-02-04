import ot
import torch
import numpy as np

def compute_sq_dist_mat(X_1, X_2):
    '''Computes the l2 squared cost matrix between two point cloud inputs.
    Args:
        X_1: [n, #features] point cloud, tensor
        X_2: [m, #features] point cloud, tensor
    Output:
        [n, m] matrix of the l2 distance between point pairs
    '''
    n_1, _ = X_1.size()
    n_2, _ = X_2.size()
    X_1 = X_1.view(n_1, 1, -1)
    X_2 = X_2.view(1, n_2, -1)
    squared_dist = (X_1 - X_2) ** 2
    cost_mat = torch.sum(squared_dist, dim=2)
    return cost_mat


def compute_ot_emd(cost_mat, device):
    cost_mat_detach = cost_mat.detach().cpu().numpy()
    a = np.ones([cost_mat.shape[0]]) / cost_mat.shape[0]
    b = np.ones([cost_mat.shape[1]]) / cost_mat.shape[1]
    ot_mat = ot.emd(a=a, b=b, M=cost_mat_detach, numItermax=10000)
    ot_mat_attached = torch.tensor(ot_mat, device=device, requires_grad=False).float()
    ot_dist = torch.sum(ot_mat_attached * cost_mat)
    return ot_dist, ot_mat_attached
