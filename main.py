import torch
import numpy as np
from cluster_utils import inverse
def inv_H(H_prev):
    H_prev_inv = H_prev.t()
    d = H_prev_inv.sum(1).expand_as(H_prev_inv)
    d[d==0] = 1
    H_prev_inv = H_prev_inv / d
    return H_prev_inv

def update_H(H_prev, U, dist, Y=None):
    H = np.zeros(H_prev.size())
    H_prev_inv = inv_H(H_prev)
    Z = torch.mm(H_prev_inv, U)
    for example_idx in xrange(U.size()[0]):
        curr_u = U[example_idx]
        curr_u = curr_u[None].repeat(Z.size()[0],1)
        curr_h_ind = dist.phi(curr_u - Z).numpy().argmin()
        if Y is not None:
            curr_y = Y[example_idx]
            if curr_y.sum() == 1: #labelled-example
                curr_h_ind = curr_y.argmax()
        H[example_idx, curr_h_ind] = 1
    return torch.FloatTensor(H), Z


def solve_H(H_init, U ,dist,Y=None,iters=100, thresh=0.001):
    H = H_init
    # make sure initial H satisfies Y
    for example_idx in xrange(U.size()[0]):
        if Y is not None:
            curr_y = Y[example_idx]
            if curr_y.sum() == 1: #labelled-example
                H[example_idx] *=0
                H[example_idx, curr_y.argmax()] = 1
    ##
    for i in xrange(iters):
        H,Z = update_H(H, U, dist, Y)
        if i > 0 and dist.phi(Z-Z_prev).numpy().sum() < thresh:
            break
        Z_prev = Z
    return H

def grad_F(F, H):
    # return 2*torch.mm(torch.mm(torch.mm((torch.mm(F,inverse(F)) - torch.eye(F.size()[0])),H),inverse(H)),inverse(F).t())
    inv_F = inverse(F)
    return torch.mm(torch.mm(F,torch.mm(inv_F,H)) - H,torch.mm(inv_H(H),inv_F.t()))
