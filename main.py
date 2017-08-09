from __future__ import division
import torch
import numpy as np
from cluster_utils import inverse
def inv_H(H_prev, p_norm=1):
    H_prev_inv = H_prev.t()
    d = H_prev_inv.pow(p_norm).sum(1).expand_as(H_prev_inv)
    # d = H_prev_inv.pow(p_norm).sum(1).pow(1/p_norm).expand_as(H_prev_inv)
    d[d==0] = 1
    H_prev_inv = H_prev_inv / d
    return H_prev_inv

def update_H(H_prev, U, dist, Y=None, episilon=1, p_norm =1):
    H = np.zeros(H_prev.size())
    H_prev_inv = inv_H(H_prev, p_norm)
    Z = torch.mm(H_prev_inv, torch.mm(torch.diag(H_prev.sum(1)[...,0]),U))
    for example_idx in xrange(U.size()[0]):
        curr_u = U[example_idx]
        curr_u = curr_u[None].repeat(Z.size()[0],1)
        curr_h_ind = dist.phi(curr_u - Z).numpy().argmin()
        H_updated = False
        if Y is not None:
            curr_y = Y[example_idx]
            if curr_y.sum() == 1: #labelled-example
                curr_h_ind = curr_y.argmax()
                H[example_idx, curr_h_ind] = 1
            else:
                H[example_idx, curr_h_ind] = episilon
            H_updated = True
        if not H_updated:
            H[example_idx, curr_h_ind] = 1
    return torch.FloatTensor(H), Z


def solve_H(H_init, U ,dist,Y=None,iters=100, thresh=0.001, return_Z=False,episilon=1, p_norm=1):
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
        H,Z = update_H(H, U, dist, Y, episilon, p_norm)
        if i > 0 and dist.phi(Z-Z_prev).numpy().sum() < thresh:
            break
        Z_prev = Z
    if not return_Z:
        return H
    else:
        return H,Z

def grad_F(F, H):
    # return 2*torch.mm(torch.mm(torch.mm((torch.mm(F,inverse(F)) - torch.eye(F.size()[0])),H),inverse(H)),inverse(F).t())
    inv_F = inverse(F)
    return torch.mm(torch.mm(F,torch.mm(inv_F,H)) - H,torch.mm(inv_H(H),inv_F.t()))

def grad_F_episilon(F, H):
    inv_F = inverse(F)
    eyeD = torch.eye(F.size()[0])
    B = torch.diag(H.sum(1)[...,0])
    inv_B = torch.diag(1./H.sum(1)[...,0])
    vinv_H = inv_H(H, p_norm=2)
    return torch.mm(
            torch.mm(
                    torch.mm(F, inv_F) - eyeD,
                    torch.mm(B,torch.mm(H,torch.mm(vinv_H, inv_B)))
                    +torch.mm(inv_B,torch.mm(H,torch.mm(vinv_H, B)))),
                      inv_F.t())
if __name__ == '__main__':
    n = 100
    d = 20
    k = 10
    F = torch.rand(n, d)
    H = torch.rand(n, k)
    grad_F_episilon(F, H)