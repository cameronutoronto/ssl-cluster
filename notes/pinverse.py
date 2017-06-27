#!/usr/bin/python
"""
difference between proper and inproper inverses
"""
import os
import torch
import numpy


def pinverse(X):
    u, s, v = torch.svd(X)
    h = torch.max(s) * float(max(X.size(0),X.size(1))) * 1e-15
    indices = torch.ge(s,h)
    indices2 = indices.eq(0)
    s[indices] = 1.0 / s[indices]
    s[indices2] = 0
    return torch.mm(torch.mm(v, torch.diag(s)), u.t())

def pinverse2(X):
    u, s, v = torch.svd(X)
    return torch.mm(torch.mm(v, torch.diag(1.0 / (s+0.000001))), u.t())


x = torch.FloatTensor([[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
print(x)
print(pinverse(x))
print(pinverse2(x))