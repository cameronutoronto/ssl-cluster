import torch


class Euclidean(object):
    def phi(self, X):
        return torch.pow(X, 2).sum(1)
    
    def grad_phi(self, X):
        return 2 * X