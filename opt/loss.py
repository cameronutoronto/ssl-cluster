import torch
from torch.autograd import Variable
import numpy as np
from main import solve_H, grad_F, inv_H
from cluster_utils import inverse


class SSLCluster(object):
    """docstring for SSLCluster"""
    def __init__(self, dist, support_X, support_Y):
        super(SSLCluster, self).__init__()
        self.dist = dist
        self.support_X = support_X
        self.support_Y = support_Y

    def train(self, F, Y_batch):
        u_r, s, v = torch.svd(F)  
        H_init = np.array(Y_batch.numpy(), copy=True)
        H_init[H_init.sum(1) > 1] *= 0
        H = solve_H(torch.FloatTensor(H_init), u_r,self.dist,Y=Y_batch.numpy(),iters=10)
        ##
        loss = torch.dot(torch.mm(inverse(H),F),torch.mm(inverse(F), H).t())
        G = grad_F(F,H)
        train_pred = H.numpy().argmax(1)
        return loss, G, train_pred

    def infer(self, model, X_val, N_support):
        inds = np.arange(len(self.support_X))
        np.random.shuffle(inds)
        inds = torch.cuda.LongTensor(inds[:N_support])
        val_pred, non_spectral_val_pred = get_pred_join_SVD(model, self.support_X[inds], self.support_Y[inds], X_val, get_non_spectral=True)
        val_pred_norm,_ = get_pred_join_SVD(model, self.support_X[inds], self.support_Y[inds], X_val, normalize=True)
        return val_pred, non_spectral_val_pred, val_pred_norm

class CE(object):
    """docstring for CE
    standard classification loss
    """
    def __init__(self):
        super(CE, self).__init__()
        self.lsoftmax = torch.nn.LogSoftmax()
        self.nll = torch.nn.NLLLoss()
    def train(self, F, Y_batch):
        tv_F = Variable(F, requires_grad=True)
        tv_Y = Variable(torch.LongTensor(Y_batch.numpy().argmax(1)))
        py_x = self.lsoftmax(tv_F)
        loss = self.nll(py_x, tv_Y)
        ##
        loss.backward()
        G = tv_F.grad.data
        train_pred = py_x.data.numpy().argmax(1)
        return loss.data[0], G, train_pred

    def infer(self, model, X_val, N_support):
        py_x = self.lsoftmax(model.forward(X_val))
        val_pred = py_x.data.cpu().numpy().argmax(1)
        return val_pred, val_pred, val_pred
        
# ##TODO: replace C_hat with F for faster computation
# ##Question on my mind: how much does unlabelled data actually help (how much do we need here)
def _normalize(X):
    return X / torch.sqrt(torch.pow(X,2).sum(1)).expand_as(X)
def _center_normalize(X):
    X = X - X.mean(0).expand_as(X)
    return _normalize(X)


def solve_kmeans_to_predict(support_U, support_Y, val_U):
    support_Z = torch.mm(inv_H(support_Y), support_U)

    #compute distance
    Z_aug = support_Z[None].repeat(val_U.size()[0], 1,1)
    U_aug = val_U[:,None].repeat(1,support_Z.size()[0],1)
    val_pred = torch.pow(Z_aug- U_aug,2).cpu().numpy().sum(-1).argmin(-1)
    return val_pred

def get_pred_join_SVD(model, support_X, support_Y, X_val, normalize=False, get_non_spectral=False):
    """
    concatenate support_X with X_val,
    perform SVD, using support_X to compute centroid in U, 
    do assignment for X_val
    """
    joint_X = torch.cat((support_X, X_val), 0)
    if normalize:
        joint_X = _center_normalize(joint_X)

    joint_F = model.forward(joint_X).data
    joint_U, _, _ = torch.svd(torch.mm(joint_F, inverse(joint_F)))
    joint_U = joint_U[:,:10]

    support_U = joint_U[:support_X.size()[0]]
    val_U = joint_U[support_X.size()[0]:]
    
    val_pred = solve_kmeans_to_predict(support_U, support_Y, val_U)
    if get_non_spectral:
        non_spectral_val_pred = solve_kmeans_to_predict(_normalize(joint_U[:support_X.size()[0],:10]), support_Y, _normalize(joint_U[support_X.size()[0]:,:10]))
    else:
        non_spectral_val_pred = None

    return val_pred, non_spectral_val_pred