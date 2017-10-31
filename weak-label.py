from __future__ import division
from docopt import docopt
import yaml
import torch
import tensorflow as tf
from torch import optim
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,10)
from torch.autograd import Variable
# from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import numpy as np

from metric import Euclidean
# from cleverhans.utils_mnist import data_mnist
from utils import MiniBatcher, MiniBatcherPerClass

import datetime
from opt.loss import *
from model.fc import fc
from model.cnn import *
from model.gen import *
from ssl_utils import ssl_basic, ssl_per_class
import cPickle as pkl
# from multiprocessing import Pool

#### 
def _compute_log_inception_score(py_x):
    """
    py_x (B, D)
    return kls (B, )
    """
    part = py_x
    kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
    return np.sum(kl, 1)

def main(X, data_t7_path):
    rdata_t7_path = 'gen-data/wgan-real-normalized.t7'
    # real-net config
    f_model_config = 'model/config/cnn-globe-large.yaml'
    real_net_path = 'saves/CE.wgan-fixed:cnn-globe-large-X-sl-128-lr-X-real-normalized@2017-08-10/model_best.t7'

    ####
    #gen data
    _,_,X_val,Y_val = torch.load(rdata_t7_path)
    #real-net
    model_config = yaml.load(open(f_model_config, 'rb'))
    real_net = eval(model_config['name'])(**model_config['kwargs'])
    real_net.load(real_net_path)
    real_net.type(torch.cuda.FloatTensor)
       
    softmax = torch.nn.Softmax()

    batch_size=100
    weak_labels = []
    inception_scores = []
    # for idx in tqdm(xrange(1)):
    for idx in tqdm(xrange(X.size()[0]//batch_size)):
        idxs = np.arange(idx*batch_size, (idx+1)*batch_size)
        X_batch = X[torch.LongTensor(idxs)].type(torch.cuda.FloatTensor)
        ## network
        py_x = softmax(real_net.forward(X_batch))
        py_x = py_x.data.cpu().numpy()
        lincep_scores = _compute_log_inception_score(py_x)
        inception_scores += list(lincep_scores)
        hardmax = np.zeros(py_x.shape)
        hardmax[np.arange(len(hardmax)), py_x.argmax(1)] = 1 
        weak_labels.append(hardmax)
    torch.save((X, torch.FloatTensor(np.concatenate(weak_labels, 0)), X_val, Y_val), data_t7_path)
    # print( data_t7_path)
    # ### Inception Score is the mean of the per-example score
    # ## so we need to first do that
    # ## then compute mean, std of that score 
    # inception_scores = np.exp(np.array(inception_scores).reshape(10000,-1).mean(0))
    # print('[Inception Score]: %g, %g'%(np.mean(inception_scores), np.std(inception_scores)))

if __name__ == '__main__':
    # gen samples
    # data_t7_path = 'gen-data/real.t7'


    # data_t7_path = 'gen-data/dcgan-M-%i.t7'
    # iters = [1000,10000,100000,199000]
    # for it in iters:
    #     X,_,_,_ = torch.load(data_t7_path%it)
    #     main(X, data_t7_path%it)
    


    # exp_name = 'wgan-gp'
    # src_path = '../improved_wgan_training/saves/%s-%d.samples.npy'
    # dest_path = 'gen-data/%s-%d.t7'
    # iters = np.arange(1000,10000,1000)
    # m, std = torch.load('gen-data/wgan-real-normalizer.t7')
    # # iters = np.arange(10000,200000,10000)
    # for it in tqdm(iters):
    #     rawX = np.load(src_path%(exp_name, it))
    #     rawX = ((rawX) ).reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
    #     rawX = (rawX -m ) / std
    #     main(Variable(torch.FloatTensor(rawX)), dest_path%(exp_name, it))

    N_TRAINs = [1000,2000,3000,4000]

    src_path = '../improved_wgan_training/data/cifar10_train_x.npy'
    m, std = torch.load('gen-data/wgan-real-normalizer.t7')
    
    rawX = np.load(src_path)
    rawX = ((rawX) ).reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
    rawX = (rawX -m ) / std
    for N_TRAIN in N_TRAINs:
        dest_path = 'gen-data/real-%d.t7'%N_TRAIN
        rawX_ = rawX[:N_TRAIN]
        main(Variable(torch.FloatTensor(rawX_)), dest_path)
