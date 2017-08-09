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

def main(data_t7_path):
    rdata_t7_path = 'gen-data/real.t7'
    # real-net config
    f_model_config = 'model/config/cnn-globe.yaml'
    real_net_path = 'saves/CE.sl-baseline:cnn-globe-X-sl-128-lr-X-cifar@2017-08-03/model_19900.t7'

    ####
    #gen data
    X,_,_,_ = torch.load(data_t7_path)
    _,_,X_val,Y_val = torch.load(rdata_t7_path)
    #real-net
    model_config = yaml.load(open(f_model_config, 'rb'))
    real_net = eval(model_config['name'])(**model_config['kwargs'])
    real_net.load(real_net_path)
    real_net.type(torch.cuda.FloatTensor)
       
    softmax = torch.nn.Softmax()

    batch_size=100
    weak_labels = []
    # for idx in tqdm(xrange(1)):
    for idx in tqdm(xrange(X.size()[0]//batch_size)):
        idxs = np.arange(idx*batch_size, (idx+1)*batch_size)
        X_batch = X[torch.LongTensor(idxs)].type(torch.cuda.FloatTensor)
        ## network
        py_x = softmax(real_net.forward(X_batch))
        weak_labels.append(py_x.data.cpu().numpy())
    torch.save((X, torch.FloatTensor(np.concatenate(weak_labels, 0)), X_val, Y_val), data_t7_path)

if __name__ == '__main__':
    # gen samples
    # data_t7_path = 'gen-data/real.t7'
    data_t7_path = 'gen-data/dcgan-M-%i.t7'
    iters = [1000,10000,100000,199000]
    for it in iters:
        main(data_t7_path%it)