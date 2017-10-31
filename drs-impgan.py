"""drs-impgan.py
Usage:
    drs-impgan.py <gan_mode> <gan_dsize> [--dim <dim>]

Example:

Options:
"""
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
    real_net_path = 'saves/CE.cnn-globe-large-X-sl-128-lr-X-real-normalized@2017-09-18/model_best.t7'
    # real_net_path = 'saves/CE.wgan-fixed:cnn-globe-large-X-sl-128-lr-X-real-normalized@2017-08-10/model_best.t7'

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
    print( data_t7_path)
    ### Inception Score is the mean of the per-example score
    ## so we need to first do that
    ## then compute mean, std of that score 
    inception_scores = np.exp(np.array(inception_scores).reshape(10000,-1).mean(0))
    print('[Inception Score]: %g, %g'%(np.mean(inception_scores), np.std(inception_scores)))
    return (np.mean(inception_scores), np.std(inception_scores))

if __name__ == '__main__':
    arguments = docopt(__doc__)
    from train import main as train_m
    class_arguments = {}
    class_arguments['<f_data_config>'] = 'data/config/wgan-gp/1000.yaml'
    class_arguments['<f_model_config>'] = 'model/config/cnn-globe-large.yaml'
    class_arguments['<f_opt_config>'] = 'opt/config/sl-128-lr.yaml'
    class_arguments['--ce'] = True
    class_arguments['--prefix'] = True
    class_arguments['-r'] = False
    class_arguments['--db'] = False
    gan_mode = arguments['<gan_mode>'] # ()
    if arguments['--dim']:
        gan_dim =int(arguments['<dim>'])
    else:
        gan_dim = ''
    # gan_iter = 2000
    gan_dsize = int(arguments['<gan_dsize>'])
    src_path = '../improved_wgan_training/-{gan_mode}-{gan_dsize}{gan_dim}/{gan_mode}-{gan_iter}.samples.npy'
    dest_path = 'gen-data/%s.t7'
    iters = list(np.arange(10000,200000,20000))
    # iters = list(np.arange(1000,10000,1000)) + list(np.arange(10000,200000,10000))
    m, std = torch.load('gen-data/wgan-real-normalizer.t7')

    for gan_iter in iters:
        exp_name = '{gan_mode}-{gan_iter}-{gan_dsize}-{gan_dim}'.format(gan_mode=gan_mode, gan_iter=gan_iter, gan_dsize=gan_dsize, gan_dim=gan_dim)
        rawX = np.load(src_path.format(gan_mode=gan_mode, gan_iter=gan_iter, gan_dsize=gan_dsize, gan_dim=gan_dim))
        rawX = ((rawX) ).reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
        rawX = (rawX -m ) / std
        is_mean, is_std = main(Variable(torch.FloatTensor(rawX)), dest_path%(exp_name))
        class_arguments['fromfile'] = dest_path%(exp_name)
        class_arguments['<p>'] = '{exp_name}'.format(exp_name=exp_name)
        drs = train_m(class_arguments)
        pkl.dump((is_mean, is_std, drs), open('scores/%s.p'%exp_name,'wb'))

# real 500(6.90302, 0.0392672, .43 )

