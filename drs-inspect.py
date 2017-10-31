
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

gan_modes = ['wgan', 'dcgan'] # ()
# gan_iter = 2000
gan_dsizes = [2000, 4000, 6000, 8000, 10000, 20000, 40000]
src_path = '../improved_wgan_training/-{gan_mode}-{gan_dsize}/{gan_mode}-{gan_iter}.samples.npy'
dest_path = 'gen-data/%s.t7'
iters = list(np.arange(1000,10000,1000)) + list(np.arange(10000,200000,10000))

for gan_mode in gan_modes:
    for gan_dsize in gan_dsizes:
        for gan_iter in iters:
            exp_name = '%s-%g-%g'%(gan_mode, gan_iter, gan_dsize)
            try:
                pkl.load('scores/{exp_name}.pkl'.format(exp_name))
            except Exception as e:
                pass

