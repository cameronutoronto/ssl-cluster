from __future__ import division
from docopt import docopt
import yaml
import torch
import tensorflow as tf
from torch import optim
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
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
# gen samples
data_t7_path = 'gen-data/dcgan-M-%i.t7'
iters = [1000,10000,100000,199000]
# real-net config
f_model_config = 'model/config/gan-dc-M.yaml'
f_saved_model = 'saves/fixed2:gan-dc-M-X-gan-clip-X-cifar-pc-400@2017-08-01/gen_%i.t7'
def _tile_images(imgs):
    """(100,32,32,3)"""
    return np.concatenate([np.concatenate(imgs[idx*10:(idx+1)*10], 0) for idx in xrange(10)],1)[None]
####
for model_idx in xrange(len(iters)):
	X, _, _, _ = torch.load(data_t7_path%iters[model_idx])
	X = X.data.numpy()[:100]
	plt.imshow(_tile_images(X)[0])
	plt.savefig((data_t7_path%iters[model_idx]).split('.')[0]+'.png')

	