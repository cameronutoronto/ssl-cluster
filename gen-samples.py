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
# gen samples
data_t7_path = 'gen-data/dcgan-M-%i.t7'
iters = [1000,10000,100000,199000]
# real-net config
f_model_config = 'model/config/gan-dc-M.yaml'
f_saved_model = 'saves/fixed2:gan-dc-M-X-gan-clip-X-cifar-pc-400@2017-08-01/gen_%i.t7'

####
for model_idx in xrange(len(iters)):
	#real-net
	model_config = yaml.load(open(f_model_config, 'rb'))
	gennet = eval(model_config['gen']['name'])(**model_config['gen']['kwargs'])
	# model_idx = 0
	gennet.load(f_saved_model%iters[model_idx])
	gennet.type(torch.cuda.FloatTensor)
	   

	batch_size=100
	noise = torch.FloatTensor(batch_size, gennet.nz, 1, 1)
	noise = noise.cuda()
	samples = []
	for idx in tqdm(xrange(400)):
	    noise.normal_(0,.1)
	    noisev = Variable(noise, requires_grad=False)
	    sample = gennet.forward(noisev)
	    sample = sample.data.cpu()
	    samples.append(sample)
	    gennet.zero_grad()
	torch.save((Variable(torch.cat(samples, 0)), None, None, None), data_t7_path%iters[model_idx])
