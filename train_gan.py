"""train_gan.py
Usage:
    train_gan.py <f_data_config> <f_model_config> <f_opt_config> [--prefix <p>] [--ce] [--db] [--smalldata]
    train_gan.py -r <exp_name> <idx> [--test]

Arguments:
    <f_data_config>  example ''data/config/train_rev0.yaml''
    <f_model_config> example 'model/config/conv2d-3layers.yaml'
    <f_opt_config> example 'opt/config/basic.yaml'

Example:

Options:
"""
from __future__ import division
from docopt import docopt
import yaml
import torch
import tensorflow as tf
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
# sess = tf.Session(config=config)
from torch import optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(10,10)
from torch.autograd import Variable
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import numpy as np

from metric import Euclidean
from cleverhans.utils_mnist import data_mnist
from utils import MiniBatcher, MiniBatcherPerClass

import datetime
from opt.loss import *
from model.fc import fc
from model.cnn import *
from model.gen import *
from ssl_utils import ssl_basic, ssl_per_class
import cPickle as pkl
from multiprocessing import Pool

#### 
arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")

if arguments['-r']:
    exp_name = arguments['<exp_name>']
    f_model_config = 'model/config/'+exp_name[exp_name.find(':')+1:].split('-X-')[0]+'.yaml'
    f_opt_config = 'opt/config/'+exp_name[exp_name.find(':')+1:].split('-X-')[1]+'.yaml'
    f_data_config = 'data/config/'+exp_name[exp_name.find(':')+1:].split('-X-')[2].split('@')[0] +'.yaml'
    old_exp_name = exp_name
    exp_name += '_resumed'
else:
    f_data_config = arguments['<f_data_config>']
    f_model_config = arguments['<f_model_config>']
    f_opt_config = arguments['<f_opt_config>']
    data_name = os.path.basename(f_data_config).split('.')[0]
    model_name = os.path.basename(f_model_config).split('.')[0]
    opt_name = os.path.basename(f_opt_config).split('.')[0]
    timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())
    if arguments['--prefix']:
        exp_name = '%s:%s-X-%s-X-%s@%s' % (arguments['<p>'], model_name, opt_name, data_name, timestamp)
    else:
        exp_name = '%s-X-%s-X-%s@%s' % (model_name, opt_name, data_name, timestamp)
    if arguments['--ce']:
        exp_name = 'CE.' + exp_name
    
data_config = yaml.load(open(f_data_config, 'rb'))
model_config = yaml.load(open(f_model_config, 'rb'))
opt_config = yaml.load(open(f_opt_config, 'rb'))

print ('\n\n\n\n>>>>>>>>> [Experiment Name]')
print (exp_name)
print ('<<<<<<<<<\n\n\n\n')


def load_1_pkl(i):
    return pkl.load(open('data/cifar10-train-x-%i.p'%i,'rb'))
def make_one_hot(train_y):
    new_y = np.zeros((train_y.shape[0], 10))
    new_y[np.arange(train_y.shape[0]),train_y[:,0]] = 1
    return new_y
def data_cifar10():
    train = []
    train_y = pkl.load(open('data/cifar10-train-y.p','rb'))
    test_y = pkl.load(open('data/cifar10-test-y.p','rb'))
    # test_x = pkl.load(open('data/cifar10-test-x.p','rb')).transpose(0,2,3,1)

    p = Pool(5)
    train_x = p.map(load_1_pkl, [0, 1, 2, 3, 4])
    return np.concatenate(train_x,0), train_y, None, test_y

# def load_1_pkl(i):
#     return pkl.load(open('data/cifar10-train-x-%i.p'%i,'rb'))
# def data_cifar10():
#     train = []
#     train_y = pkl.load(open('data/cifar10-train-y.p','rb'))
#     test_y = pkl.load(open('data/cifar10-test-y.p','rb'))
#     test_x = pkl.load(open('data/cifar10-test-x.p','rb'))

#     p = Pool(5)
#     train_x = p.map(load_1_pkl, [0, 1, 2, 3, 4])
#     return np.concatenate(train_x,0), train_y, test_x, test_y


## Experiment stuff
if not os.path.exists('./saves/%s'%exp_name):
    os.makedirs('./saves/%s'%exp_name)


## Metric
dist = Euclidean()



## Data
if not arguments['--smalldata']:
    X, Y, _, _ = eval('data_%s'%(data_config['name']))()
    SPLIT = 40000
    Y_val = Y[SPLIT:SPLIT+5000]
    X_val = X[SPLIT:SPLIT+5000]
    Y = Y[:SPLIT]
    X = X[:SPLIT]
else:
    X, Y = pkl.load(open('data/cifar10-small.p','rb'))
    SPLIT = 400
    Y_val = Y[SPLIT:SPLIT+100]
    X_val = X[SPLIT:SPLIT+100]
    Y = Y[:SPLIT]
    X = X[:SPLIT]
# NUM_VAL=data_config['num_val'] ## TODO: allow for more validation examples, loop over them...

X = torch.FloatTensor(X)
X = Variable(X)
Y = torch.FloatTensor(Y)
# Dataset (X size(N,D) , Y size(N,K))

## SSL mask
# Given completely labelled training Y, and a function, 
# prepare Y_semi
Y_semi = eval(data_config['ssl_config']['fname'])(Y, **data_config['ssl_config']['kwargs'])
torch.save(Y_semi, './saves/%s/Y_semi.t7'%(exp_name))


## Model
discnet = eval(model_config['disc']['name'])(**model_config['disc']['kwargs'])
discnet.type(torch.cuda.FloatTensor)
gennet = eval(model_config['gen']['name'])(**model_config['gen']['kwargs'])
gennet.type(torch.cuda.FloatTensor)
## Optimizer
discopt = eval(opt_config['name'])(discnet.parameters(), **opt_config['kwargs'])
genopt = eval(opt_config['name'])(gennet.parameters(), **opt_config['kwargs'])



if arguments['-r']:
    raise NotImplementedError() # not made changes yet
    model.load('./saves/%s/model_%s.t7'%(old_exp_name,arguments['<idx>']))
    opt.load_state_dict(torch.load('./saves/%s/opt_%s.t7'%(old_exp_name,arguments['<idx>'])))

    if arguments['--test']:
        raise NotImplementedError()


## tensorboard
#ph
ph_accuracy = tf.placeholder(tf.float32,  name='accuracy')
ph_images = tf.placeholder(tf.float32, shape=(1,320,320,3), name='images')
if not os.path.exists('./logs'):
    os.mkdir('./logs')
tf_acc = tf.summary.scalar('accuracy', ph_accuracy)
tf_images = tf.summary.image('image',ph_images)


log_folder = os.path.join('./logs', exp_name)
# remove existing log folder for the same model.
if os.path.exists(log_folder):
    import shutil
    shutil.rmtree(log_folder, ignore_errors=True)

sess = tf.InteractiveSession(config=config)   

train_writer = tf.summary.FileWriter(
    os.path.join(log_folder, 'train'), sess.graph)
# val_writer = tf.summary.FileWriter(os.path.join(log_folder, 'val'), sess.graph)
tf.global_variables_initializer().run()

opt_config['batcher_kwargs']['Y_semi'] = Y_semi
batcher = eval(opt_config['batcher_name'])(X.size()[0], **opt_config['batcher_kwargs'])

support_example_indices = np.nonzero(Y_semi.numpy().sum(1)==1)[0]
support_example_indices = support_example_indices##WARNING ... 
support_X = X[torch.LongTensor(support_example_indices)].type(torch.cuda.FloatTensor)
support_Y = Y[torch.LongTensor(support_example_indices)].type(torch.cuda.FloatTensor)

## Loss
Loss = CE()

noise = torch.FloatTensor(batcher.batch_size, gennet.nz, 1, 1)
noise = noise.cuda()

real_label = torch.zeros(batcher.batch_size, 2)
real_label[:,1] = 1
fake_label = torch.zeros(batcher.batch_size, 2)
fake_label[:,0] = 1


if not arguments['--db']:
    ## Algorithm
    for idx in tqdm(xrange(opt_config['max_train_iters'])):
        if 'lrsche' in opt_config and opt_config['lrsche'] != [] and opt_config['lrsche'][0][0] == idx:
            _, tmp_fac = opt_config['lrsche'].pop(0)
            def _update_lr(opt, tmp_fac):
                sd = opt.state_dict()
                assert len(sd['param_groups']) ==1
                sd['param_groups'][0]['lr'] *= tmp_fac
                opt.load_state_dict(sd)
            _update_lr(discopt, tmp_fac)
            _update_lr(genopt, tmp_fac)
            
        idxs = batcher.next(idx)
        X_batch = X[torch.LongTensor(idxs)].type(torch.cuda.FloatTensor)
        # Y_batch = Y_semi[torch.LongTensor(idxs)]
        ################
        # update discnet
        ################
        if 'disc_iters' in opt_config:
            disc_iters = opt_config['disc_iters']
        else:
            disc_iters = 1
        for _ in xrange(disc_iters):
            if 'disc_clamp' in opt_config:
                for p in discnet.parameters():
                    p.data.clamp_(-opt_config['disc_clamp'],opt_config['disc_clamp'])
            discnet.zero_grad()
            ## network output given real
            tv_F_real = discnet.forward(X_batch)
            loss, G_real, real_pred = Loss.train(tv_F_real.data.clone().type(torch.FloatTensor), real_label)
            tv_F_real.backward(gradient=((G_real)/2).type(torch.cuda.FloatTensor))
            ## network output given fake
            noise.normal_(0,.1)
            noisev = Variable(noise)
            fake_imgs = gennet.forward(noisev)
            tv_F_fake = discnet.forward(fake_imgs.detach())
            loss, G_fake, fake_pred = Loss.train(tv_F_fake.data.clone().type(torch.FloatTensor), fake_label)
            tv_F_fake.backward(gradient=((G_fake)/2).type(torch.cuda.FloatTensor))
            discopt.step()

        ################
        # update gennet
        ################
        discnet.zero_grad()
        noise.normal_(0,.1)
        noisev = Variable(noise)
        fake_imgs = gennet.forward(noisev)

        tv_F = discnet.forward(fake_imgs)
        loss, G, _ = Loss.train(tv_F.data.clone().type(torch.FloatTensor), real_label)
        gennet.zero_grad()
        tv_F.backward(gradient=G.type(torch.cuda.FloatTensor))
        genopt.step()

        # TensorBoard
        # summarize
        if idx>0 and idx%100==0:
            train_acc = ((fake_pred==0).sum() + (real_pred==1).sum() ) / (batcher.batch_size * 2)
            acc= sess.run(tf_acc, feed_dict={ph_accuracy:train_acc})
            def _tile_images(imgs):
                """(100,32,32,3)"""
                return np.concatenate([np.concatenate(imgs[idx*10:(idx+1)*10], 0) for idx in xrange(10)],1)[None]

            su_images = sess.run(tf_images, feed_dict={ph_images:_tile_images(fake_imgs.data.cpu().numpy())})
            train_writer.add_summary(acc+su_images, idx)

        ## checkpoint
        if idx>0 and idx%1000==0:
            name = './saves/%s/%s_%i.t7'
            print ("[Saving to]")
            print (name%(exp_name,'gen/disc',idx))
            gennet.save(name%(exp_name,'gen',idx))
            discnet.save(name%(exp_name,'disc',idx))
            torch.save(genopt.state_dict(), './saves/%s/genopt_%i.t7'%(exp_name,idx))
            torch.save(discopt.state_dict(), './saves/%s/discopt_%i.t7'%(exp_name,idx))




