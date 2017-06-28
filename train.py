"""train.py
Usage:
    train.py <f_data_config> <f_model_config> <f_opt_config> [--prefix <p>]

Arguments:
    <f_data_config>  example ''data/config/train_rev0.yaml''
    <f_model_config> example 'model/config/conv2d-3layers.yaml'
    <f_opt_config> example 'opt/config/basic.yaml'

Example:

Options:
"""

from docopt import docopt
import yaml
import torch
import tensorflow as tf
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

from main import solve_H, grad_F, inv_H
from cluster_utils import centroids, energy, inverse
from metric import Euclidean
from cleverhans.utils_mnist import data_mnist
from utils import MiniBatcher, MiniBatcherPerClass

import datetime

from model.fc import fc
from model.cnn import *
from ssl_utils import ssl_basic, ssl_per_class
import cPickle as pkl
from multiprocessing import Pool

#### 
arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")

f_data_config = arguments['<f_data_config>']
f_model_config = arguments['<f_model_config>']
f_opt_config = arguments['<f_opt_config>']
data_config = yaml.load(open(f_data_config, 'rb'))
model_config = yaml.load(open(f_model_config, 'rb'))
opt_config = yaml.load(open(f_opt_config, 'rb'))
data_name = os.path.basename(f_data_config).split('.')[0]
model_name = os.path.basename(f_model_config).split('.')[0]
opt_name = os.path.basename(f_opt_config).split('.')[0]

timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())
if arguments['--prefix']:
    exp_name = '%s:%s-X-%s-X-%s-%s' % (arguments['<p>'], model_name, opt_name, data_name, timestamp)
else:
    exp_name = '%s-X-%s-X-%s-%s' % (model_name, opt_name, data_name, timestamp)
print ('\n\n\n\n>>>>>>>>> [Experiment Name]')
print (exp_name)
print ('<<<<<<<<<\n\n\n\n')


def load_1_pkl(i):
    return pkl.load(open('data/cifar10-train-x-%i.p'%i,'rb'))
def data_cifar10():
    """
    Preprocess CIFAR10 dataset
    :return:
    """

    # # These values are specific to CIFAR10
    # img_rows = 32
    # img_cols = 32
    # nb_classes = 10

    # # the data, shuffled and split between train and test sets
    # (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # if keras.backend.image_dim_ordering() == 'th':
    #     X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    #     X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    # else:
    #     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    #     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255
    # print('X_train shape:', X_train.shape)
    # print(X_train.shape[0], 'train samples')
    # print(X_test.shape[0], 'test samples')

    # # convert class vectors to binary class matrices
    # Y_train = np_utils.to_categorical(y_train, nb_classes)
    # Y_test = np_utils.to_categorical(y_test, nb_classes)
    # return X_train, Y_train, X_test, Y_test
    # return pkl.load(open('data/cifar10.p','rb'))

    train = []
    train_y = pkl.load(open('data/cifar10-train-y.p','rb'))
    test_y = pkl.load(open('data/cifar10-test-y.p','rb'))
    test_x = pkl.load(open('data/cifar10-test-x.p','rb'))

    p = Pool(5)
    train_x = p.map(load_1_pkl, [0, 1, 2, 3, 4])
    return np.concatenate(train_x,0), train_y, test_x, test_y







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
    val_pred = torch.pow(Z_aug- U_aug,2).numpy().sum(-1).argmin(-1)
    return val_pred

def get_pred_join_SVD(model, support_X, support_Y, X_val, Y_val, normalize=False, get_non_spectral=False):
    """
    concatenate support_X with X_val,
    perform SVD, using support_X to compute centroid in U, 
    do assignment for X_val
    """
    joint_X = torch.cat((support_X, Variable(torch.FloatTensor(X_val))), 0)
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



## Experiment stuff
if not os.path.exists('./saves/%s'%exp_name):
    os.makedirs('./saves/%s'%exp_name)


## Metric
dist = Euclidean()



## Data
X, Y, X_test, Y_test = eval('data_%s'%(data_config['name']))()
NUM_VAL=data_config['num_val'] ## TODO: allow for more validation examples, loop over them...
Y_val = Y_test[:NUM_VAL]
X_val = X_test[:NUM_VAL]

X = torch.FloatTensor(X)
X = Variable(X)
Y = torch.FloatTensor(Y)
# Dataset (X size(N,D) , Y size(N,K))

## SSL mask
# Given completely labelled training Y, and a function, 
# prepare Y_semi
Y_semi = eval(data_config['ssl_config']['fname'])(Y, **data_config['ssl_config']['kwargs'])



## Model
model = eval(model_config['name'])(**model_config['kwargs'])

## Optimizer
opt = eval(opt_config['name'])(model.parameters(), **opt_config['kwargs'])


## tensorboard
#ph
ph_accuracy = tf.placeholder(tf.float32,  name='accuracy')
ph_accuracy_norm = tf.placeholder(tf.float32,  name='accuracy_norm')
ph_accuracy_non_spectral = tf.placeholder(tf.float32,  name='accuracy_non_spectral')
ph_loss = tf.placeholder(tf.float32,  name='gain')
ph_Gnorm = tf.placeholder(tf.float32, name='G_norm')
ph_Ysemi_labs = tf.placeholder(tf.int32, shape=[None], name='Ysemi_labs')
ph_class_min = tf.placeholder(tf.int32, name='class_min')
if not os.path.exists('./logs'):
    os.mkdir('./logs')
tf_acc = tf.summary.scalar('accuracy', ph_accuracy)
tf_accuracy_norm = tf.summary.scalar('accuracy_norm', ph_accuracy_norm)
tf_accuracy_non_spectral = tf.summary.scalar('accuracy_non_spectral', ph_accuracy_non_spectral)
tf_loss = tf.summary.scalar('gain', ph_loss)
tf_Gnorm = tf.summary.scalar('G_norm', ph_Gnorm)
tf_Ysemi_labs = tf.summary.histogram('Ysemi_labs', ph_Ysemi_labs)
tf_class_min = tf.summary.scalar('class_min_count', ph_class_min)

tf_summary = tf.summary.merge_all()
log_folder = os.path.join('./logs', exp_name)
# remove existing log folder for the same model.
if os.path.exists(log_folder):
    import shutil
    shutil.rmtree(log_folder, ignore_errors=True)

sess = tf.InteractiveSession()   

train_writer = tf.summary.FileWriter(
    os.path.join(log_folder, 'train'), sess.graph)
val_writer = tf.summary.FileWriter(os.path.join(log_folder, 'val'), sess.graph)
tf.global_variables_initializer().run()

opt_config['batcher_kwargs']['Y_semi'] = Y_semi
batcher = eval(opt_config['batcher_name'])(X.size()[0], **opt_config['batcher_kwargs'])

support_example_indices = np.nonzero(Y_semi.numpy().sum(1)==1)[0]
support_example_indices = support_example_indices[:1000] ##WARNING ... 
support_X = X[torch.LongTensor(support_example_indices)]
support_Y = Y[torch.LongTensor(support_example_indices)]

## Algorithm
for idx in tqdm(xrange(opt_config['max_train_iters'])):
    idxs = batcher.next()
    X_batch = X[torch.LongTensor(idxs)]
    Y_batch = Y_semi[torch.LongTensor(idxs)]
    F = model.forward(X_batch).data
    # C_hat = torch.mm(F, inverse(F))
    u, s, v = torch.svd(F)
    ## URGENT TODO: use svd of F, not C_hat
    # u_r = u[:,:np.linalg.matrix_rank(C_hat.numpy())]
    u_r = u
    
    H_init = np.array(Y_batch.numpy(), copy=True)
    H_init[H_init.sum(1) > 1] *= 0
    H = solve_H(torch.FloatTensor(H_init), u_r,dist,Y=Y_batch.numpy(),iters=100)

    model.zero_grad()
    G = grad_F(F,H)
    F = model.forward(X_batch)
    F.backward(gradient=G)
    opt.step()


    # TensorBoard
    #accuracy
    train_pred = H.numpy().argmax(1)
    train_gt = Y[torch.LongTensor(idxs)].numpy().argmax(1)
    train_accuracy = (train_pred[batcher.start_unlabelled:] == train_gt[batcher.start_unlabelled:]).mean()
    #loss
    pt_F = F.data
    pt_H = H
    # C = torch.mm(pt_H, inverse(pt_H))
    # loss = torch.trace( torch.mm(C, torch.mm(pt_F, inverse(pt_F)).t() ))
    loss = torch.dot(torch.mm(inverse(pt_H),pt_F),torch.mm(inverse(pt_F), pt_H).t())

    # summarize
    acc= sess.run(tf_acc, feed_dict={ph_accuracy:train_accuracy})
    loss = sess.run(tf_loss, feed_dict={ph_loss:loss})
    tmp_Gnorm = sess.run(tf_Gnorm, feed_dict={ph_Gnorm:G.norm()})
    tmp = Y_batch.numpy()
    ylab = tmp[tmp.sum(1)==1].argmax(1)
    tmp_Ysemi_labs = sess.run(tf_Ysemi_labs, feed_dict={ph_Ysemi_labs:ylab.astype('int32')})
    tmp_class_min = sess.run(tf_class_min, feed_dict={ph_class_min:np.bincount(tmp[tmp.sum(1)==1].argmax(1)).min()})
    train_writer.add_summary(acc+loss+tmp_Gnorm+tmp_Ysemi_labs+tmp_class_min, idx)

    #validate
    if idx>0 and idx%10==0:
        model.eval()
        # val_pred = get_pred(model, support_X, support_Y, X_val, Y_val)
        val_pred, non_spectral_val_pred = get_pred_join_SVD(model, support_X, support_Y, X_val, Y_val, get_non_spectral=True)
        val_accuracy = np.mean(Y_val.argmax(1) == val_pred)
        val_accuracy_non_spectral = np.mean(Y_val.argmax(1) == non_spectral_val_pred)
        print (val_accuracy)
        acc= sess.run(tf_acc, feed_dict={ph_accuracy:val_accuracy})
        acc_non_spectral= sess.run(tf_accuracy_non_spectral, feed_dict={ph_accuracy_non_spectral:val_accuracy_non_spectral})
        val_pred,_ = get_pred_join_SVD(model, support_X, support_Y, X_val, Y_val, normalize=True)
        val_accuracy = np.mean(Y_val.argmax(1) == val_pred)
        acc_norm= sess.run(tf_accuracy_norm, feed_dict={ph_accuracy_norm:val_accuracy})
        val_writer.add_summary(acc+acc_norm+acc_non_spectral, idx)
        model.train()
    ## checkpoint
    if idx>0 and idx%5==0:
        name = './saves/%s/model_%i.t7'%(exp_name,idx)
        print ("[Saving to]")
        print (name)
        model.save(name)
        torch.save(opt.state_dict(), './saves/%s/opt_%i.t7'%(exp_name,idx))




