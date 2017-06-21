import tensorflow as tf
import torch
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
from utils import MiniBatcher

import datetime

from model.fc import fc
from model.cnn import cnn

from tensorflow.python.platform import flags


def get_pred(model, support_X, support_Y, X_val, Y_val):
    support_F = model.forward(support_X).data
    support_Z = torch.mm(inv_H(support_Y), support_F)
    # support_U, _, _ = torch.svd(torch.mm(support_F, inverse(support_F)))
    # support_U = support_U[:,:10]
    # support_Z = torch.mm(inv_H(support_Y), support_U)

    val_F = model.forward(Variable(torch.FloatTensor(X_val))).data
    # val_U, _, _ = torch.svd(torch.mm(val_F, inverse(val_F)))
    # val_U = val_U[:,:10]
    val_U = val_F
    #compute distance
    Z_aug = support_Z[None].repeat(val_U.size()[0], 1,1)
    U_aug = val_U[:,None].repeat(1,support_Z.size()[0],1)
    val_pred = torch.pow(Z_aug- U_aug,2).numpy().sum(-1).argmin(-1)
    return val_pred


FLAGS = flags.FLAGS

flags.DEFINE_integer('max_train_iters', 5000, 'Number of iterations to train model')
flags.DEFINE_integer('batch_size', 1000, 'Size of training batches')
flags.DEFINE_float('fraction_labelled', 0.2, 'fraction of labelled training examples')
flags.DEFINE_string('exp_prefix', '', 'prefix for experiment name')
flags.DEFINE_string('model_name', 'cnn', '"fc" or "cnn"')
### Constant
NUM_VAL=1000



# def plot(ax, X, labels=None):
#     if labels is None:
#         X = X.numpy()
#         ax.scatter(X[:,0], X[:,1], color='k')
#     else:
#         for i in range(labels.size(1)):
#             if len(labels[:,i].nonzero().size()) == 0:
#                 continue
#             X_filt = X[labels[:,i].nonzero().view(-1)].numpy()
#             ax.scatter(X_filt[:,0], X_filt[:,1])

## Experiment stuff
if not os.path.exists('./saves/mnist'):
    os.makedirs('./saves/mnist')


## Metric
dist = Euclidean()


## Data
X, Y, X_test, Y_test = data_mnist()
X_val = X_test[:NUM_VAL]
Y_val = Y_test[:NUM_VAL]

X = torch.FloatTensor(X)
X = Variable(X)
Y = torch.FloatTensor(Y)
# Dataset (X size(N,D) , Y size(N,K))

## SSL mask
fraction_unlabelled = 1 - FLAGS.fraction_labelled
Y_semi = torch.FloatTensor(np.array(Y.numpy(),copy=True))
# Y_semi.index(torch.floor(torch.rand(X.size(0)) * n_cluster).long())
idxs = np.arange(Y.size()[0])
np.random.shuffle(idxs)
for idx in xrange(int(fraction_unlabelled * Y.size()[0])):
    Y_semi[idxs[idx]] = torch.ones(Y.size()[1])



## Model
if FLAGS.model_name =='fc':
    model = fc(1000)
elif FLAGS.model_name == 'cnn':
    model = cnn(128)

opt = optim.SGD(model.parameters(), lr = .01, momentum=0.5)


## tensorboard
timestamp = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
exp_name = '{}-{}-{}-{}'.format(FLAGS.exp_prefix, fraction_unlabelled, FLAGS.model_name,timestamp)

#ph
ph_accuracy = tf.placeholder(tf.float32,  name='accuracy')
ph_loss = tf.placeholder(tf.float32,  name='gain')
if not os.path.exists('./logs'):
    os.mkdir('./logs')
tf_acc = tf.summary.scalar('accuray', ph_accuracy)
tf_loss = tf.summary.scalar('gain', ph_loss)
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

# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
batcher = MiniBatcher(X.size()[0], batch_size=FLAGS.batch_size)

support_example_indices = np.nonzero(Y_semi.numpy().sum(1)==1)[0]
support_example_indices = support_example_indices[:FLAGS.batch_size]
support_X = X[torch.LongTensor(support_example_indices)]
support_Y = Y[torch.LongTensor(support_example_indices)]

## Algorithm
for idx in tqdm(xrange(FLAGS.max_train_iters)):
    idxs = batcher.next()
    X_batch = X[torch.LongTensor(idxs)]
    Y_batch = Y_semi[torch.LongTensor(idxs)]
    F = model.forward(X_batch).data
    # C_hat = torch.mm(F, inverse(F))
    u, s, v = torch.svd(F)
    ##TODO: use svd of F, not C_hat
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
    train_accuracy = (train_pred == train_gt).mean()
    #loss
    pt_F = F.data
    pt_H = H
    # C = torch.mm(pt_H, inverse(pt_H))
    # loss = torch.trace( torch.mm(C, torch.mm(pt_F, inverse(pt_F)).t() ))
    loss = torch.dot(torch.mm(inverse(pt_H),pt_F),torch.mm(inverse(pt_F), pt_H).t())
    
    ## checking this way of computing accuracy is correct
    ## works
    # val_pred = get_pred(model, X_batch,Y[torch.LongTensor(idxs)] , X_batch.data.numpy(), Y_batch.numpy())
    # doesn't work
    # val_pred = get_pred(model, X_batch,Y[torch.LongTensor(idxs)] , X[torch.LongTensor(np.arange(FLAGS.batch_size))].data.numpy(), Y[torch.LongTensor(np.arange(FLAGS.batch_size))].numpy())
    # doesn't work
    # val_pred = get_pred(model, support_X,support_Y , X_batch.data.numpy(), Y_batch.numpy())
    # doesn't work
    # val_pred = get_pred(model, X[torch.LongTensor(np.arange(FLAGS.batch_size))],Y[torch.LongTensor(np.arange(FLAGS.batch_size))] , X_batch.data.numpy(), Y_batch.numpy())
    # train_accuracy = np.mean(train_gt == val_pred)
    # print ('train')
    # print (train_accuracy)
    # summarize
    acc= sess.run(tf_acc, feed_dict={ph_accuracy:train_accuracy})
    loss = sess.run(tf_loss, feed_dict={ph_loss:loss})
    train_writer.add_summary(acc+loss, idx)

    #validate
    if idx>0 and idx%10==0:
        
        # val_pred = get_pred(model, X_batch, Y[torch.LongTensor(idxs)], X_val, Y_val)
        val_pred = get_pred(model, support_X, support_Y, X_val, Y_val)
        val_accuracy = np.mean(Y_val.argmax(1) == val_pred)
        print (val_accuracy)
        acc= sess.run(tf_acc, feed_dict={ph_accuracy:val_accuracy})
        val_writer.add_summary(acc, idx)




