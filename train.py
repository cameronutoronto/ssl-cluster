"""train.py
Usage:
    train.py <f_data_config> <f_model_config> <f_opt_config> [--prefix <p>] [--ce] [--db]
    train.py -r <exp_name> <idx> [--test]

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
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,10)
from torch.autograd import Variable
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import numpy as np

from metric import Euclidean
from cleverhans.utils_mnist import data_mnist
from utils import MiniBatcher, MiniBatcherPerClass
import torchvision.models as tvm
import datetime
from opt.loss import *
from model.fc import fc
from model.cnn import *
from ssl_utils import ssl_basic, ssl_per_class
import cPickle as pkl
from multiprocessing import Pool

#### 


def main(arguments):
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
        return pkl.load(open('data/ladder-cifar10-train-x-%i.p'%i,'rb')).transpose(0,2,3,1)
    def make_one_hot(train_y):
        new_y = np.zeros((train_y.shape[0], 10))
        new_y[np.arange(train_y.shape[0]),train_y[:,0]] = 1
        return new_y
    def data_cifar10():
        train = []
        train_y = make_one_hot(pkl.load(open('data/ladder-cifar10-train-y.p','rb')))
        test_y = make_one_hot(pkl.load(open('data/ladder-cifar10-test-y.p','rb')))
        test_x = pkl.load(open('data/ladder-cifar10-test-x.p','rb')).transpose(0,2,3,1)

        p = Pool(5)
        train_x = p.map(load_1_pkl, [0, 1, 2, 3, 4])
        return np.concatenate(train_x,0), train_y, test_x, test_y

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
    if 'fromfile' in arguments:
        X, Y, X_val, Y_val = torch.load(arguments['fromfile'])
    elif 'fromfile' in data_config:
        X, Y, X_val, Y_val = torch.load(data_config['fromfile'])
    else:
        if 'data_%s'%(data_config['name']) == 'data_cifar10':
            X, Y, X_val, Y_val = torch.load('data/t.t7')
        else:
            X, Y, _, _ = eval('data_%s'%(data_config['name']))()
            # NUM_VAL=data_config['num_val'] ## TODO: allow for more validation examples, loop over them...
            SPLIT = 40000
            Y_val = Y[SPLIT:SPLIT+5000]
            X_val = X[SPLIT:SPLIT+5000]
            Y = Y[:SPLIT]
            X = X[:SPLIT]

            X = torch.FloatTensor(X)
            Y = torch.FloatTensor(Y)
            X = Variable(X)
    # Dataset (X size(N,D) , Y size(N,K))

    ## SSL mask
    # Given completely labelled training Y, and a function, 
    # prepare Y_semi
    Y_semi = eval(data_config['ssl_config']['fname'])(Y, **data_config['ssl_config']['kwargs'])
    torch.save(Y_semi, './saves/%s/Y_semi.t7'%(exp_name))


    ## Model
    model = eval(model_config['name'])(**model_config['kwargs'])
    model.type(torch.cuda.FloatTensor)
    ## Optimizer
    opt = eval(opt_config['name'])(model.parameters(), **opt_config['kwargs'])
    # if 'lrsche' in opt_config:
    #     scheduler = eval(opt_config['lrsche']['name'])(opt, **opt_config['lrsche']['kwargs'])


    if arguments['-r']:
        model.load('./saves/%s/model_%s.t7'%(old_exp_name,arguments['<idx>']))
        opt.load_state_dict(torch.load('./saves/%s/opt_%s.t7'%(old_exp_name,arguments['<idx>'])))

        if arguments['--test']:
            raise NotImplementedError()


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

    log_folder = os.path.join('./logs', exp_name)
    # remove existing log folder for the same model.
    if os.path.exists(log_folder):
        import shutil
        shutil.rmtree(log_folder, ignore_errors=True)

    sess = tf.InteractiveSession(config=config)   

    train_writer = tf.summary.FileWriter(
        os.path.join(log_folder, 'train'), sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(log_folder, 'val'), sess.graph)
    tf.global_variables_initializer().run()

    opt_config['batcher_kwargs']['Y_semi'] = Y_semi
    batcher = eval(opt_config['batcher_name'])(X.size()[0], **opt_config['batcher_kwargs'])

    support_example_indices = np.nonzero(Y_semi.numpy().sum(1)==1)[0]
    support_example_indices = support_example_indices##WARNING ... 
    support_X = X[torch.LongTensor(support_example_indices)].type(torch.cuda.FloatTensor)
    support_Y = Y[torch.LongTensor(support_example_indices)].type(torch.cuda.FloatTensor)

    ## Loss
    if arguments['--ce']:
        Loss = CE()
    else:
        if 'loss' in opt_config:
            Loss = eval(opt_config['loss']['name'])(dist,support_X,support_Y, **opt_config['loss']['kwargs'])
        else:
            Loss = SSLCluster(dist,support_X,support_Y)
    best_val_acc = 0
    val_errors = []
    if not arguments['--db']:
        ## Algorithm
        for idx in tqdm(xrange(opt_config['max_train_iters'])):
            if 'lrsche' in opt_config and opt_config['lrsche'] != [] and opt_config['lrsche'][0][0] == idx:
                _, tmp_fac = opt_config['lrsche'].pop(0)
                sd = opt.state_dict()
                assert len(sd['param_groups']) ==1
                sd['param_groups'][0]['lr'] *= tmp_fac
                opt.load_state_dict(sd)

                
            idxs = batcher.next(idx)
            X_batch = X[torch.LongTensor(idxs)].type(torch.cuda.FloatTensor)
            Y_batch = Y_semi[torch.LongTensor(idxs)]#.type(torch.cuda.FloatTensor)
            ## network
            tv_F = model.forward(X_batch)
            F = tv_F.data.clone().type(torch.FloatTensor)
            ### loss layer
            loss, G, train_pred = Loss.train(F, Y_batch)

            model.zero_grad()
            tv_F.backward(gradient=G.type(torch.cuda.FloatTensor))
            opt.step()


            # TensorBoard
            #accuracy
            train_gt = Y[torch.LongTensor(idxs)].numpy().argmax(1)
            train_accuracy = (train_pred[batcher.start_unlabelled:] == train_gt[batcher.start_unlabelled:]).mean()

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
            if idx>0 and idx%200==0:
                def _validate_batch(model, X_val_batch, Y_val_batch, N_support):
                    model.eval()
                    val_pred, non_spectral_val_pred, val_pred_norm = Loss.infer(model, Variable(torch.FloatTensor(X_val_batch)).type(torch.cuda.FloatTensor), N_support)
                    val_accuracy = np.mean(Y_val_batch.argmax(1) == val_pred)
                    val_accuracy_non_spectral = np.mean(Y_val_batch.argmax(1) == non_spectral_val_pred)
                    val_accuracy_norm = np.mean(Y_val_batch.argmax(1) == val_pred_norm)
                    model.train()
                    return val_accuracy, val_accuracy_non_spectral, val_accuracy_norm
                start_valid = batcher.start_unlabelled
                if start_valid == 0:
                    start_valid = batcher.batch_size //2
                # WARNING: fixed size 50/50
                start_valid = batcher.batch_size //2
                val_batch_size = batcher.batch_size - start_valid ##WARNING: might need another split if we eventually train with very little labels per batch (in the case of #labels annealing)
                val_batches = Y_val.shape[0] // val_batch_size
                v1 = []
                v2 = []
                v3 = []
                for vidx in xrange(val_batches):
                    val_accuracy, val_accuracy_non_spectral, val_accuracy_norm = _validate_batch(model, X_val[vidx*val_batch_size:(vidx+1)*val_batch_size], Y_val[vidx*val_batch_size:(vidx+1)*val_batch_size], start_valid)
                    v1.append(val_accuracy)
                    v2.append(val_accuracy_non_spectral)
                    v3.append(val_accuracy_norm)
                val_accuracy = np.mean(v1)
                val_accuracy_non_spectral = np.mean(v2)
                val_accuracy_norm = np.mean(v3)
                print (val_accuracy)
                acc= sess.run(tf_acc, feed_dict={ph_accuracy:val_accuracy})
                acc_non_spectral= sess.run(tf_accuracy_non_spectral, feed_dict={ph_accuracy_non_spectral:val_accuracy_non_spectral})
                acc_norm= sess.run(tf_accuracy_norm, feed_dict={ph_accuracy_norm:val_accuracy_norm})
                val_writer.add_summary(acc+acc_norm+acc_non_spectral, idx)
                val_errors.append(val_accuracy)
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    name = './saves/%s/model_best.t7'%(exp_name)
                    print ("[Saving to]")
                    print (name)
                    model.save(name)
            ## checkpoint
            if idx>0 and idx%1000==0:
                name = './saves/%s/model_%i.t7'%(exp_name,idx)
                print ("[Saving to]")
                print (name)
                model.save(name)
                torch.save(opt.state_dict(), './saves/%s/opt_%i.t7'%(exp_name,idx))
    pkl.dump(val_errors, open(os.path.join(log_folder, 'val.log'), 'wb'))
    return best_val_acc


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    main(arguments)

