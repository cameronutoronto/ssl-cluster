import os
import sys
import numpy as np
exp_name = 'wgan-gp'
# iters = np.arange(10000,200000,10000)
iters = np.arange(1000,10000,1000)
data_src = 'gen-data/wgan-gp-%d.t7'

if not os.path.exists(os.path.join('./data/config',exp_name)):
    os.makedirs(os.path.join('./data/config',exp_name))
for it in iters:
    content = """fromfile: '{data_src}'
name: cifar10
num_val: 1000

ssl_config:
  fname: ssl_basic
  kwargs:
    fraction_labelled: 1""".format(data_src=data_src%it)
    with open(os.path.join('./data/config',exp_name,'%d.yaml'%it),'w') as f:
        f.write(content)