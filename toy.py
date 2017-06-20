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

from main import solve_H, grad_F
from cluster_utils import centroids, energy, inverse
from metric import Euclidean

def plot(ax, X, labels=None):
    if labels is None:
        X = X.numpy()
        ax.scatter(X[:,0], X[:,1], color='k')
    else:
        for i in range(labels.size(1)):
            if len(labels[:,i].nonzero().size()) == 0:
                continue
            X_filt = X[labels[:,i].nonzero().view(-1)].numpy()
            ax.scatter(X_filt[:,0], X_filt[:,1])

## Experiment stuff
max_train_iters = 200
if not os.path.exists('./saves/toy'):
    os.makedirs('./saves/toy')


## Metric
dist = Euclidean()


## Data
n_cluster = 5
n_examples = 100 #per cluster
sigma = 0.2

theta = torch.linspace(0, 2 * np.pi, n_cluster+1)[:-1]
mu = torch.cat([torch.cos(theta).view(-1, 1), torch.sin(theta).view(-1, 1)], 1)
eps = sigma * torch.randn(n_cluster, n_examples, 2)
X = mu.view(n_cluster, 1, 2).expand_as(eps) + eps
X.resize_(n_cluster * n_examples, 2)
X = Variable(X)
Y = np.tile(np.eye(n_cluster), (1,n_examples)).reshape((-1,n_cluster))
Y = torch.FloatTensor(Y)

# Dataset (X size(N,D) , Y size(N,K))

## SSL mask
fraction_unlabelled = .8
Y_semi = Y
# Y_semi.index(torch.floor(torch.rand(X.size(0)) * n_cluster).long())
idxs = np.arange(n_cluster * n_examples)
np.random.shuffle(idxs)
for idx in xrange(int(fraction_unlabelled * Y.size()[0])):
    Y_semi[idxs[idx]] = torch.ones(n_cluster)

## Model
D_in = 2
Hn = 30
D_out = D_in
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, Hn),
    torch.nn.ReLU(),
    torch.nn.Linear(Hn, Hn),
    torch.nn.ReLU(),
    torch.nn.Linear(Hn, Hn),
    torch.nn.ReLU(),
    torch.nn.Linear(Hn, D_out),
)

opt = optim.SGD(model.parameters(), lr = .01, momentum=0.5)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
## Algorithm
for idx in tqdm(xrange(max_train_iters)):
	F = model(X).data
	C_hat = torch.mm(F, inverse(F))
	u, s, v = torch.svd(C_hat)
	u_r = u[:,:np.linalg.matrix_rank(C_hat.numpy())]
	H_init = torch.eye(n_cluster).index(torch.floor(torch.rand(X.size(0)) * n_cluster).long()) ## fix this to satisfy SSL constraint
	H = solve_H(H_init, u_r,dist,Y=Y_semi.numpy(),iters=100)

	model.zero_grad()
	G = grad_F(F,H)
	F = model(X)
	F.backward(gradient=G)
	opt.step()

	## Plot
	ax1.cla()
	ax2.cla()
	plot(ax1, F.data, H)
	plot(ax2, X.data, H)
	plt.savefig('./saves/toy/{}.png'.format(idx))
