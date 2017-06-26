import torch
import numpy as np


def ssl_basic(Y, fraction_labelled=1):
	fraction_unlabelled = 1 - fraction_labelled
	Y_semi = torch.FloatTensor(np.array(Y.numpy(),copy=True))
	idxs = np.arange(Y.size()[0])
	np.random.shuffle(idxs)
	for idx in xrange(int(fraction_unlabelled * Y.size()[0])):
	    Y_semi[idxs[idx]] = torch.ones(Y.size()[1])
	return Y_semi



def ssl_per_class(Y, labelled_per_class=10):
	Y_semi = torch.FloatTensor(np.array(Y.numpy(),copy=True))
	idxs_per_class = [np.where(Y.numpy().argmax(1)==class_idx)[0] for class_idx in xrange(Y.size()[1])]
	for class_idx in xrange(Y.size()[1]):
		idxs = idxs_per_class[class_idx]
		np.random.shuffle(idxs)
		for idx in xrange(labelled_per_class):
			Y_semi[idxs[idx]] = torch.ones(Y.size()[1])
	return Y_semi