from __future__ import division
import numpy as np


class MiniBatcher(object):
	def __init__(self, N, batch_size=32, loop=True, Y_semi=None, fraction_labelled_per_batch=None):
		self.N = N
		self.batch_size=batch_size
		self.loop = loop
		self.idxs = np.arange(N)
		np.random.shuffle(self.idxs)
		self.curr_idx = 0
		self.fraction_labelled_per_batch = fraction_labelled_per_batch
		if fraction_labelled_per_batch is not None:
			bool_labelled = Y_semi.numpy().sum(1) == 1
			self.labelled_idxs = np.nonzero(bool_labelled)[0]
			self.unlabelled_idxs = np.where(bool_labelled==0)[0]
			np.random.shuffle(self.labelled_idxs)
			np.random.shuffle(self.unlabelled_idxs)
			self.N_labelled = int(self.batch_size*self.fraction_labelled_per_batch)
			self.N_unlabelled = self.batch_size - self.N_labelled
			### check if number of labels are enough, if not repeat labels
			if self.labelled_idxs.shape[0]<self.N_labelled:
				fac = np.ceil(self.N_labelled / self.labelled_idxs.shape[0])
				self.labelled_idxs = self.labelled_idxs.repeat(fac)
			

	def next(self):
		if self.fraction_labelled_per_batch is None:
			if self.curr_idx+self.batch_size >= self.N:
				self.curr_idx=0
				if not self.loop:
					return None
			ret = self.idxs[self.curr_idx:self.curr_idx+self.batch_size]
			self.curr_idx+=self.batch_size
			return ret
		else:
			# WARNING: never terminate (i.e. return None)
			np.random.shuffle(self.labelled_idxs)
			np.random.shuffle(self.unlabelled_idxs)
			return np.array(list(self.labelled_idxs[:self.N_labelled])+list(self.unlabelled_idxs[:self.N_unlabelled]))


