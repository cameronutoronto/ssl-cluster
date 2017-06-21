import numpy as np


class MiniBatcher(object):
	def __init__(self, N, batch_size=32, loop=True):
		self.N = N
		self.batch_size=batch_size
		self.loop = loop
		self.idxs = np.arange(N)
		np.random.shuffle(self.idxs)
		self.curr_idx = 0

	def next(self):
		if self.curr_idx+self.batch_size >= self.N:
			self.curr_idx=0
			if not self.loop:
				return None
		ret = self.idxs[self.curr_idx:self.curr_idx+self.batch_size]
		self.curr_idx+=self.batch_size
		return ret


