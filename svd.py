"""
Singular Value Decomposition (SVD)

This returns the Left Singaular Values, Singular Values,
and Right Singular Values of an input data matrix X with
two modes: 'full' and 'economy'

author: Shashank Srivastava, PhD (shas.srivastava@gmail.com)

"""

import numpy as np

class decomposition:

	def __init__(self, X):		# default mode set to full SVD
		self.X = X


	def svd(self, mode='full'):
		self.mode = mode

		if self.mode=='full':
			U, S, V = np.linalg.svd(self.X, full_matrices=True)

		elif self.mode=='economy':
			U, S, V = np.linalg.svd(self.X, full_matrices=False)
			
		else:
			# Exception handling
			pass

		return U, S, V


	def truncated_svd(self, rank):
		self.rank = rank

		if rank>np.min(np.shape(self.X)):
			# Exception handling
			pass
		else:
			U, S, V = np.linalg.svd(self.X, full_matrices=False)

		return U[:, :rank], S[:rank], V[:, :rank]