"""
Main file to run contituent functions

author: Shashank Srivastava, PhD

"""
# Imports
from svd import decomposition
import numpy as np

# Create data
X = np.random.randn(50, 9)
#print(X)

# Test SVD
u, s, v = decomposition(X).svd('economy')
#print('u = {}'.format(u))
#print('Sigma = {}'.format(s))
#print('V = {}'.format(v))

# r-rank truncated
r = 3
ur, sr, vr = decomposition(X).truncated_svd(r)
print(ur)
print(sr)
print(vr)