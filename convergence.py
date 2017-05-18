import numpy as np
import math
from utils import compute_gradient
import matplotlib.pylab as plt

d = 3 # the number of dimensions
n_node = 2 # the number of hidden nodes in the network
nIter = 10000 # the number of iterations during gradient descent

ws_star = np.random.randn(d, n_node) # parameters of the "real" network
#print 'ws_star: ', ws_star

#ws_star[:n_node,:n_node] = np.eye(n_node)
ws_star_norm = np.linalg.norm(ws_star, axis=0) # norms of vectors(each vector is a parameter of a node in "real" network)
#print 'ws_star_norm: ', ws_star_norm

ratio = 0.5 # how far will the init point be from the "real" parameter
ws0 = np.copy(ws_star) + np.random.randn(d, n_node) * ratio # init point
#print 'ws0: ', ws0

ws = np.copy(ws0) # init point
ws_all = np.zeros((d, n_node, nIter)) # records of gd

eta = 0.01 # learning rate

for t in range(nIter):
    ws_all[:,:,t] = ws # store in the record
    grad = compute_gradient(ws, ws_star) # compute the gradient
    ws += eta * grad # gradient descent
#print 'ws_all: ', ws_all

errs_per_w = np.sum(np.power(ws_all - ws_star[:,:,None], 2), axis=0) # loss function
#print 'errs_per_w: ', errs_per_w
errs = np.sum(errs_per_w, axis=0) # loss
#print 'errs: ', errs
#plt.plot(errs_per_w.T)
#plt.show()
#plt.plot(errs)
#plt.show()
print(errs[0]) # the maximum loss
print(errs[-1]) # the minimum loss

