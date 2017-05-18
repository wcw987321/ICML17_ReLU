import numpy as np
import math
from utils import compute_gradient
import matplotlib.pylab as plt

d = 4 # the number of dimensions
n_node = 3 # the number of hidden nodes in the network
nIter = 100000 # the number of iterations during gradient descent
eta = 0.001 # learning rate
eps = 0.001
PI = 3.1415927

def compute_err(errs, ws_tmp):
    err = 0
    for i in range(n_node):
        for j in range(n_node):
            err += g(ws_tmp[:,i],ws_tmp[:,j]) - 2 * g(ws_tmp[:,i],ws_star[:,j]) + g(ws_star[:,i],ws_star[:,j])
    err = err / n_node / n_node
    errs.append(err)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def g(a, b):
    val = 1 / 2.0 / PI
    val *= np.linalg.norm(a)
    val *= np.linalg.norm(b)
    theta = angle_between(a, b)
    val *= math.sin(theta) + (PI - theta) * math.cos(theta)
    return val

########################## begin of main part ##################################3

ws_star = np.random.randn(d, n_node) # parameters of the "real" network
#print 'ws_star: ', ws_star

#ws_star[:n_node,:n_node] = np.eye(n_node)
ws_star_norm = np.linalg.norm(ws_star, axis=0) # norms of vectors(each vector is a parameter of a node in "real" network)
#print 'ws_star_norm: ', ws_star_norm

#ratio = 0.5 # how far will the init point be from the "real" parameter
#ws0 = np.copy(ws_star) + np.random.randn(d, n_node) * ratio # init point
ws0 = - np.copy(ws_star) + np.random.randn(d, n_node)
#print 'ws0: ', ws0

ws = np.copy(ws0) # init point
ws_all = np.zeros((d, n_node, nIter)) # records of gd

for t in range(nIter):
    ws_all[:,:,t] = ws # store in the record
    grad = compute_gradient(ws, ws_star) # compute the gradient
    ws += eta * grad # gradient descent
#print 'ws_all: ', ws_all
#print ws_star
#print ws_all
#print ws_all - ws_star[:,:,None]

errs = []

for t in range(nIter):
    ws_tmp = ws_all[:,:,t]
    compute_err(errs, ws_tmp)

#errs_per_w = np.sum(np.power(ws_all - ws_star[:,:,None], 2), axis=0) # loss function
#print 'errs_per_w: ', errs_per_w
#errs = np.sum(errs_per_w, axis=0) # loss
#print 'errs: ', errs
#plt.plot(errs_per_w.T)
#plt.show()
#plt.plot(errs)
#plt.show()
print(errs[0], errs[-1]) # the maximum loss and minimum loss
if errs[-1] >= eps:
    print 'ws_star: ', ws_star
    print 'ws_star_norm: ', ws_star_norm
    print 'ws0: ', ws0
    print 'ws_last: ', ws_all[:,:,nIter-1]
    print 'eta: ', eta
    print 'grad: ', grad
    print '*********************************************************************8'
#print(errs[-1]) # the minimum loss

