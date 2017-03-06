"""
FastICA example
Developed originally by Aapo Hyvärinen
Non-gaussianity is a proxy for statistical independence
Help from "Independent Component Analysis: Algorithms and Applications" by Hyvärinen, A. and Oja, E. (2000)
Help from "Independent Component Analysis" by Hyvärinen, A., Karhunen, J., & Oja, E. (2001)
Help from https://en.wikipedia.org/wiki/FastICA
"""

import numpy as np
from scipy import signal, cluster
import scipy
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

n_samples = 2000
time = np.linspace(0,8,n_samples)
m = 5 # number of components

# create sources

source_1 = np.sin(2*time)
source_2 = np.sign(np.sin(3*time))
source_3 = signal.sawtooth(2 * np.pi * time)

S = np.c_[source_1,source_2,source_3] #sources
S += 0.2 * np.random.normal(size=S.shape) # add some noise

S /= S.std(axis=0) # standardize/center

mixing = np.random.randn(m,3) # create mixing matrix

# my FastICA example

X1 = np.dot(S, mixing.T) # mixed signals for my fastICA
X1 = scipy.cluster.vq.whiten(X1) # pre-whitening data

# to measure Non-gaussianity, non-linearity functions are used:

def G(u):
    # Non-linearity function derivative
    res = np.tanh(u)
    return res

def DG(u):
    # Non-linearity function second derivative
    k = np.tanh(u)
    res = np.ones(np.shape(k)) - (k * k)
    return res

O = np.ones((m,1))

weights = np.ones((2000,3)) # create matrix to hold weights

for p in range(0,3): # for each source
    W = np.random.randn(n_samples,1) # create a random weight vector
    W = W / np.linalg.norm(W,2) # normalize
    W0 = np.random.randn(n_samples,1) # create another random weight vector for comparison
    W0 = W0 / np.linalg.norm(W0,2) # normalize
    while abs(abs(np.dot(W0.T,W)) - 1) > 0.0001:
        # check if weight vector converges (the length of the difference vector is less than a certain threshold)
        W0 = W[:,:] # copy W into W0
        one = (np.dot(X1,G(np.dot(W.T,X1)).T)) / m # see pg. 14 pg of Hyvärinen (2000) for explanation
        two = (np.dot(DG(np.dot(W.T,X1)),O)*W) / m
        W = one - two
        for i in range(0,p):
            W = W[:,:] - (np.dot(W.T,weights[:,i:i+1])*weights[:,i:i+1]) # Deflationary orthogonalization
        W = W/np.linalg.norm(W,2) # normalize
    weights[:,p:p+1] = W[:,:]

X = np.dot(S, mixing.T) # mixed signals for sklearn fastICA

ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)
A_ = ica.mixing_

# visualization

plt.figure()
models = [X, S, S_, weights]
names = ["Observations Signals", "True Sources","Sklearn ICA Signals", "My FastICA Signals"]
colors = ["c","m","y"]

for ii, (model,name) in enumerate(zip(models,names),1):
    plt.subplot(4,1,ii)
    plt.title(name)
    for sig, color in zip(model.T,colors):
        plt.plot(sig,color=color)

plt.show()