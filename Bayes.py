from future.utils import iteritems
import numpy as np
from scipy.stats import multivariate_normal as mvn

class Bayes(object):
    def fit(self, X, Y, smoothing=1e-2):
        N, D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            indices = [i for i, x in enumerate(Y) if x == c]
            #print(indices)
            #current_x = X[indices[0]:indices[-1]]               ###!!!!!!!!!!!!!!
            current_x = X[indices, :]                           ### much better
            self.gaussians[c] = {
                'mean': current_x.mean(axis = 0),
                'var': np.cov(current_x.T) + np.eye(D)*smoothing,
            }
            self.priors[c] = float(len(indices)) / len(Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in iteritems(self.gaussians):
            index = list(self.gaussians.keys()).index(c)
            mean, var = g['mean'], g['var']
            P[:, index] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
        P = np.argmax(P, axis = 1)
        key_list = list(self.gaussians.keys())
        Z = []
        for i in range(N):
            Z.append(key_list[P[i]])
        return Z
    
    def score(self, X, Y, Z = None):
        if Z is None:
            Z = self.predict(X)
        correct = 0
        for i in range(len(Y)):
            #print(Z[i], Y[i])
            if Z[i] == Y[i]:
                correct += 1
        
        return correct / len(Y)