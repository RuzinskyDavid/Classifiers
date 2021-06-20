from future.utils import iteritems
import numpy as np
from queue import PriorityQueue
#from sortedcontainers import SortedList
    
class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    def predict(self, X):
        y = np.zeros(len(X))
        y = []
        for i in range(len(X)):
            y.append([])
        for i, x in enumerate(X):
            #sl = SortedList()
            pq = PriorityQueue(maxsize = self.k)
            for j, xt in enumerate(self.X):
                diff = x - xt
                d = diff.dot(diff)
                if len(pq.queue) < self.k:
                    pq.put( (-d, self.Y[j]) )
                    
                else:
                    for item in pq.queue:
                        if -d > item[0]:
                            pq.get()
                            pq.put( (-d, self.Y[j]) )
                        break;

            votes = {}
            sl = []
            while not pq.empty():
                sl.append(pq.get())
            for _, v in sl:
                votes[v] = votes.get(v,0) + 1
            max_votes = 0
            max_votes_class = -1
            for v,count in iteritems(votes):
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class
        return y

    def score(self, X, Y, Z = None):
        if Z is None:
            Z = self.predict(X)
        
        correct = 0
        for i in range(len(Y)):
            #print(int(Z[i]), Y[i])
            if Z[i] == Y[i]:
                correct += 1
        
        #print(type(Z[0]), type(Y[0]))
        return correct / len(Y)