"""
Created on Sun Oct 24 12:22:28 2021

@author: Klowdz
"""

# -*- coding: utf-8 -*-

###### EX 1 - INTRO TO ML ########

### Part 3

import numpy as np
from matplotlib import pyplot as plt

N = 200000 #rows
n = 20 #columns
p = 0.5

mat = np.random.binomial(1,p,(N,n))

mean = mat.sum(axis = 1)/n #empirical mean

e = np.linspace(0,1, 50)
prob = np.zeros(len(e))

for i in range(len(e)) :
    for j in range(len(mean)) :
        if (abs(mean[j]-p) > e[i]) :
            prob[i] += 1
    prob[i] = prob[i]/N #empirical probability

plt.title("Visualizing Hoeffding bound") 
plt.xlabel("epsilon") 
plt.ylabel("P(|X-p|>e)") 
plt.plot(e,prob, 'b') #plot empirical probability
plt.plot(e,2*np.exp(-2*n*e**2),'r')
plt.show()



##### Programming part 

#Part a
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

def calcDist(pop,unit) :
    rslt = np.zeros(len(pop))
    for i in range(len(pop)) :
        rslt[i] = np.linalg.norm(pop[i]-unit, 2)
    return rslt

def getLabel(train_label, idx) :
    ref = np.zeros(len(idx))
    for i in range(len(idx)) :
        ref[i]= train_label[idx[i]]
    hist = np.bincount(ref.astype(int))
    return np.argmax(hist)

def kNN(train_img, train_label, image, k, n) :
    train_img = train_img[:n]
    train_label = train_label[:n]
    dist = calcDist(train_img,image)
    k_idx = np.argsort(dist)[:k]
    label = getLabel(train_label, k_idx)
    return label


def accuracy(train, train_labels, test, test_labels,  k, n) :
    rslt = 0
    for i in range(len(test)) :
        if ( int(test_labels[i]) != kNN(train, train_labels, test[i], k, n) ) :
            rslt += 1
    rslt /= len(test)
    return 1 - rslt

# Part b
n = 1000
k = 10
accuracy(train, train_labels, test, test_labels, k, n)

#Part c 
n = 1000
k_accurate = np.zeros(100)
for i in range(1,101) :
    k_accurate[i] = accuracy(train, train_labels, test, test_labels, i, n)

plt.title("K-NN accuracy as a function of k") 
plt.xlabel("k") 
plt.ylabel("accuracy of K-NN (zero-one loss)") 
plt.plot(np.arange(1,101),k_accurate) 
plt.show()

#Discuss the result : The best k seems to be k = 1.
#Different number when they are hand written number looks sometimes similar.
#Thus the way people wrote number, might induce some errors in the labels given by our algorithm.
#Then, if we increase the k, we might get the wrong label.

#Part d
k = 1
n_accurate = np.zeros(50)
for i in range(1,51) :
    n_accurate[i] = accuracy(train, train_labels, test, test_labels, k, i*100)

plt.title("K-NN accuracy as a function of k") 
plt.xlabel("k") 
plt.ylabel("accuracy of K-NN (zero-one loss)") 
plt.plot(np.arange(100,5100,100),n_accurate) 
plt.show()

#Discuss the result : We can see that when n increase, the accuracy increase as well.
#It seems logical. In the sens that we have more training picture then we have more chance to have close image.
#Increasing the training set, increase as well the chance to have the same small inexactitude in the way the numbers were wrotten.





