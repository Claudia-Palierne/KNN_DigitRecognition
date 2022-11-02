"""
Created on Sun Oct 24 12:22:28 2021

@author: Klowdz
"""

# -*- coding: utf-8 -*-

from functools import cache
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

#Take a subset of the MNIST data set
idx = np.random.RandomState(0).choice(70000, 11000)
#Split into train-test set
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

def calcDist(pop,unit) :
    ''''
    Calculate the distance with norm L2 between a vector (unit) and all the others vector in the population (pop).

    Input : unit - a vector
            pop - numpy array, each row represent a vector

    Output : an array with all the distance between unit and each vector of pop.
    '''

    rslt = np.zeros(len(pop))
    for i in range(len(pop)) :
        rslt[i] = np.linalg.norm(pop[i]-unit, 2)
    return rslt

def getLabel(train_label, idx) :
    '''
    Get the label of the given idx.

    Input : idx - an array of index.
            train_label - a numoy array containing the label of all the vectors in the population.

    Output : the label with the most occurence.
    '''
    ref = np.zeros(len(idx))
    for i in range(len(idx)) :
        ref[i]= train_label[idx[i]]
    hist = np.bincount(ref.astype(int))
    return np.argmax(hist)

def kNN(train_img, train_label, image, k, n) :
    '''
    Calculate k-nearest neighbors on n samples.
    
    Input : train_img - a numpy array countaining the image of the population
            train_label - a numpy array countaining the label of the population
            image - the image onwhich we want to predict the label
            n - the number of samples
            k - the hyperparameter of KNN
    
    Output : the predicted label of image.
    '''
    train_img = train_img[:n]
    train_label = train_label[:n]
    #Calculate the distances between image and the population.
    dist = calcDist(train_img,image)
    #take the k closest vectors to image.
    k_idx = np.argsort(dist)[:k]
    #Output the most probable label i.e. with the most occurence betweeen the k closest vector.
    label = getLabel(train_label, k_idx)
    return label


def accuracy(train, train_labels, test, test_labels,  k, n) :
    '''
    Calculate the accuracy of the test set.

    Input : train - an numpy array of all the images in the population
            train_labels - an numpy array of all the labels in the population
            test - an numpy array of all the images in the test set
            test_labels - an numpy array of all the labels in the test set
            k - the hyperparameter of KNN
            n - the number of samples
    
    Output : the accuracy of the test using KNN with n samples.
    '''
    rslt = 0
    for i in range(len(test)) :
        if ( int(test_labels[i]) != kNN(train, train_labels, test[i], k, n) ) :
            rslt += 1
    rslt /= len(test)
    return 1 - rslt

#Test of 10-NN with 1000 samples 
n = 1000
k = 10
acc = accuracy(train, train_labels, test, test_labels, k, n)
print("KNN with k=" + str(k) + " and n=" + str(n) + ". The accuracy is : " + str(acc))

#Let's find the best value for the hyper parameter k
n = 1000
#Build a grid for k :
k_accurate = np.zeros(100)
for i in range(1,101) :
    k_accurate[i-1] = accuracy(train, train_labels, test, test_labels, i, n)

plt.title("K-NN accuracy as a function of k") 
plt.xlabel("k") 
plt.ylabel("accuracy of K-NN (zero-one loss)") 
plt.plot(np.arange(1,101),k_accurate) 
plt.show()

print("Discuss the result : The best value for seems to be k = 1. Indede since some numbers look alike when they are handwritten, it would not be a good idea to add variance i.e. increase k.")
#Different number when they are hand written number looks sometimes similar.
#Thus the way people wrote number, might induce some errors in the labels given by our algorithm.
#Then, if we increase the k, we might get the wrong label.

#Let's find the best value for n :
k = 1
#Build the grid for n
n_accurate = np.zeros(50)
for i in range(1,51) :
    n_accurate[i-1] = accuracy(train, train_labels, test, test_labels, k, i*100)

plt.title("K-NN accuracy as a function of k") 
plt.xlabel("k") 
plt.ylabel("accuracy of K-NN (zero-one loss)") 
plt.plot(np.arange(100,5100,100),n_accurate) 
plt.show()

print("Discuss the result : We can see that when n increase, the accuracy increase as well.")
#It seems logical. In the sens that we have more training picture then we have more chance to have close image.
#Increasing the training set, increase as well the chance to have the same small inexactitude in the way the numbers were wrotten.





