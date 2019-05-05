#This module feeds data to the main experimental script. 
#For it to work properly, you should have a local copy of any of the data files
#named in this file. 

import numpy as np

#Produce randomly sampled data for the "Single Object" computer vision example.
class DataFeederOneObj():
    def __init__(self, n):
        self.n = n
        self.test_set = self.batch(2000)

    def batch(self, b):
        n = self.n
        re = np.random.randint(low=0, high=(n - (n/8))+1, size=b).astype('int')
        test = np.zeros((b, n))
        test[np.arange(b).astype('int'), [re + i for i in range(int(n/8))]] = 1.0
        return [test.copy(), test.copy()]

#Produce randomly sampled data for the "Double Object" computer vision example.     
class DataFeederTwoObj():
    def __init__(self, n):
        self.n = n
        self.test_set = self.batch(2000)

    def batch(self, b):
        toreturn = np.zeros((0, self.n))
        while toreturn.shape[0] < b:
            re = np.random.randint(low=0, high=(self.n - (self.n/8))+1, size=(b,2)).astype('int')
            test = np.zeros((b, self.n))
            for i in range(int(self.n/8)):
                for k in range(2):
                    test[np.arange(b).astype('int'), re[:,k] + i] = 1.0
            toreturn = np.vstack([toreturn, test])
        return [toreturn.copy()[:b, :],toreturn.copy()[:b, :]]

#Produce random MNIST training data.    
class DataFeederMNIST():
    def __init__(self):
        self.train_set = np.loadtxt("MNIST_train.csv",delimiter=',')
        self.test_data = np.loadtxt("MNIST_test.csv",delimiter=',')
        self.test_set = [self.test_data, self.test_data]

    def batch(self, b):
        re = np.random.randint(low=0, high=self.train_set.shape[0]-1, size=b).astype('int')
        test = self.train_set[re,:]
        return [test.copy(), test.copy()]

#Produce random CIFAR training data.     
class DataFeederCIFAR():
    def __init__(self):
        self.train_set = np.loadtxt("CIFAR_train.csv",delimiter=',')
        self.test_set = np.loadtxt("CIFAR_test.csv",delimiter=',')

    def batch(self, b):
        re = np.random.randint(low=0, high=self.train_set.shape[0]-1, size=b).astype('int')
        test = self.train_set[re,:]
        return [test.copy(), test.copy()]
