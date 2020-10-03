
# Julia Kim jmkim2 (neuralnet.py)

from __future__ import print_function
import sys, csv
from math import exp as exp
import math
import numpy as np

def sigmoid(x):
    return np.exp(x) / (1.0 + np.exp(x))

def softmax(b):
    y_hat = np.zeros(len(b))
    y_hat_sum = sum(np.exp(b))
    for i in range(len(b)):
        y_hat[i] = np.exp(b[i]) / y_hat_sum
    return y_hat

# returns cross entropy loss
def loss(y_hat, y):
    return -1 * np.log(y_hat[int(y)])

def read_data(path):
    x, y = list(), list()
    with open(str(path)) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter=",")
        for line in tsvreader:
            y.append(int(line[0]))
            tmp = [0] + line[1:] # bias term
            M = len(tmp) # M = number of attributes
            x.append(tmp)
    x = np.array(x)
    y = np.array(y)
    return x.astype(np.float), y.astype(np.float), M

class Forward(object):
    def __init__(self, x, y, alpha, beta):
        self.x = np.array([x]).T
        self.y = y
        self.alpha = alpha
        self.beta = beta

    def nn_forward(self):
        a = self.linear_forward(self.alpha, self.x)
        z = self.sigmoid_forward(a)
        z = np.insert(z, 0, np.array([1]), 0)  
        b = self.linear_forward(self.beta, z)
        y_hat = np.array(self.softmax_forward(b)).T
        J = self.entropy_forward(self.y, y_hat)
        #print(a, z, b, y_hat, J)
        self.y_hat = y_hat
        self.J = J
        return self.x, a, z, b, y_hat, J
    
    def linear_forward(self, alpha, x):
        return np.dot(alpha, x)

    def sigmoid_forward(self, a):
        return sigmoid(a)

    def softmax_forward(self, b):
        return softmax(b)
    
    def entropy_forward(self, y, y_hat):
        return loss(y_hat, y)

class Backward(object):
    def __init__(self, x, y, alpha, beta, a, z, b, y_hat, J):
        self.x, self.y = x, y
        self.x[0] = 1
        self.alpha, self.beta = alpha, beta
        self.a = a
        self.z = z
        self.b = b
        self.y_hat = y_hat
        self.J = J
    
    def nn_backward(self):
        db = self.softmax_backward(self.y, self.y_hat)
        dbeta = np.dot(db, self.z.T)
        new_beta = np.delete(self.beta, 0, 1)
        
        dz = np.dot(db.T, new_beta)
        new_z = np.delete(self.z, 0, 0)
        da = dz.T * (new_z * (1 - new_z))
        dalpha = np.dot(da, self.x.T)
        return dalpha, dbeta

    def softmax_backward(self, y, y_hat):
        # print(y_hat)
        # print(y)
        y_hat[int(y)] -= 1
        return np.array([y_hat]).T

class SGD(object):
    def __init__(self, x, y, alpha, beta, num_epoch, lr, test_x, test_y):
        self.x = x
        self.y = y
        self.alpha = alpha
        self.beta = beta
        self.epoch = num_epoch
        self.lr = lr
        self.test_x = test_x
        self.test_y = test_y
    
    def mean_entropy(self, x, y, alpha, beta):
        entropy = 0.0
        for i in range(len(y)):
            f = Forward(x[i], y[i], alpha, beta)
            f.nn_forward()
            entropy += f.J
        return entropy / float(len(y))

    def sgd(self):
        train_entropy, test_entropy = list(), list()
        for epoch in range(self.epoch):
            for row in range(len(self.y)):
                f = Forward(self.x[row], self.y[row], self.alpha, self.beta)
                x, a, z, b, y_hat, J = f.nn_forward()
                b = Backward(x, self.y[row], self.alpha, self.beta, a, z, b, y_hat, J)
                g_alpha, g_beta = b.nn_backward()
                self.alpha -= self.lr * g_alpha
                self.beta -= self.lr * g_beta
            train_entropy.append(self.mean_entropy(self.x, self.y, self.alpha, self.beta))
            test_entropy.append(self.mean_entropy(self.test_x, self.test_y, self.alpha, self.beta))
        return self.alpha, self.beta, train_entropy, test_entropy
    
    def predict(self, x, y, alpha, beta, path):
        incorrect = 0
        entropy = []
        with open(str(path), "w") as outputFile:
            for row in range(len(y)):
                f = Forward(x[row], y[row], alpha, beta)
                f.nn_forward()
                prediction = np.argmax(f.y_hat)
                outputFile.write(str(prediction) + "\n")
                if prediction != y[row]: incorrect += 1.0
        return incorrect / len(y)


# initializes weights based on init_flag
# if flag = 1, weights are randomly initialized
# if flag = 2, weights are initialized to 0
def get_weight(init_flag, M, hidden_units):
    if init_flag == 1:
        alpha = np.random.uniform(-0.1, 0.1, (hidden_units, M))
        beta = np.random.uniform(-0.1, 0.1, (10, hidden_units + 1))
        alpha[:,0] = 0
        beta[:,0] = 0
    else: # init_flag == 2
        alpha = np.zeros(shape=(hidden_units, M))
        beta = np.zeros(shape=(10, hidden_units + 1))
    return alpha, beta

def main():
    train_x, train_y, M = read_data(sys.argv[1])
    test_x, test_y, M = read_data(sys.argv[2])
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    
    alpha, beta = get_weight(int(sys.argv[8]), M, int(sys.argv[7]))
    model = SGD(train_x, train_y, alpha, beta, int(sys.argv[6]), float(sys.argv[9]), test_x, test_y)
    alpha, beta, train_entropy, test_entropy = model.sgd()
    train_error = model.predict(train_x, train_y, alpha, beta, train_out)
    test_error = model.predict(test_x, test_y, alpha, beta, test_out)

    with open(str(sys.argv[5]), "w") as outputFile:
        for n in range(len(train_entropy)):
            outputFile.write("epoch=" + str(n+1) + " crossentropy(train): " + 
                             str(train_entropy[n]) + "\n")
            outputFile.write("epoch=" + str(n+1) + " crossentropy(validation): " + 
                             str(test_entropy[n]) + "\n")
        outputFile.write("error(train): " + str(train_error) + "\n")
        outputFile.write("error(validation): " + str(test_error) + "\n")
    print('done!')

if __name__ == '__main__':
    main()