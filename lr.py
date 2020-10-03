# Julia Kim ML Assignment 4 (lr.py)
from __future__ import print_function
import sys, csv
from math import exp as exp
import numpy as np


# input: label and attributes
# output: list of labels (y)
# output: list of dictionaries with bias term folded in (x)
def read_data(path):
    x, y = list(), list()
    with open(path) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            y.append(int(line[0]))
            new_d = { int(i[:-2]) + 1 : 1.0 for i in line[1:] } # fold in bias as x_0
            new_d[0] = 1.0 # bias term
            x.append(new_d)
    return x, y
        
# Logistic regression model
class Model(object):
    def __init__(self, x, y, num_epoch, learning_rate):
        self.x = x
        self.y = y
        self.epoch = num_epoch
        self.lr = learning_rate
        self.theta = {0 : 0.0} # bias

    def sgd(self):
        for n in range(self.epoch):
            # N rows, M+1 features
            for row in range(len(self.y)):
                x, y = self.x[row], self.y[row]
                gradient = self.gradient(x, y, self.theta) # returns a scalar
                tmp = {0 : 0.0}
                for feat in x:
                    tmp[feat] = self.lr * gradient
                # update weights based on learning rate * gradient
                self.theta = self.add_matrix(self.theta, tmp)
        print('sgd complete')
        return self.theta
    
    # only sums elements where x != 0
    def dot_product(self, x, theta):
        sum = 0.0
        for i in x:
            if i in theta: sum += theta[i]
        return sum
    
    # add gradient to weights
    def add_matrix(self, theta, gradient):
        for key in gradient:
            theta[key] = theta.get(key, 0.0) + gradient[key]
        return theta

    # computes gradient given x_i, y_i, theta
    def gradient(self, x, y, theta):
        dot = self.dot_product(x, theta)
        # subtract sigmoid
        return y - 1.0 / (1.0 + exp(-dot))
    
    # returns error
    def test_model(self, x, y, file, weights):
        incorrect = 0.0
        with open(file, "w") as outputFile:
            for row in range(len(y)):
                dot = self.dot_product(x[row], weights)
                # compute sigmoid
                prob = 1.0 / (1.0 + exp(-dot))
                if prob < 0.5: label = 0
                else: label = 1
                if label != y[row]: incorrect += 1
                outputFile.write(str(label) + "\n")
        return incorrect / len(y)

def main():
    train_x, train_y = read_data(sys.argv[1])

    # only used to make estimations on held-out negative log-likelihood
    # valid_x, valid_y = read_data(sys.argv[2])
    test_x, test_y = read_data(sys.argv[3])
    train_out = str(sys.argv[5])
    test_out = str(sys.argv[6])
    num_epoch = int(sys.argv[8])
    learning_rate = 0.1
    
    # building the model and getting the weights
    sgd_model = Model(train_x, train_y, num_epoch, learning_rate)
    weights = sgd_model.sgd()

    # running the model and getting the error
    train_error = sgd_model.test_model(train_x, train_y, train_out, weights)
    test_error = sgd_model.test_model(test_x, test_y, test_out, weights)

    # metrics out
    with open(str(sys.argv[7]), "w") as outputFile:
        outputFile.write("error(train): " + str(train_error) + "\n")
        outputFile.write("error(test): " + str(test_error) + "\n")
    print('done!')

if __name__ == '__main__':
    main()
