# Julia Kim ML Assignment 2 (decisionTree.py)

from __future__ import print_function
import sys
import getopt, csv, copy
import numpy as np
from math import log as log

# Leaf node containing classifying decision
class Leaf:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.vote = vote(data) # Final classifying decision

# Inner node containing branches to other nodes
class InnerNode:
    def __init__(self, attribute, leftBranch, rightBranch, leftLabel, rightLabel, data):
        self.attr = attribute
        leftData, rightData = partition(leftLabel, attribute, data)

        self.left = leftBranch
        self.leftLabel = leftLabel
        self.leftDict = majorityDict(leftData)

        self.right = rightBranch
        self.rightLabel = rightLabel
        self.rightDict = majorityDict(rightData)

# Returns dictionary with breakdown of labels
def majorityDict(data):
    majority = dict()
    for line in data[1:]:
        majority[line[-1]] = majority.get(line[-1], 0) + 1
    return majority

# Assigns label based on majority vote
# Breaks ties lexicographically
def vote(data):
    majority = majorityDict(data)
    labels = getLabel(data[0][-1], data)
    label_one, label_two = labels[0], labels[1]
    count = majority[label_one]
    if label_two != "": 
        count -= majority[label_two]
    if count > 0: return label_one
    elif count < 0: return label_two
    else: return min(label_one, label_two)

# Returns labels under attributes (ex. yes/no, democrat/republican)
def getLabel(attr, data):
    i = np.where(data[0] == attr)[0][0]
    labels = list(set(data[1:,i]))
    if len(labels) == 1: labels.append("")
    return labels

# Partitions data based on a certain attribute
def partition(option, attr, data):
    i = np.where(data[0] == attr)[0][0]
    left_data, right_data = list(), list()
    for row in data[1:]:
        new_row = np.delete(row, i)
        if row[i] == option:
            left_data.append(new_row)
        else:
            right_data.append(new_row)
    left_data.insert(0, np.delete(data[0], i))
    right_data.insert(0, np.delete(data[0], i))
    return np.array(left_data), np.array(right_data)

# Calculates entropy before splitting
def entropy(data):
    count = 0
    l = vote(data)
    for line in data[1:]:
        if line[-1] == l: count += 1
    p = count / float(len(data)-1)
    if p in [0, 1]: return 0
    return -p * log(p, 2) - (1 - p) * log(1 - p, 2)

# Calculates mutual information
def mutualInfo(attr, data):
    total = entropy(data)
    leftLabel, rightLabel = getLabel(attr, data)
    leftData, rightData = partition(leftLabel, attr, data)
    if len(leftData) <= 1 or len(rightData) <= 1: return 0
    leftEntropy = entropy(leftData)
    rightEntropy = entropy(rightData)
    p = len(leftData) / float(len(data))
    return total - p * leftEntropy - (1 - p) * rightEntropy

def trainTree(currDepth, maxDepth, data):
    if (currDepth >= maxDepth): return Leaf(data) # stop splitting

    gain, attr = 0, ""
    for attribute in data[0][:-1]:
        if mutualInfo(attribute, data) > gain:
            gain = mutualInfo(attribute, data)
            attr = attribute

    if (gain <= 0): return Leaf(data)

    # find two labels of the attribute (yes/no, right/wrong, etc.)
    leftLabel, rightLabel = getLabel(attr, data)
    leftData, rightData = partition(leftLabel, attr, data)
    leftBranch = trainTree(currDepth+1, maxDepth, leftData)
    rightBranch = trainTree(currDepth+1, maxDepth, rightData)

    return InnerNode(attr, leftBranch, rightBranch, leftLabel, rightLabel, data)

# Function for pretty printing the tree
def printTree(node, data):
    print(majorityDict(data))
    printHelper(node)

# Helper function for pretty printing the tree
def printHelper(node, space="| "):

    if isinstance(node, Leaf):
        print(" ", end="\n")
        return
    
    if isinstance(node.left, Leaf): leftSpace = ""
    else: leftSpace = "\n"

    if isinstance(node.right, Leaf): rightSpace = ""
    else: rightSpace = "\n"

    print(space + str(node.attr), end="")
    print(" = " + str(node.leftLabel) + " " + str(node.leftDict), end=leftSpace)
    printHelper(node.left, space + "| ")
    print(space + str(node.attr), end="")
    print(" = " + str(node.rightLabel) + " " + str(node.rightDict), end=rightSpace)
    printHelper(node.right, space + "| ")

# Reads data and returns a np array of data
def readData(n):
    data = list()
    with open(str(sys.argv[n])) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            data.append(line)
    data = np.array(data)
    return data

# Uses trained tree to make predictions on data
# Returns error
def test(tree, data):
    error = 0
    output = list()
    for line in range(1, len(data)):
        res, prediction = parse(tree, data[line], data)
        error += res
        output.append(prediction)
    return (error / float(len(data) - 1), output)

# Recursive function that traverses tree
# Returns 0 if tree's decision matches the data's decision; 1 otherwise
# Also returns the prediction made by the tree
def parse(tree, line, data):
    if isinstance(tree, Leaf):
        prediction = tree.vote
        if prediction != line[-1]: return 1, prediction
        else: return 0, prediction
    else:
        i = np.where(data[0] == tree.attr)[0][0]
        if line[i] == tree.leftLabel:
            return parse(tree.left, line, data)
        else:
            return parse(tree.right, line, data)

def main():
    # Build tree using train data
    trainData = readData(1)
    tree = trainTree(0, int(sys.argv[3]), trainData)
    printTree(tree, trainData) # Print tree

    # Evaluate train data using tree
    trainError, trainOutput = test(tree, trainData)

    # Evaluate test data using tree
    testData = readData(2)
    testError, testOutput = test(tree, testData)
    
    # Metrics (metrics out)
    with open(str(sys.argv[6]), "w") as outputFile:
        outputFile.write("error(train): " + str(trainError) + "\n")
        outputFile.write("error(test): " + str(testError) + "\n")

    # Labels (labels out)
    with open(str(sys.argv[4]), "w") as outputFile:
        for line in trainOutput:
            outputFile.write(line + "\n")

    with open(str(sys.argv[5]), "w") as outputFile:
        for line in testOutput:
            outputFile.write(line + "\n")

if __name__ == '__main__':
    main()