# Julia Kim ML Assignment 2 (inspection.py)
import sys
import csv
from math import log as log

# reads in data
def readData(n):
    data = list()
    with open(str(sys.argv[n])) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader)
        for line in tsvreader:
            data.append(line[-1])
    return data

# returns classifying labels
def getLabels(data):
    return set(data)

# returns entropy
def entropy(data):
    labelOne = getLabels(data).pop()
    p = data.count(labelOne) / float(len(data))
    return -p * log(p, 2) - (1 - p) * log(1 - p, 2)

# returns majority vote
def majorityVote(data, labels):
    labelOne = labels.pop()
    count = data.count(labelOne)
    if len(data) - count < count: return labelOne
    else: return labels.pop()

# calculates error based on majority vote
def majorityError(data):
    label = majorityVote(data, getLabels(data))
    error = 0
    for entry in data:
        if entry != label: error += 1
    return error / float(len(data))

def main():
    data = readData(1)
    # metrics
    with open(str(sys.argv[2]), "w") as outputFile:
        outputFile.write("entropy: " + str(entropy(data)) + "\n")
        outputFile.write("error: " + str(majorityError(data)))

if __name__ == '__main__':
    main()