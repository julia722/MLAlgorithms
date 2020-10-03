# Julia Kim ML Assignment 1 (decisionStump.py)

import sys, numpy as np, csv, copy

# reads in data
def read_data(n):
    data = list()
    with open(str(sys.argv[n])) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader)
        for line in tsvreader:
            data.append(line)
    data = np.array(data)
    return data

# returns attributes that data is being split on
def get_attributes(data, split_index):
    attributes = set()
    for row in data:
        if row[split_index] not in attributes:
            attributes.add(row[split_index])
    return attributes

# returns classifying labels
def get_labels(data):
    labels = set()
    for row in data:
        if row[-1] not in labels:
            labels.add(row[-1])
    return labels
                
# splits data based on given attribute
def partition(data, split_index, left_attribute, right_attribute):
    left_data = list()
    right_data = list()
    for row in data:
        if row[split_index] == left_attribute:
            left_data.append(row)
        else:
            right_data.append(row)
    return left_data, right_data

# returns majority vote
def majority_vote(data_subset, labels):
    label_one = labels.pop()
    if len(labels) > 0: label_two = labels.pop()
    else: label_two = ""
    count = 0
    for row in data_subset:
        if row[-1] == label_one: count += 1
        elif row[-1] == label_two: count -= 1

    if count > 0: return label_one
    else: return label_two

# use decision tree structure to test
# assumed to be binary
# returns error
def test(data, left_attribute, left_label, right_attribute, right_label, split_index, n):
    error = 0
    with open(str(sys.argv[n]), "w") as outputFile:
        for line in data:
            label = ""
            if line[split_index] == left_attribute:
                label = left_label
            elif line[split_index] == right_attribute:
                label = right_label
            if line[-1] != label: error += 1
            outputFile.write(label + "\n")
    error = error / float(len(data))
    return error

def main():
    # train the decision tree stump
    train_data = read_data(1)
    test_data = read_data(2)
    split_index = int(sys.argv[3])
    attributes = get_attributes(train_data, split_index)
    labels = get_labels(train_data)
    left_attribute = attributes.pop()
    if len(attributes) > 0: right_attribute = attributes.pop()
    else: right_attribute = ""

    # partition data based on splitting attribute
    left_data, right_data = partition(train_data, split_index, left_attribute, right_attribute)
    
    # perform a majority vote for left and right data
    left_label = majority_vote(left_data, copy.deepcopy(labels))
    right_label = majority_vote(right_data, copy.deepcopy(labels))

    train_error = test(train_data, left_attribute, left_label, right_attribute, right_label, split_index, 4)
    test_error = test(test_data, left_attribute, left_label, right_attribute, right_label, split_index, 5)

    # metrics
    with open(str(sys.argv[6]), "w") as outputFile:
        outputFile.write("error(train): " + str(train_error) + "\n")
        outputFile.write("error(test): " + str(test_error))

if __name__ == '__main__':
    main()