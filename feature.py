# Julia Kim ML Assignment 4 (feature.py)
from __future__ import print_function
import sys, csv


# input: label and attribute (set of words separated by whitespace)
# output: list of labels (0 and 1)
# output: 2D list of data separated by line then word
def read_data(path):
    data = list()
    label = list()
    with open(path) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            label.append(line[0])
            data.append(line[1].split(" "))
    return label, data

# input: words and indexes separated by whitespace
# output: dictionary with index mapped to word
def read_dict(path):
    f = open(path, "r")
    d = dict()
    for line in f.readlines():
        word, index = line.split(" ")
        d[word] = index[:-1]
    f.close()
    return d

class Model(object):
    def __init__(self, train, valid, test,
                 train_path, valid_path, test_path, dict, flag):
        self.train = train
        self.valid = valid
        self.test = test
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.dict = dict
        self.flag = flag
    
    def output(self):
        self.write(self.train_path, self.train)
        self.write(self.valid_path, self.valid)
        self.write(self.test_path, self.test)
        print("done!")    
    
    # helper function for output
    # writes formatted data to provided path
    def write(self, path, data):
        if self.flag == "1":
            label, fdata = self.model_one(self.dict, data)
        else: # self.flag == 2
            label, fdata = self.model_two(self.dict, data)

        with open(path, "w") as outputFile:
            for i in range(len(label)):
                line = label[i]
                for index in fdata[i]:
                    line += "\t" + index + ":1"
                outputFile.write(line + "\n")

    # sparse implementation for model one
    # adds index of word to list if word is in dictionary
    def model_one(self, dict, data):
        label, data = data
        fdata = list()
        for line in data:
            new_line = set()
            for word in set(line):
                if word in dict.keys():
                    new_line.add(dict[word])
            fdata.append(new_line)
        return label, fdata

    # sparse implementation for model two
    # trimming threshold is 4
    def model_two(self, dict, data):
        label, data = data
        fdata = list()
        for line in data:
            seen = set()
            for word in line:
                if word in dict.keys() and line.count(word) < 4:
                    seen.add(dict[word])
            fdata.append(seen)
        return label, fdata

def main():
    train = read_data(sys.argv[1])
    valid = read_data(sys.argv[2])
    test = read_data(sys.argv[3])
    dict = read_dict(sys.argv[4])
    flag = sys.argv[8]

    m = Model(train, valid, test,
              sys.argv[5], sys.argv[6], sys.argv[7], dict, flag)
    m.output()

if __name__ == '__main__':
    main()

