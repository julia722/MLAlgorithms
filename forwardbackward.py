# Julia Kim jmkim2 (forwardbackward.py)
import sys, csv
import collections
import numpy as np
import math

class HMM(object):
    def __init__(self, output):
        self.output = str(output)
    
    # use log sum trick to prevent underflow
    def log_sum(self, v1, v2):
        if v1 == 0.0: return v2
        elif v1 > v2: return v1 + math.log(1 + math.exp(v2 - v1))
        else: return v2 + math.log(1 + math.exp(v1 - v2))

    # performs forward pass of HMM
    def forward(self, alpha, prior, emit, trans, words):
        for j in range(len(alpha[0])):
            alpha[0][j] = prior[j][0] + emit[j][words[0]]
        for t in range(1, len(alpha)):
            for j in range(len(alpha[0])):
                sum = 0.0
                for k in range(len(alpha[0])):
                    sum = self.log_sum(sum, alpha[t-1][k] + trans[k][j])
                alpha[t][j] = emit[j][words[t]] + sum
        return alpha

    # performs backward pass of HMM
    def backward(self, beta, emit, trans, words):
        for j in range(len(beta[0])):
            beta[-1][j] = math.log(1)
        for t in range(len(beta)-2, -1, -1):
            for j in range(len(beta[0])):
                sum = 0.0
                for k in range(len(beta[0])):
                    sum = self.log_sum(sum, emit[k][words[t+1]] + 
                                       beta[t+1][k] + trans[j][k])
                beta[t][j] = sum

    # predicts label using backwards and forwards passes
    def predict(self, valid_words, valid_tags, prior, emit, trans, words, tags,
                index_to_tag, data):
        file = open(str(self.output), "w")
        correct = 0.0; total = 0.0; likelihood = 0.0
        for i in range(len(valid_words)):
            alpha = np.zeros(shape=(len(valid_words[i]), len(tags)))
            beta = np.zeros(shape=(len(valid_words[i]), len(tags)))
            # forward and backward passes
            self.forward(alpha, prior, emit, trans, valid_words[i])
            self.backward(beta, emit, trans, valid_words[i])
            # predicting
            output = ""
            for t in range(len(alpha)):
                prediction = None; max = None; sum = 0.0
                for j in range(len(alpha[0])):
                    if max == None or alpha[t][j] + beta[t][j] >= max:
                        max = alpha[t][j] + beta[t][j]
                        prediction = tags[j]
                    sum = self.log_sum(sum, alpha[-1][j]) # likelihood
                # calculating error
                if index_to_tag[prediction] == valid_tags[i][t]: correct += 1
                total += 1
                output += data[i][t] + "_" + str(prediction) + " "
            file.write(output[:-1] + "\n") # no trailing space
            likelihood += sum
        file.close()
        # return average log-likelihood and accuracy
        return likelihood / len(data), correct / total
        
# given a file path, returns data partitioned into words and tags
def read_data(path, words, tags):
    valid_words = []; valid_tags = []; data = []
    file = open(str(path), "rt")
    reader = file.readlines()
    file.close()
    for line in reader:
        word_lst = []; tag_lst = []; word_data = []
        line = line.replace("\n", "").split(" ")
        for entry in line:
            word, tag = entry.split("_")
            word_data.append(word)
            word_lst.append(words[word]); tag_lst.append(tags[tag])
        valid_words.append(word_lst); valid_tags.append(tag_lst)
        data.append(word_data)
    return valid_words, valid_tags, data

# returns a dictionary mapping indices to elements
def get_dict(path):
    d = collections.defaultdict(int)
    file = open(str(path), "rt")
    reader = file.readlines()
    file.close()
    i = 0
    for line in reader:
        line = line.replace("\n", "")
        d[line] = i
        i += 1
    return d

# reads in transition, emission, and prior prob matrices
def read_matrix(path):
    data = []
    with open(str(path)) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter = " ")
            for line in tsvreader:
                data.append(line)
    new_data = np.array(data, float)
    # log-space calculations
    return np.log(new_data)

def read_txt(path):
    data = []
    with open(str(path)) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter = " ")
            for line in tsvreader:
                data.append(line[0])
    return data

def main():
    # read files in
    index_to_word = get_dict(sys.argv[2]); index_to_tag = get_dict(sys.argv[3])
    wordlist = read_txt(sys.argv[2])
    taglist = read_txt(sys.argv[3])
    valid_word, valid_tag, data = read_data(sys.argv[1], index_to_word,
                                            index_to_tag)
    prior = read_matrix(sys.argv[4])
    emit = read_matrix(sys.argv[5])
    trans = read_matrix(sys.argv[6])
    # forward backward algorithm
    hmm = HMM(sys.argv[7])
    likelihood, err = hmm.predict(valid_word, valid_tag, prior, emit, trans, 
                                  wordlist, taglist, index_to_tag, data)
    # metrics out
    metrics = open(str(sys.argv[8]), "w")
    metrics.write("Average Log-Likelihood: " + str(likelihood) + "\n")
    metrics.write("Accuracy: " + str(err) + "\n")
    metrics.close()
    print('done!')

if __name__ == '__main__':
    main()