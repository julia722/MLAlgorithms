# Julia Kim jmkim2 (learnhmm.py)
import sys, csv
import collections

# read data
def read_data(path):
    pi = dict()
    emit = collections.defaultdict(dict)
    trans = collections.defaultdict(dict)
    f = open(str(path), "rt")
    data = f.readlines()
    f.close()
    for line in data:
        line = line.replace("\n", "").split(" ")
        for i in range(len(line)):
            word, label = line[i].split("_")
            emit[label][word] = emit[label].get(word, 1) + 1
            if i == 0:
                pi[label] = pi.get(label, 1) + 1
            if i != len(line) - 1:
                next_word, next_label = line[i+1].split("_")
                trans[label][next_label] = trans[label].get(next_label, 1) + 1
    return pi, emit, trans

# reads in txt files
def read_txt(path):
    lst = []
    file = open(str(path), "rt")
    reader = file.readlines()
    file.close()
    for line in reader:
        line = line.replace("\n", "")
        lst.append(line)
    return lst

# returns prior probabilities
def prior(d, path_out, tags):
    total = float(sum(d.values()) + (len(tags) - len(d)))
    file = open(str(path_out), "w")
    for tag in tags:
        if tag in d:
            file.write(str(d[tag] / total) + "\n")
        else: file.write(str(1 / total) + "\n")
    file.close()

# returns emission probabilities
def emit(d, path_out, tags, words):
    file = open(str(path_out), "w")
    for tag in tags:
        output = ""
        tmp = d[tag]
        total = float(sum(tmp.values()) + len(words) - len(tmp))
        for word in words:
            if word in tmp:
                output += str(tmp[word] / total) + " "
            else: output += (str(1 / total) + " ")
        file.write(output[:-1] + "\n") # remove trailing space
    file.close()

def trans(d, path_out, tags):
    file = open(str(path_out), "w")
    for i in tags:
        output = ""
        tmp = d[i]
        total = float(sum(tmp.values()) + len(tags) - len(tmp))
        for j in tags:
            if j in tmp: output += str(tmp[j] / total) + " "
            else: output += str(1 / total) + " "
        file.write(output[:-1] + "\n")
    file.close()

def main():
    words = read_txt(sys.argv[2])
    tags = read_txt(sys.argv[3])

    pi, e, t = read_data(sys.argv[1])
    prior(pi, sys.argv[4], tags)
    emit(e, sys.argv[5], tags, words)
    trans(t, sys.argv[6], tags)
    
    print('done!')

if __name__ == '__main__':
    main()