#!/opt/python-2.7/bin/python2.7

'''
Author: Andrea Kahn
Last Modified: June 4, 2015

This script takes a single line of MALLET predictions, in the following format, as input:
array:<instance_number> \t <label1> \t <probability of label1> \t <label2> \t <probability2> etc.

It then prints to standard out the label with the highest probability.

'''

from sys import argv

def main():
    labels = {}
    tokens = argv[1].split()
    for i in range(1, len(tokens)-1, 2):
        labels[tokens[i]] = tokens[i+1]
    print max(labels.keys(), key=labels.get)


if __name__ == '__main__':
    main()