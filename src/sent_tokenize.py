#!/opt/python-2.7/bin/python2.7

'''
Author: Andrea Kahn
Last Modified: June 3, 2015

This script takes a string of text as an argument, sentence-tokenizes the text, and prints
the sentences to standard out, one per line.

'''

from sys import argv
from nltk import sent_tokenize

def main():
    sentences = sent_tokenize(argv[1])
    for s in sentences:
        print s


if __name__=='__main__':
    main()