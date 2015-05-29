#!/opt/python-2.7/bin/python2.7

'''
Author: Andrea Kahn
Last Modified: May 28, 2015

This script takes as input a path to a text file of labeled sentences in the following
format:

LABEL1 This is an example sentence.
LABEL2 This is another example sentence.
etc.

It then prints to standard out feature vectors representing the sentences (one per line;
same order) in MALLET SVM lite format.

'''

import logging
from sys import argv
from extract_features import *
from nltk import RegexpTokenizer

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


def main():
    logging.basicConfig()
    
    # Read in the sentence file and store sentences and their corresponding labels
    labels = []
    sentences = []
    
    sentence_f = open(argv[1])
    for line in sentence_f.readlines():
        line = line.strip().split()
        if line:
            label = line[0]
            sentence = ' '.join(line[1:])
            labels.append(label)
            sentences.append(sentence)
    sentence_f.close()
    
    # Iterate through the sentences
    for i in range(len(sentences)):
    
        # Preprocess the text
        # FIXME: Store capitals, exclamation points, emoticons
        s = sentences[i]
        s = s.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(s)
        s = ' '.join(tokens)
    
        # Extract features and store (feature, value) tuples in a list
        features = []
        features.extend(extract_unigrams(s))
        features.extend(extract_bigrams(s))
        features.extend(extract_trigrams(s))
        features.extend(extract_skipgrams(s))
        features.extend(extract_pos_ngrams(s))
        
        # Print the vector to standard out
        print labels[i],
        for (f,v) in features:
            print f+':'+str(v),
        print

if __name__ == '__main__':
    main()