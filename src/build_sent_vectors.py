#!/opt/python-2.7/bin/python2.7

'''
Author: Andrea Kahn
Last Modified: June 9, 2015

This script takes as input a path to a text file of labeled sentences in the following
format:

LABEL1 This is an example sentence.
LABEL2 This is another example sentence.
etc.

It then prints feature vectors representing the sentences (one per line; same order) in
MALLET SVM lite format to text files in ../data/vec (one for each label).

'''

import logging
from sys import argv
from extract_features import *
from nltk import RegexpTokenizer

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


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
    
    # Keep track of which files you've opened in a dictionary so you don't reopen them,
    # and can later close them
    # Keys are labels, values are corresponding open file objects
    open_files = {}
    
    # Iterate through the sentences
    for i in range(len(sentences)):
        LOG.debug("Building vector for sentence: %s" % sentences[i])
    
        # Preprocess the text
        # FIXME: Store capitals, exclamation points, emoticons
        s = sentences[i]
        s = s.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(s)
        s = ' '.join(tokens)
    
        # Extract features and store (feature, value) tuples in a list
        features = []
        features.extend(extract_unigrams([s]))
        features.extend(extract_bigrams([s]))
        features.extend(extract_trigrams([s]))
#         features.extend(extract_skipgrams([s]))
        features.extend(extract_pos_ngrams([s]))
        
        # Old code to print all vectors together to standard out
#         print labels[i],
#         for (f,v) in features:
#             print f+':'+str(v),
#         print

        # Convert the features to a string in MALLET SVM lite format
        feature_str = ''
        for (f,v) in features:
            feature_str += ' '+f+':'+str(v)
        
        # Print each list of vectors to a file named after their label
        if open_files.get(labels[i]):
#             LOG.debug("Writing vector: %s" % labels[i]+' '+feature_str)
            open_files[labels[i]].write(labels[i]+feature_str+'\n')
        else:
            open_files[labels[i]] = open('../data/vec/%s.txt' % labels[i], 'w')
#             LOG.debug("Writing vector: %s" % labels[i]+' '+feature_str)
            open_files[labels[i]].write(labels[i]+feature_str+'\n')
    
    # Close all the open files
    for f in open_files.values():
        f.close()
        
if __name__ == '__main__':
    main()