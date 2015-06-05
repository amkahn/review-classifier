#!/opt/python-2.7/bin/python2.7

'''
Author: Andrea Kahn
Last Modified: June 3, 2015

This script takes as input a path to a text file of labeled sentences in the following
format:

This is an example sentence.
This is another example sentence.
etc.

It then prints feature vectors representing the sentences (one per line; same order) in
MALLET SVM lite format to text files in ../data/vec (one for each label).

'''

import logging
from sys import argv, exit
from extract_features import *
from nltk import RegexpTokenizer

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def main():
    logging.basicConfig()
    
    # Read in the sentence file and store sentences
    sentences = []  
    sentence_f = open(argv[1])
    for line in sentence_f.readlines():
        line = line.strip().split()
        if line:
            sentence = ' '.join(line[0:])
            sentences.append(sentence)
    sentence_f.close()
    
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
        features.extend(extract_unigrams(s))
        features.extend(extract_bigrams(s))
        features.extend(extract_trigrams(s))
#         features.extend(extract_skipgrams(s))
        features.extend(extract_pos_ngrams(s))

        # If sentence has no features, exit with a warning message
        if len(features) < 1:
            LOG.warn("Sentence has no features; skipping: %s" % sentences[i])
            exit(1) 

        # Convert the features to a string in MALLET SVM lite format
        feature_str = ''
        for (f,v) in features:
            feature_str += ' '+f+':'+str(v)
        
        # Print the vector to standard out
        # NB: A random label is appended to all vectors in order to comply with MALLET's input
        # requirements. It is not necessarily the correct label and will not affect classification.
        print 'PRAISE'+feature_str

        
if __name__ == '__main__':
    main()