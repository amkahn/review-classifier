#!/opt/python-2.7/bin/python2.7

'''
Author: Andrea Kahn
Last Modified: June 5, 2015

This script takes as input:
1) a class label, and
2) a path to a txt file containing a review (can be multiple lines; line breaks are
treated as spaces)

It then prints a feature vector representing the text in MALLET SVM lite format to
standard out.

'''

import logging
from sys import argv
from extract_features import *
from nltk import RegexpTokenizer

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def main():
    logging.basicConfig()
    
    label = argv[1]
    review_f = open(argv[2])
    lines = review_f.readlines()
    lines = map(lambda x: x.strip(), lines)
    review = ' '.join(lines)
    review_f.close()
    
    LOG.debug("Building vector for review: %s" % review)

    # Preprocess the text
    # FIXME: Store capitals, exclamation points, emoticons
    review = review.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(review)
    review = ' '.join(tokens)

    # Extract features and store (feature, value) tuples in a list
    features = []
    features.extend(extract_unigrams(review))
    features.extend(extract_bigrams(review))
    features.extend(extract_trigrams(review))
#         features.extend(extract_skipgrams(review))
    features.extend(extract_pos_ngrams(review))

    # Convert the features to a string in MALLET SVM lite format
    feature_str = ''
    for (f,v) in features:
        feature_str += ' '+f+':'+str(v)
    
    # Print the vector to standard out
    print label+feature_str
    
if __name__ == '__main__':
    main()