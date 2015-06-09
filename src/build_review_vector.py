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
from nltk import RegexpTokenizer, sent_tokenize

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


def main():
    logging.basicConfig()
    
    label = argv[1]
    review_f = open(argv[2])
    lines = review_f.readlines()
    review_f.close()
    
    lines = map(lambda x: x.strip(), lines)
    review = ' '.join(lines)
    LOG.debug("Building vector for review: %s" % review)
    
    sentences = sent_tokenize(review)
    LOG.debug("Here are the sentences: %s" % sentences)

    # Preprocess the text
    # FIXME: Store capitals, exclamation points, emoticons
    for i in range(len(sentences)):
        sentences[i] = sentences[i].lower()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentences[i])
        sentences[i] = ' '.join(tokens)

    # Extract features and store (feature, value) tuples in a list
    features = []
    features.extend(extract_unigrams(sentences))
    features.extend(extract_bigrams(sentences))
    features.extend(extract_trigrams(sentences))
#     features.extend(extract_skipgrams(sentences))
    features.extend(extract_pos_ngrams(sentences))

    # Convert the features to a string in MALLET SVM lite format
    feature_str = ''
    for (f,v) in features:
        feature_str += ' '+f+':'+str(v)
    
    # Print the vector to standard out
    print label+feature_str
    
if __name__ == '__main__':
    main()