#!/opt/python-2.7/bin/python2.7

'''
Author: Andrea Kahn
Last Modified: June 9, 2015

This script takes as input:
1) a class label, and
2) a path to a txt file containing a review in the following format:
SENTENCE1_TAG sentence1
SENTENCE2_TAG sentence 2
etc.

It then prints a feature vector representing the text in MALLET SVM lite format to
standard out.

'''

import logging
import re
from sys import argv
from extract_features import *
from nltk import RegexpTokenizer, sent_tokenize

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


def main():
    logging.basicConfig()
    
    label = argv[1]
    features = []
    
    review_f = open(argv[2])
    lines = review_f.readlines()
    review_f.close()
    
    # Remove newlines
    lines = map(lambda x: x.strip(), lines)
    
    # Split lines into (tag, sentence) tuples
    pairs = map(lambda x: detag_sentence(x), lines)
    pairs = filter(lambda x: x, pairs)
    LOG.debug("Here are the pairs: %s" % str(pairs))
    
    # Make tuples of the tag sequence and the sentence sequence, respectively
    tags, sentences = zip(*pairs)
#     tags = list(tags)
#     sentences = list(sentences)
    LOG.debug("Here are the sentence tags: %s" % str(tags))
    LOG.debug("Here are the sentences: %s" % str(sentences))
    
    review = ' '.join(sentences)
    LOG.debug("Building vector for review: %s" % review)
    
    # Preprocess the text, adding START and STOP tokens to sentences
    # FIXME: Store capitals, exclamation points, emoticons
    for sentence in sentences:
#         tokens = ['START']
        sentence = sentence.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
#         tokens.extend(tokenizer.tokenize(sentence))
#         tokens.append('STOP')
        sentence = ' '.join(tokens)

        # Extract (feature, value) tuples and append them to the features list 
        features.extend(extract_unigrams(sentence))
        features.extend(extract_bigrams(sentence))
        features.extend(extract_trigrams(sentence))
#         features.extend(extract_skipgrams(sentence))
        features.extend(extract_pos_ngrams(sentence))

    # Convert the features to a string in MALLET SVM lite format
    feature_str = ''
    for (f,v) in features:
        feature_str += ' '+f+':'+str(v)
    
    # Print the vector to standard out
    print label+feature_str


def detag_sentence(sent):
    '''
    Function: detag_sentences
    Given a list of strings in the format consisting of a sentence tag followed by
    whitespace followed by the sentence, return the tag sequence and the sentence
    sequence as separate lists
    @param: tagged_sents = a list of strings in the format: TAG sentence
    @return: a 2-tuple of (tag_list, sentence_list)
    '''
#     tags = []
#     sentences = []
    tag_pattern = re.compile(r'[A-Z]+')
    
#     for sent in tagged_sents:
    tokens = sent.split()
    if (len(tokens)<2) or not tag_pattern.match(tokens[0]):
        LOG.warn("Line is not in expected format; skipping: %s" % sent)
        return
    else:
        tag = tokens[0]
        sentence = ' '.join(tokens[1:])
        return (tag, sentence)
#             sentences.append(' '.join(tokens[1:]))
#     return (tags, sentences)

    
if __name__ == '__main__':
    main()