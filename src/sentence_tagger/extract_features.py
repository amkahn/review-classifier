#!/opt/python-2.7/bin/python2.7

'''
Author: Andrea Kahn
Last Modified: May 28, 2015

This module contains various functions for extracting features from text strings.

FIXME: Add START and STOP tokens

'''

import logging
from collections import defaultdict
from nltk import sent_tokenize, word_tokenize, pos_tag

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def extract_unigrams(text, min_val=0):
    '''
    Function: extract_unigrams
    Given a string of preprocessed text, count the word unigrams, normalize the counts by
    the total unigram count, and return them as features
    @param: text = a string of text from which to extract features,
        min_val = a float value corresponding to the minimum value of feature that should
        be included (optional)
    @return: feature_list = a list of (feature, value) tuples
    '''
#     LOG.info("Extracting unigram features")
    num_unigrams = 0
    unigram_counts = defaultdict(lambda: 0)
    tokens = text.split()
    
    for token in tokens:
        num_unigrams += 1
        unigram_counts[token] += 1
    
    # Create the feature list
    feature_list = []
    for feature in sorted(unigram_counts.keys(), key=lambda t: unigram_counts[t], reverse=True):
        value = unigram_counts[feature]/float(num_unigrams)
        # Check for underflow
        if value == 0.0:
            LOG.warning("Value for feature %s is 0" % feature)
        elif value >= min_val:
            feature_list.append((feature, value))
        else:
            LOG.debug("Skipping feature %s (value %s is below threshold %s)" % (feature, value, min_val))
    return feature_list

        
def extract_bigrams(text):
    '''
    Function: extract_bigrams
    Given a string of preprocessed text, count the word bigrams, normalize the counts by
    the total bigram count, and return them as features
    @param: text = a string of text from which to extract features
    @return: feature_list = a list of feature:value strings in MALLET SVM lite format
    '''
#     LOG.info("Extracting bigram features")

    num_bigrams = 0
    bigram_counts = defaultdict(lambda: 0)
    tokens = text.split()
    
    if len(tokens) < 2:
        LOG.debug("Sentence has fewer than 2 tokens: %s" % text)
        return []
    
    for i in range(1, len(tokens)):
        num_bigrams += 1
        bigram = tokens[i-1]+'~'+tokens[i]
        bigram_counts[bigram] += 1
    
    # Create the feature list
    feature_list = []
    for feature in sorted(bigram_counts.keys(), key=lambda t: bigram_counts[t], reverse=True):
        value = float(num_bigrams)
        # Check for underflow
        if value == 0.0:
            LOG.warning("Value for feature %s is 0" % feature)
        feature_list.append((feature, bigram_counts[feature]/value))
        
    return feature_list


def extract_trigrams(text):
    '''
    Function: extract_trigrams
    Given a string of preprocessed text, count the word trigrams, normalize the counts by
    the total trigram count, and return them as features
    @param: text = a string of text from which to extract features
    @return: feature_list = a list of feature:value strings in MALLET SVM lite format
    '''
#     LOG.info("Extracting trigram features")
    num_trigrams = 0
    trigram_counts = defaultdict(lambda: 0)
    tokens = text.split()

    if len(tokens) < 3:
        LOG.debug("Sentence has fewer than 3 tokens: %s" % text)
        return []
    
    for i in range(2, len(tokens)):
        num_trigrams += 1
        trigram = tokens[i-2]+'~'+tokens[i-1]+'~'+tokens[i]
        trigram_counts[trigram] += 1
    
    # Create the feature list
    feature_list = []
    for feature in sorted(trigram_counts.keys(), key=lambda t: trigram_counts[t], reverse=True):
        value = trigram_counts[feature]/float(num_trigrams)
        # Check for underflow
        if value == 0.0:
            LOG.warning("Value for feature %s is 0" % feature)
        feature_list.append((feature, value))
        
    return feature_list


def extract_skipgrams(text):
    '''
    Function: extract_skipgrams
    Given a string of preprocessed text, count the skip-1 bigrams, normalize the counts by
    the total trigram count, and return them as features
    @param: text = a string of text from which to extract features
    @return: feature_list = a list of feature:value strings in MALLET SVM lite format
    '''
#     LOG.info("Extracting trigram features")
    num_trigrams = 0
    skipgram_counts = defaultdict(lambda: 0)
    tokens = text.split()

    if len(tokens) < 3:
        LOG.debug("Sentence has fewer than 3 tokens: %s" % text)
        return []
    
    for i in range(2, len(tokens)):
        num_trigrams += 1
#         skipgram = tokens[i-2]+tokens[i]
        skipgram = tokens[i-2]+'~*~'+tokens[i]
        skipgram_counts[skipgram] += 1
    
    # Create the feature list
    feature_list = []
    for feature in sorted(skipgram_counts.keys(), key=lambda t: skipgram_counts[t], reverse=True):
        value = skipgram_counts[feature]/float(num_trigrams)
        # Check for underflow
        if value == 0.0:
            LOG.warning("Value for feature %s is 0" % feature)
        feature_list.append((feature, value))
        
    return feature_list


def extract_pos_ngrams(text):
    '''
    Function: extract_pos_ngrams
    Given a string of preprocessed text, count the POS-tag unigrams/bigrams/trigrams/
    skipgrams, normalize the counts by the total unigram/bigram/trigram/skipgram count,
    and return them as features
    @param: text = a string of text from which to extract features
    @return: feature_list = a list of feature:value strings in MALLET SVM lite format
    '''
#     LOG.info("Extracting POS-tag n-gram features")
    num_unigrams = 0
    num_bigrams = 0
    num_trigrams = 0
    
    unigram_counts = defaultdict(lambda: 0)
    bigram_counts = defaultdict(lambda: 0)
    trigram_counts = defaultdict(lambda: 0)
    skipgram_counts = defaultdict(lambda: 0)
    
    tokens = text.split()
    tagged_tokens = pos_tag(tokens)
    LOG.debug("Here are the tagged tokens: %s" % tagged_tokens)

    num_unigrams += 1
    unigram = tagged_tokens[0][1]
    unigram_counts[unigram] += 1

    if len(tagged_tokens) > 1:
        num_unigrams += 1
        unigram = tagged_tokens[1][1]
        unigram_counts[unigram] += 1
        
        bigram = tagged_tokens[0][1]+'~'+tagged_tokens[1][1]
        bigram_counts[bigram] += 1
        
        if len(tagged_tokens) > 2:
            for i in range(2, len(tagged_tokens)):
                num_unigrams += 1
        
                unigram = tagged_tokens[i][1]
                unigram_counts[unigram] += 1
        
                bigram = tagged_tokens[i-1][1]+'~'+tagged_tokens[i][1]
                bigram_counts[bigram] += 1

                trigram = tagged_tokens[i-2][1]+'~'+tagged_tokens[i-1][1]+'~'+tagged_tokens[i][1]
                trigram_counts[trigram] += 1

        #         skipgram = tokens[i-2][1]+tokens[i][1]
                skipgram = tagged_tokens[i-2][1]+'~*~'+tagged_tokens[i][1]
                skipgram_counts[skipgram] += 1
    
            num_trigrams = num_unigrams - 2
                
        num_bigrams = num_unigrams - 1
    
    # Create the feature list
    feature_list = []
    for feature in sorted(unigram_counts.keys(), key=lambda t: unigram_counts[t], reverse=True):
        value = unigram_counts[feature]/float(num_unigrams)
        # Check for underflow
        if value == 0.0:
            LOG.warning("Value for feature %s is 0" % feature)
        feature_list.append((feature, value))
    for feature in sorted(bigram_counts.keys(), key=lambda t: bigram_counts[t], reverse=True):
        value = bigram_counts[feature]/float(num_bigrams)
        # Check for underflow
        if value == 0.0:
            LOG.warning("Value for feature %s is 0" % feature)
        feature_list.append((feature, value))
    for feature in sorted(trigram_counts.keys(), key=lambda t: trigram_counts[t], reverse=True):
        value = trigram_counts[feature]/float(num_trigrams)
        # Check for underflow
        if value == 0.0:
            LOG.warning("Value for feature %s is 0" % feature)
    for feature in sorted(skipgram_counts.keys(), key=lambda t: skipgram_counts[t], reverse=True):
        value = skipgram_counts[feature]/float(num_trigrams)
        # Check for underflow
        if value == 0.0:
            LOG.warning("Value for feature %s is 0" % feature)
        feature_list.append((feature, value))
        feature_list.append((feature, value))
    LOG.debug(skipgram_counts)
#     LOG.debug("Here is the POS-tag unigram feature list: %s" % feature_list)
        
    return feature_list
