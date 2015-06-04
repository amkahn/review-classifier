#!/bin/sh

# Author: Andrea Kahn
# Last Modified: June 3, 2015
# 
# This script takes as input a path to a one-line text file containing a single review.
# 
# It then prints to standard out:
# tag1 sentence1
# tag2 sentence2
# tag3 sentence3
# etc.
# 
# ...where the tag is the most likely tag as determined by an already trained classifier.

# Tokenize the sentences and insert newlines between them
R=$(cat $1)
# ./sent_tokenize.py "$R"
SENTS=$(./sent_tokenize.py "$R")

# Iterate through the sentences
oldifs="$IFS"
IFS=$'\n'
for SENT in $SENTS; do
    # Store the sentence in a temporary file
    echo "Tagging sentence: " $SENT
    echo $SENT > ../out/sent.txt
    
    # Build a feature vector from the sentence and store it in a file
    ./build_unlabeled_vectors.py ../out/sent.txt > ../data/vec.txt

    # Perform classification using the already-trained classifier
    mallet classify-svmlight --input ../data/vec.txt --output - --classifier ../out/r1_maxent.trial0
done
IFS="$oldifs"