#!/bin/sh

# Author: Andrea Kahn
# Last Modified: June 3, 2015
#
# This script uses an already-trained classifier to tag sentences stored in a txt file
# (one per line).

# First argument to script is run tag
TAG=$1

# Build the feature vectors from the input sentences
./build_unlabeled_vectors.py ../data/unlabeled.txt > ../data/vec_to_label.txt

# Perform classification using the already-trained classifier
mallet classify-svmlight --input ../data/vec_to_label.txt --output ../out/tmp2.txt --classifier ../out/"$TAG"_maxent.trial0
