#!/bin/sh

# Author: Andrea Kahn
# Last Modified: June 3, 2015
#
# This script uses an already-trained classifier to tag sentences.

# First argument to script is run tag
TAG=$1

# Clear the text files to for the cross-validation folds
for((i=1;i<N+1;++i)); do
    > ../out/"$TAG"_$i.txt
done

# Build the feature vectors from the input sentences
./build_unlabeled_vectors.py ../data/unlabeled.txt > ../data/vec_to_label.txt

# Perform classification using the already-trained classifier
mallet classify-svmlight --input ../data/vec_to_label.txt --output - --classifier ../out/"$TAG"_maxent.trial0
