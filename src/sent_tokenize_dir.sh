#!/bin/sh

# Author: Andrea Kahn
# Last Modified: June 8, 2015
# 
# This script runs sent_tokenize.py on all files in a directory, redirecting the
# output to the original files (WARNING: original files are overwritten).

# Change this line to change the directory to be sentence-tokenized
FILES=$(find /home2/amkahn/workspace/RA/review-classifier/data/reviews-tagged/TRUST -maxdepth 1 -type f)
for FILE in $FILES; do
    REVIEW=$(cat $FILE)
    ./sent_tokenize.py "$REVIEW" > $FILE
done