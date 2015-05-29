#!/opt/python-2.7/bin/python2.7

'''
Author: Andrea Kahn
Last Modified: Feb. 12, 2015

This script parses a MALLET txt file representing a MaxEnt classifier and prints the
features in descending sorted order by weight.

Command line usage: ./sort_features.py <classifier-file> <class-name>

'''

from sys import argv, stderr
from collections import defaultdict
import re

def sort_features(f, class_name):
    '''
    Function: sort_features
    Parse a txt file output by MALLET to represent a MaxEnt classfier and store the
        features in a dictionary in which feature names are keys and weights are values.
    @param: f = an open file object as described above
    @return: feat_dict = a dictionary object as described above
    '''
    feat_dict = defaultdict(lambda: 0)
    header_re = re.compile(r'FEATURES FOR CLASS')
    class_header_re = re.compile(r'FEATURES FOR CLASS %s' % class_name)
    section = False
    
    for line in f:
        line = line.strip()
        
        # If this line is a header:
        if header_re.match(line):
        
            # If it's the correct header, set section variable to True
            if class_header_re.match(line):
                section = True
#                 stderr.write("Starting %s section\n" % line)
#                 if feat_dict:
#                     dict_list.append(feat_dict)
                feat_dict = defaultdict(lambda: 0)
                
            # If it's the wrong header, set section variable to True
            else:
                section = False
#                 stderr.write("Starting %s section\n" % line)
                
        # Else update the dictionary if section variable is True
        elif section == True:
                split_line = line.split()
                if split_line[0] in feat_dict:
                    stderr.write("WARNING: Feature %s already in dictionary\n" % split_line[0])
                feat_dict[split_line[0]] = float(split_line[1])

    return feat_dict

if __name__=='__main__':
    classifier_f = open(argv[1])
    class_name = argv[2]
    d = sort_features(classifier_f, class_name)
    stderr.write("Class %s has %s features\n" % (class_name, len(d)))
#     for feature in sorted(d.keys(), key=lambda f: d[f], reverse=True):
#         print feature, d[feature]
    for feature in sorted(d.keys(), key=lambda f: d[f], reverse=True)[:15]:
        print feature+', ',
    classifier_f.close()
    
    