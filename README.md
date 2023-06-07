# Loraphow_caltech101.py
==================

Script for content based image classification using the bag of visual words approach.

The script is a Python version of [phow_caltech101.m][1], a 'one file' example script using the [VLFeat library][6] to train and evaluate a image classifier on the [Caltech-101 data set][4]. 

Like the original Matlab version this Python script achives the same (State-of-the-Art in 2008) average accuracy of around 65% as the original file:

- PHOW features (dense multi-scale SIFT descriptors)
- Elkan k-means for fast visu