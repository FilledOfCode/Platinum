
#!/usr/bin/env python

"""
Python rewrite of http: //www.vlfeat.org/applications/caltech-101-code.html
"""
from os.path import exists, isdir, basename, join, splitext
from os import makedirs
from glob import glob
from random import sample, seed
from scipy import ones, mod, arange, array, where, ndarray, hstack, linspace, histogram, vstack, amax, amin
from scipy.misc import imread, imresize