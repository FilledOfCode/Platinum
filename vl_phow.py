
from scipy import shape, dstack, sqrt, floor, array, mean, ones, vstack, hstack, ndarray
from vlfeat import vl_rgb2gray, vl_imsmooth, vl_dsift
from sys import maxint

"""
Python rewrite of https://github.com/vlfeat/vlfeat/blob/master/toolbox/sift/vl_phow.m