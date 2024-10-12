
from scipy import shape, dstack, sqrt, floor, array, mean, ones, vstack, hstack, ndarray
from vlfeat import vl_rgb2gray, vl_imsmooth, vl_dsift
from sys import maxint

"""
Python rewrite of https://github.com/vlfeat/vlfeat/blob/master/toolbox/sift/vl_phow.m
### notice no hsv support atm
### comments are largely copied from the code

"""


def vl_phow(im,
            verbose=True,
            fast=True,
            sizes=[4, 6, 8, 10],
            step=2,
            color='rgb',
            floatdescriptors=False,
            magnif=6,
            windowsize=1.5,
            contrastthreshold=0.005):

    opts = Options(verbose, fast, sizes, step, color, floatdescriptors,
                   magnif, windowsize, contrastthreshold)
    dsiftOpts = DSiftOptions(opts)

    # make sure image is float, otherwise segfault
    im = array(im, 'float32')

    # Extract the features
    imageSize = shape(im)