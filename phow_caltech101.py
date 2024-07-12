
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
from scipy.cluster.vq import vq
import numpy
from vl_phow import vl_phow
from vlfeat import vl_ikmeans
from scipy.io import loadmat, savemat
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
import pylab as pl
from datetime import datetime
from sklearn.kernel_approximation import AdditiveChi2Sampler
from cPickle import dump, load


IDENTIFIER = '05.04.13'
SAVETODISC = False
FEATUREMAP = True
OVERWRITE = False  # DON'T load mat files genereated with a different seed!!!
SAMPLE_SEED = 42
TINYPROBLEM = False
VERBOSE = True  # set to 'SVM' if you want to get the svm output
MULTIPROCESSING = False


class Configuration(object):
    def __init__(self, identifier=''):
        self.calDir = '../../../datasets/Caltech/101_ObjectCategories'
        self.dataDir = 'tempresults'  # should be resultDir or so
        if not exists(self.dataDir):
            makedirs(self.dataDir)
            print "folder " + self.dataDir + " created"
        self.autoDownloadData = True
        self.numTrain = 15
        self.numTest = 15
        self.imagesperclass = self.numTrain + self.numTest
        self.numClasses = 102
        self.numWords = 600
        self.numSpatialX = [2, 4]
        self.numSpatialY = [2, 4]
        self.quantizer = 'vq'  # kdtree from the .m version not implemented
        self.svm = SVMParameters(C=10)
        self.phowOpts = PHOWOptions(Verbose=False, Sizes=[4, 6, 8, 10], Step=3)
        self.clobber = False
        self.tinyProblem = TINYPROBLEM
        self.prefix = 'baseline'
        self.randSeed = 1
        self.verbose = True
        self.extensions = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
        self.images_for_histogram = 30
        self.numbers_of_features_for_histogram = 100000
        
        self.vocabPath = join(self.dataDir, identifier + '-vocab.py.mat')
        self.histPath = join(self.dataDir, identifier + '-hists.py.mat')
        self.modelPath = join(self.dataDir, self.prefix + identifier + '-model.py.mat')
        self.resultPath = join(self.dataDir, self.prefix + identifier + '-result')
        
        if self.tinyProblem:
            print "Using 'tiny' protocol with different parameters than the .m code"
            self.prefix = 'tiny'
            self.numClasses = 5
            self.images_for_histogram = 10
            self.numbers_of_features_for_histogram = 1000
            self.numTrain
            self.numSpatialX = 2
            self.numWords = 100
            self.numTrain = 2
            self.numTest = 2
            self.phowOpts = PHOWOptions(Verbose=2, Sizes=7, Step=5)