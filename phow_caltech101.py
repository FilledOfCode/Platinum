
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

        # tests and conversions
        self.phowOpts.Sizes = ensure_type_array(self.phowOpts.Sizes)
        self.numSpatialX = ensure_type_array(self.numSpatialX)
        self.numSpatialY = ensure_type_array(self.numSpatialY)
        if (self.numSpatialX != self.numSpatialY).any():
            messageformat = [str(self.numSpatialX), str(self.numSpatialY)]
            message = "(self.numSpatialX != self.numSpatialY), because {0} != {1}".format(*messageformat)
            raise ValueError(message)


def ensure_type_array(data):
    if (type(data) is not ndarray):
        if (type(data) is list):
            data = array(data)
        else:
            data = array([data])
    return data


def standarizeImage(im):
    im = array(im, 'float32') 
    if im.shape[0] > 480:
        resize_factor = 480.0 / im.shape[0]  # don't remove trailing .0 to avoid integer devision
        im = imresize(im, resize_factor)
    if amax(im) > 1.1:
        im = im / 255.0
    assert((amax(im) > 0.01) & (amax(im) <= 1))
    assert((amin(im) >= 0.00))
    return im


def getPhowFeatures(imagedata, phowOpts):
    im = standarizeImage(imagedata)
    frames, descrs = vl_phow(im,
                             verbose=phowOpts.Verbose,
                             sizes=phowOpts.Sizes,
                             step=phowOpts.Step)
    return frames, descrs


def getImageDescriptor(model, im):
    im = standarizeImage(im)
    height, width = im.shape[:2]
    numWords = model.vocab.shape[1]

    frames, descrs = getPhowFeatures(im, conf.phowOpts)
    # quantize appearance
    if model.quantizer == 'vq':
        binsa, _ = vq(descrs.T, model.vocab.T)
    elif model.quantizer == 'kdtree':
        raise ValueError('quantizer kdtree not implemented')
    else:
        raise ValueError('quantizer {0} not known or understood'.format(model.quantizer))

    hist = []
    for n_spatial_bins_x, n_spatial_bins_y in zip(model.numSpatialX, model.numSpatialX):
        binsx, distsx = vq(frames[0, :], linspace(0, width, n_spatial_bins_x))
        binsy, distsy = vq(frames[1, :], linspace(0, height, n_spatial_bins_y))
        # binsx and binsy list to what spatial bin each feature point belongs to
        if (numpy.any(distsx < 0)) | (numpy.any(distsx > (width/n_spatial_bins_x+0.5))):
            print 'something went wrong'
            import pdb; pdb.set_trace()
        if (numpy.any(distsy < 0)) | (numpy.any(distsy > (height/n_spatial_bins_y+0.5))):
            print 'something went wrong'
            import pdb; pdb.set_trace()

        # combined quantization
        number_of_bins = n_spatial_bins_x * n_spatial_bins_y * numWords
        temp = arange(number_of_bins)
        # update using this: http://stackoverflow.com/questions/15230179/how-to-get-the-linear-index-for-a-numpy-array-sub2ind
        temp = temp.reshape([n_spatial_bins_x, n_spatial_bins_y, numWords])
        bin_comb = temp[binsx, binsy, binsa]
        hist_temp, _ = histogram(bin_comb, bins=range(number_of_bins+1), density=True)
        hist.append(hist_temp)

    hist = hstack(hist)
    hist = array(hist, 'float32') / sum(hist)
    return hist


class Model(object):
    def __init__(self, classes, conf, vocab=None):
        self.classes = classes
        self.phowOpts = conf.phowOpts
        self.numSpatialX = conf.numSpatialX
        self.numSpatialY = conf.numSpatialY
        self.quantizer = conf.quantizer
        self.vocab = vocab


class SVMParameters(object):
    def __init__(self, C):
        self.C = C


class PHOWOptions(object):
    def __init__(self, Verbose, Sizes, Step):
        self.Verbose = Verbose
        self.Sizes = Sizes
        self.Step = Step


def get_classes(datasetpath, numClasses):
    classes_paths = [files
                     for files in glob(datasetpath + "/*")
                     if isdir(files)]
    classes_paths.sort()
    classes = [basename(class_path) for class_path in classes_paths]
    if len(classes) == 0:
       raise ValueError('no classes found')