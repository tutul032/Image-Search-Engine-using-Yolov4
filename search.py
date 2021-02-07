# import the necessary packages
import numpy as np
import cv2
import glob
import os
import _pickle as cpickle
import pickle
from scipy.spatial.distance import euclidean


class RGBHistogram:

    def __init__(self):
        # store the number of bins the histogram will use
        self.bins = [8, 8, 8]

    def describe(self, image):
        arr = np.array([])
        # compute a 3D histogram in the RGB colorspace,
        # then normalize the histogram so that images
        # with the same content, but either scaled larger
        # or smaller will have (roughly) the same histogram
        hist = cv2.calcHist([image], [0, 1, 2],
            None, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist,arr).flatten()
        # return out 3D histogram as a flattened array
        return hist

    def feature_extraction(self, dataset):
        index = {}
        
        # use glob to grab the image paths and loop over them
        for imagePath in glob.glob(dataset): # example: "lord" + "/*.png"
            j, k = os.path.split(imagePath)

            # load the image, describe it using our RGB histogram
            # descriptor, and update the index
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (400, 200)) 
            features = self.describe(image)
            index[k] = features

            # open and save the index file
            f = open("index", "wb")
            f.write(cpickle.dumps(index))
            f.close


class Searcher:
    def __init__(self, index):
        # store our index of images
        self.index = index

    def search(self, queryFeatures):
        # initialize our dictionary of results
        results = {}

        # loop over the index
        for (k, features) in self.index.items():

            d = self.chi2_distance(features, queryFeatures)
            results[k] = d

        results = sorted([(v, k) for (k, v) in results.items()])

        # return our results
        return results

    def chi2_distance(self, histA, histB, eps = 1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])
        # return the chi-squared distance
        return d
    
