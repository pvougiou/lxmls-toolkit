import numpy as np
import scipy as scipy
import lxmls.classifiers.linear_classifier as lc
import sys
from lxmls.distributions.gaussian import *


class MultinomialNaiveBayes(lc.LinearClassifier):

    def __init__(self, xtype="gaussian"):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = True
        self.smooth_param = 1

    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words
        n_docs, n_words = x.shape

        # classes = a list of possible classes
        classes = np.unique(y)
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]

        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        if self.smooth is False:
            likelihood = np.zeros((n_words, n_classes))
        else:
            # Initialise with ones in order to introduce the regularisation term.
            likelihood = np.ones((n_words, n_classes))

         
        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
        # prior[0] is the prior probability of a document being of class 0
        # likelihood[4, 0] is the likelihood of the fifth(*) feature being
        # active, given that the document is of class 0
        # (*) recall that Python starts indices at 0, so an index of 4
        # corresponds to the fifth feature!

        words_per_class = np.zeros(n_classes)
        unique, counts = np.unique(y, return_counts=True)
        class_distr = dict(zip(unique, counts / float(n_docs)))

        for feat in range(0, n_words):
            for doc in range(0, n_docs): 
                likelihood[feat][y[doc][0]] += x[doc][feat]
                words_per_class[y[doc][0]] += x[doc][feat]
        if self.smooth is False:   
            for j in range(0, n_classes):
                prior[j] = class_distr[classes[j]]
                likelihood[:, j] = likelihood[:, j] / words_per_class[j]
        else:
            # Compute  likelihood by adding a regularisation term (i.e. n_words).
            for j in range(0, n_classes):
                prior[j] = class_distr[classes[j]]
                likelihood[:, j] = likelihood[:, j] / (words_per_class[j] + n_words)


        # Complete Exercise 1.1 
        # raise NotImplementedError("Complete Exercise 1.1")

        params = np.zeros((n_words+1, n_classes))
        for i in xrange(n_classes):
            params[0, i] = np.log(prior[i])
            params[1:, i] = np.nan_to_num(np.log(likelihood[:, i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
