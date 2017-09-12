from sklearn import preprocessing
#import primitive
import sys
#sys.path.append('corexdiscrete/')
import corexdiscrete.corex as corex_disc
#import bio_corex.corex as corex_disc
from collections import defaultdict
from scipy import sparse
import pandas as pd
import numpy as np

class CorexDiscrete:  #(Primitive):

    def __init__(self, dim_hidden = 2, **kwargs): 
        #n_hidden = None, latent_pct = None, dim_hidden = 2, **kwargs):
    	
        '''TO DO: Prune/add initialization arguments'''
        ##super().__init__(name = 'CorexDiscrete', cls = 'corex_primitives.CorexDiscrete') # inherit from primitive?
        #default for no columns fed?  taking all likely leads to errors
        # self.dim_hidden = dim_hidden
        
        # if latent_pct is not None and n_hidden is None:
        # 	self.n_hidden = int(len(self.columns)*latent_pct)
        # else:
        # 	self.n_hidden = 2 if n_hidden is None else n_hidden #DEFAULT = 2 

        #if n_hidden:
        #    self.n_hidden = n_hidden
        #    self.model = corex_disc.Corex(n_hidden= self.n_hidden, dim_hidden = self.dim_hidden, marginal_description='discrete', **kwargs)
        #else:
        #    if latent_pct is None:
        #        self.n_hidden = 2
        #        self.model = corex_disc.Corex(n_hidden= self.n_hidden, dim_hidden = self.dim_hidden, marginal_description='discrete', **kwargs)
        #   else:
        #       self.kwargs = kwargs
        #       self.model = None

        self.kwargs = kwargs
        self.is_feature_selection = False
        self.hyperparameters = {'k': 2}#, 'dim_hidden': 2}
        self.dim_hidden = dim_hidden

        #self.model = corex_disc.Corex(n_hidden= self.n_hidden, dim_hidden = self.dim_hidden, marginal_description='discrete', **kwargs)

    def fit(self, A, k, dim_hidden = 2):
        self.fit_transform(A, k, dim_hidden)
        return self

    def predict(self, A, y = None): 

        self.columns = list(A)
        A_ = A[self.columns].values # TO DO: add error handling for X[self.columns]?

        #if self.model is None and self.latent_pct:
        #    self.model = corex_disc.Corex(n_hidden= int(self.latent_pct*len(self.columns)), dim_hidden = self.dim_hidden, marginal_description='discrete', **self.kwargs)
    	
        if self.dim_hidden == 2:
        	factors = self.model.transform(A_, details = True)[0]
        	factors = np.transpose(np.squeeze(factors[:,:,1])) if len(factors.shape) == 3 else factors # assuming dim_hidden = 2
        else:
        	factors = self.model.fit_transform(A_)

        return factors

    def fit_transform(self, A, k, dim_hidden = 2, y = None):
        self.dim_hidden = dim_hidden
        self.columns = list(A)
     	A_ = A[self.columns].values

        self.model = corex_disc.Corex(n_hidden= k, dim_hidden = self.dim_hidden, **self.kwargs)

        if self.dim_hidden == 2:
        	self.model.fit(A_)
        	factors = self.model.transform(A_, details = True)[0]
        	factors = np.transpose(np.squeeze(factors[:,:,1])) if len(factors.shape) == 3 else factors # assuming dim_hidden = 2
        else:
        	factors = self.model.fit_transform(A_)

    	return factors

    def annotation(self):
        if self._annotation is not None:
            return self._annotation
        self._annotation = Primitive()
        self._annotation.name = 'CorexDiscrete'
        self._annotation.task = 'FeatureExtraction'
        self._annotation.learning_type = 'UnsupervisedLearning'
        self._annotation.ml_algorithm = ['Dimension Reduction']
        self._annotation.tags = ['feature_extraction'] #'discrete'
        return self._annotation

    def get_feature_names(self):
    	return ['CorexDisc_'+ str(i) for i in range(self.n_hidden)]
