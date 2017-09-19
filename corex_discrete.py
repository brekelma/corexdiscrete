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

sys.path.append('../')
from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from typing import NamedTuple, Union, Optional, Sequence


Input = pd.DataFrame
Output = np.ndarray
Params = NamedTuple('Params', [
    ('latent_factors', np.ndarray),  # Coordinates of cluster centers.
])

class CorexDiscrete(UnsupervisedLearningPrimitiveBase):  #(Primitive):

    def __init__(self, n_hidden: int = None, dim_hidden : int = 2, max_iter : int = 100, 
                n_repeat : int = 1, max_samples : int = 10000, n_cpu : int = 1, 
                smooth_marginals : bool = False, missing_values : float = -1, 
                seed : int = None, verbose : bool = False, **kwargs) -> None: 

        super().__init__()

        self.kwargs = kwargs
        self.is_feature_selection = False
        self.hyperparameters = {'n_hidden': 2} # NOT TRUE
        self.dim_hidden = dim_hidden

        self.max_iter = max_iter
        self.n_repeat = n_repeat
        self.max_samples = max_samples 
        self.n_cpu = n_cpu
        self.smooth_marginals = smooth_marginals
        self.missing_values = missing_values
        self.seed = seed
        self.verbose = verbose

        if n_hidden:
            self.n_hidden = n_hidden
            self.model = corex_disc.Corex(n_hidden= self.n_hidden, dim_hidden = self.dim_hidden,
                max_iter = self.max_iter, n_repeat = self.n_repeat, max_samples = self.max_samples,
                n_cpu = self.n_cpu, smooth_marginals= self.smooth_marginals, missing_values = self.missing_values, 
                verbose = self.verbose, seed = self.seed, **kwargs)
        else:
            if latent_pct is None:
                self.latent_pct = .10 # DEFAULT = 10% of ORIGINAL FACTORS
            else:
                self.latent_pct = latent_pct
            self.model = None

    def fit(self) -> None:
        if self.fitted:
            return
        if not self.training_inputs:
            raise ValueError("Missing training data.")

        self.fit_transform(self.training_inputs)
        self.fitted = True
        return self

    def produce(self, X : Sequence[Input], y : Sequence[Input] = None) -> Sequence[Output]: 

        self.columns = list(X)
        X_ = X[self.columns].values # TO DO: add error handling for X[self.columns]?

        if not self.fitted:
            self.n_hidden = int(self.latent_pct*len(self.columns))
            self.model = corex_disc.Corex(n_hidden= self.n_hidden, dim_hidden = self.dim_hidden,
                max_iter = self.max_iter, n_repeat = self.n_repeat, max_samples = self.max_samples,
                n_cpu = self.n_cpu, smooth_marginals= self.smooth_marginals, missing_values = self.missing_values, 
                verbose = self.verbose, seed = self.seed, **self.kwargs)
            self.model.fit(X_)
            self.fitted = True

        if self.dim_hidden == 2:
            self.factors = self.model.transform(X_, details = True)[0]
            self.factors = np.transpose(np.squeeze(self.factors[:,:,1])) if len(self.factors.shape) == 3 else self.factors # assuming dim_hidden = 2
        else:
            self.factors = self.model.transform(X_)

        return self.factors

    def fit_transform(self, X : Sequence[Input], y : Sequence[Input] = None) -> Sequence[Output]:

        self.columns = list(X)
        X_ = X[self.columns].values

        if not self.n_hidden:
            self.n_hidden = int(self.latent_pct*len(self.columns))

        if self.model is None and self.latent_pct:
            corex_disc.Corex(n_hidden= self.n_hidden, dim_hidden = self.dim_hidden,
                max_iter = self.max_iter, n_repeat = self.n_repeat, max_samples = self.max_samples,
                n_cpu = self.n_cpu, smooth_marginals= self.smooth_marginals, missing_values = self.missing_values, 
                verbose = self.verbose, seed = self.seed, **self.kwargs)

        if self.dim_hidden == 2:
            self.model.fit(X[self.columns].values)
            self.factors = self.model.transform(X_, details = True)[0]
            self.factors = np.transpose(np.squeeze(factors[:,:,1])) if len(factors.shape) == 3 else factors # assuming dim_hidden = 2
        else:
            self.factors = self.model.fit_transform(X_)

        self.fitted = True
        
        return self.factors

    def set_training_data(self, X : Sequence[Input]) -> None:
        self.training_inputs = X
        self.fitted = False

    def get_params(self) -> Params:
        return Params(latent_factors = self.factors)

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