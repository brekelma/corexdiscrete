from sklearn import preprocessing
#import primitive
import sys
#sys.path.append('corexdiscrete/')
import corexdiscrete.corexdiscrete.corex as corex_disc
#import bio_corex.corex as corex_disc
from collections import defaultdict
from scipy import sparse
import pandas as pd
import numpy as np

from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from typing import NamedTuple, Union, Optional, Sequence


Input = pd.DataFrame
Output = np.ndarray
Params = NamedTuple('Params', [
    ('latent_factors', np.ndarray), 
])

class CorexDiscrete(UnsupervisedLearnerPrimitiveBase):  #(Primitive):

    def __init__(self, n_hidden: int = None, dim_hidden : int = 2, latent_pct : float = None, max_iter : int = 100, 
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
            self.n_hidden = None

    def fit(self, timeout: float = None, iterations : int = None) -> None:
        if self.fitted:
            return
        if not hasattr(self, 'training_inputs'):
            raise ValueError("Missing training data.")

        self.fit_transform(self.training_inputs, timeout, iterations)
        self.fitted = True
        return self

    def produce(self, inputs : Sequence[Input], timeout : float = None, iterations : int = None) -> Sequence[Output]: 

        self.columns = list(inputs)
        X_ = inputs[self.columns].values # TO DO: add error handling for X[self.columns]?

        if iterations is not None:
            self.max_iter = iterations
        
        if not self.fitted:
            self.n_hidden = int(self.latent_pct*len(self.columns))
            self.model = corex_disc.Corex(n_hidden= self.n_hidden, dim_hidden = self.dim_hidden,
                max_iter = self.max_iter, n_repeat = self.n_repeat, max_samples = self.max_samples,
                n_cpu = self.n_cpu, smooth_marginals= self.smooth_marginals, missing_values = self.missing_values, 
                verbose = self.verbose, seed = self.seed, **self.kwargs)
            self.model.fit(X_)
            self.fitted = True

        if self.dim_hidden == 2:
            self.latent_factors = self.model.transform(X_, details = True)[0]
            self.latent_factors = np.transpose(np.squeeze(self.latent_factors[:,:,1])) if len(self.latent_factors.shape) == 3 else self.latent_factors # assuming dim_hidden = 2
        else:
            self.latent_factors = self.model.transform(X_)

        return self.latent_factors

    def fit_transform(self, inputs : Sequence[Input], timeout : float = None, iterations : int = None) -> Sequence[Output]:
        self.columns = list(inputs)
        X_ = inputs[self.columns].values

        if iterations is not None:
            self.max_iter = iterations

        if self.n_hidden is None:
            self.n_hidden = int(self.latent_pct*len(self.columns))

        if self.model is None and self.latent_pct:
            self.model = corex_disc.Corex(n_hidden= self.n_hidden, dim_hidden = self.dim_hidden,
                max_iter = self.max_iter, n_repeat = self.n_repeat, max_samples = self.max_samples,
                n_cpu = self.n_cpu, smooth_marginals= self.smooth_marginals, missing_values = self.missing_values, 
                verbose = self.verbose, seed = self.seed, **self.kwargs)

        if self.dim_hidden == 2:
            self.model.fit(X_)
            self.latent_factors = self.model.transform(X_, details = True)[0]
            self.latent_factors = np.transpose(np.squeeze(self.latent_factors[:,:,1])) if len(self.latent_factors.shape) == 3 else self.latent_factors # assuming dim_hidden = 2
        else:
            self.latent_factors = self.model.fit_transform(X_)

        self.fitted = True
        
        return self.latent_factors

    def set_training_data(self, inputs : Sequence[Input]) -> None:
        self.training_inputs = inputs
        self.fitted = False

    def get_params(self) -> Params:
        return Params(latent_factors = self.latent_factors)

    def set_params(self, params: Params) -> None:
        self.latent_factors = params.latent_factors


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