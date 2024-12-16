# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the Hedge Defence`.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor import GaussianAugmentation
from art.defences.preprocessor.preprocessor import Preprocessor
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE
logger = logging.getLogger(__name__)


class Counter_Samples(Preprocessor):
    """
    The implemetation of CounterSamples presented in the paper : https://arxiv.org/abs/2403.10562
    It acts as a preprocessor, taking preprocessed samples (x_preprocessed), the number of optimization iterations (iters),
    and the step size (k), and returns the optimized (corrected) samples.
    """

    params = [
        "estimator",
        "step_size",
        "num_iter",
        'sigma'
    ]

    def __init__(
            self,
            estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
            step_size: float = 0.03,
            num_iter: int = 10,
            sigma= 0.01,
            gpu = True,
            apply_fit: bool = False,
            apply_predict: bool = True,
    ):
        """
        Initialize a CounterSamples Defence object.

        :param estimator: A trained classifier (PyTotrch).
        :param step_size: the step size in each iteration towards the direction that minimizes the loss.
        :param num_iter: The number of iterations the sample will be updated.
        :param apply_fit: True if applied during fitting/training. (Only supports Predict.)
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.estimator = estimator
        self.step_size = step_size
        self.gpu = gpu
        self.sigma = sigma
        self.num_iter = num_iter
        self._check_params()


    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
      # Apply noise first

      use_cuda = self.gpu
      model = self.estimator.model
      gaus = GaussianAugmentation(augmentation=False, sigma=0.01)
      device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
      x_preprocessed = torch.from_numpy(gaus(x)[0]).to(device)
      x_preprocessed.requires_grad_(True)
      x_preprocessed.retain_grad()
      loss = nn.CrossEntropyLoss(reduction='none')
      for iter in range(self.num_iter):
          # predicting labels
          model_output = model(x_preprocessed)
          true_labels_indexes = torch.argmax(model_output, dim=1)
          loss_comp = loss(model_output, true_labels_indexes)
          loss_comp.backward(torch.ones_like(loss_comp))
          # update the samples.
          x_preprocessed = x_preprocessed - (self.step_size * x_preprocessed.grad) # No normalization.
          x_preprocessed.retain_grad()
      return x_preprocessed.detach().cpu().numpy() , model_output.detach().cpu().numpy()

    def _check_params(self) -> None:
        if self.apply_fit:
            raise ValueError("This defence works only on predict.")
        if self.num_iter < 0:
            raise ValueError("The number of iterations parameter must be positive or zero.")
        if self.sigma < 0 :
            raise ValueError("The sigma parameter must be positive or zero.")
        if not isinstance(self.gpu,bool):
           raise ValueError("The gpu parameter must be boolean")

