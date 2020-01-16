#!/usr/bin/env python
from scipy.stats import skellam
import numpy as np
from scipy.optimize import minimize


class SkellamRegression:

    def __init__(self, x, y):
        self.y = y
        self.x = x

    def log_likelihood(self, coefficients):
        coefficients1 = coefficients[0:len(coefficients) // 2]
        coefficients2 = coefficients[len(coefficients) // 2:]
        lambda1 = self.x @ coefficients1
        lambda2 = self.x @ coefficients2
        neg_log_likelihood = -np.sum(skellam.logpmf(self.y, mu1=np.exp(lambda1), mu2=np.exp(lambda2), loc=0))
        return neg_log_likelihood

    def _train(self, x0):
        # initial estimate
        if x0 is None:
            x0 = np.ones(self.x.shape[1] * 2)
        else:
            if x0.shape[0] != self.x.shape[1] * 2:
                raise ValueError

        results = minimize(self.log_likelihood,
                           x0,
                           method="SLSQP",
                           options={'disp': True})

        self._results = results

    def train(self, x0=None):
        self._train(x0)

    def predict(self, x):
        lambda_1_coefficients = self._results.x[0:len(self._results.x) // 2]
        lambda_2_coefficients = self._results.x[len(self._results.x) // 2:]
        _lambda1 = x @ lambda_1_coefficients
        _lambda2 = x @ lambda_2_coefficients
        y_hat = _lambda1 - _lambda2
        return y_hat
