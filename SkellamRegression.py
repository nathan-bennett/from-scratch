#!/usr/bin/env python
from scipy.stats import poisson, skellam
import numpy as np
import matplotlib.pyplot as plt
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

    def train(self):
        # initial estimate
        x0 = np.ones(self.x.shape[1] * 2)

        results = minimize(self.log_likelihood,
                           x0,
                           method="Nelder-Mead",
                           options={'disp': True})

        return results
