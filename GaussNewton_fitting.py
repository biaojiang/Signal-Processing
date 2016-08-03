# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 10:40:55 2016
Use Gauss-Newton algorithm to fit a model to some data by minimizing 
the sum of squares of errors between the data and model's predictions.

Tested with biology data which relate the substrate concentration [S] 
to the reaction rate in an enzyme-mediated reaction.
https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm

@author: biajia
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pylab as plt
# define problem class
class enzyme_reaction:
    def __init__(self, beta):
        self.beta = beta
   
    def enzyme(self, beta, x, y):
        r = y - beta[0] * x / (beta[1] + x)
        return r

    # define Jacobian derivative func
    def enzyme_der(self, beta, x, y): #
        m, n = x.shape[0], beta.shape[0]
        der = np.zeros((m, n))
        der[:, 0] = -x / (beta[1] + x)
        der[:, 1] = beta[0] * x / (beta[1] + x) ** 2
        return der

    # MLE
    def gauss_newton(self, beta0, x, y):
#        res = minimize(self.enzyme, beta0, args = (x, y), method='BFGS', jac = self.enzyme_der, \
        options={'xtol': 1e-6, 'maxiter': 100, 'disp': True}
        xtol = 1
        iter = 0
        beta = np.zeros(2)
        beta[:] = beta0[:]
        while xtol > options['xtol'] and iter < options['maxiter']:
            jac = self.enzyme_der(beta, x, y)
            r = self.enzyme(beta, x, y)
            beta -= np.linalg.inv(jac.T.dot(jac)).dot(jac.T).dot(r)
            xtol = np.linalg.norm((beta - self.beta) / self.beta)
            self.beta[:] = beta[:]
            iter += 1
        print('Interations: {0:d} \nRelative error: {1:g}'.format(iter, xtol))
        return self.beta

# substrate concentration
x = np.array([0.038, 0.194, 0.425, 0.626, 1.253, 2.500, 3.740])
# rate
y = np.array([0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317])

beta0 = np.array([0.9, 0.2])
enzyme = enzyme_reaction(beta0)
beta_est = enzyme.gauss_newton(beta0, x, y)

y_est = beta_est[0] * x / (beta_est[1] + x)
#%%
plt.close('all')
plt.figure(1)
plt.plot(x, y, 'rd', x, y_est, 'b', markersize=10, markeredgecolor = 'none', linewidth = 2)
plt.xlabel('[S]')
plt.ylabel('Reaction rate (m)')
plt.savefig('enzyme reaction data fitting.png', dpi = 150, transparent=True, bbox_inches = 'tight')