# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:40:11 2016
Conventional linear Kalman filter 
@author: biajia
"""

import numpy as np
# define kalman class
class kalman:
    def __init__(self, A, B, H, x, P, Q, R):
        self.A = A  # state transition matrix
        self.B = B  # control matrix
        self.H = H  # observation matrix
        self.x_curr = x  # initial state
        self.P_curr = P  # initial state covariance
        self.Q = Q  # estimated error in process
        self.R = R  # estimated error in ovservation

    def GetCurrentState(self):
        return self.x_curr

    # prediction
    def kalman_filt(self, control_vec, obs_vec):
        x_est = self.A.dot(self.x_curr) + self.B.dot(control_vec)
        P_est = self.A.dot(self.P_curr).dot(self.A.T) + self.Q

        # innovation
        innovation = obs_vec - self.H.dot(x_est)
        innovation_cov = self.H.dot(P_est).dot(self.H.T) + self.R
        kalman_gain = P_est.dot(self.H.T).dot(np.linalg.inv(innovation_cov))
    
        # update
        self.x_curr = x_est + kalman_gain.dot(innovation)
        n = self.x_curr.shape[0]
        self.P_curr = (np.eye(n) - kalman_gain.dot(self.H)).dot(P_est)
        
