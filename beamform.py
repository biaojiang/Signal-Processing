# -*- Functions for beamforming -*-
"""
Created on Wed Apr 20 10:30:29 2016

@author: biajia
"""
import cvxpy as cvx
import numpy as np

# define socp optimization
def socp_abf(U_RI, epsilon, Atar_RI, Atar_bar, solver_opt = 0):
    # Create cvxpy variables and constraints
    n = len(Atar_RI) # 2 * n 
    w_ri = cvx.Variable(n)
    tao = cvx.Variable()
    constraints = [ cvx.norm(U_RI * w_ri) <= tao]
    constraints.append(cvx.norm(w_ri) * epsilon <= w_ri.T * Atar_RI - 1)
    constraints.append(w_ri.T * Atar_bar == 0)
    
    # Form and solve problem.
    obj = cvx.Minimize(tao)
    prob = cvx.Problem(obj, constraints)
#    prob.solve(solver = cvx.CVXOPT) cvx.SCS
    if solver_opt == 0:    
        prob.solve(solver = cvx.CVXOPT)
    elif solver_opt == 1:
        prob.solve(solver = cvx.SCS)
    # Bisection (or fail).
    if prob.status == cvx.OPTIMAL:
        print('Problem is feasible for this epsilon = {}'.format(epsilon))
    
    elif prob.status == cvx.INFEASIBLE:
        print('Problem is not feasible for epsilon = {}'.format(epsilon))
    else:
        raise Exception('CVXPY Error')
    return w_ri

# SMI
def w_SMI(Atar_assume, R, DL = 0.0):
    N = len(Atar_assume)
    w = np.linalg.inv(R + DL * np.eye(N)).dot(Atar_assume)
    return w

# weight normalized
def w_SMI_norm(Atar_assume, R, DL = 0.0):
    N = len(Atar_assume)
    inv_R = np.linalg.inv(R + DL * np.eye(N))
    w = inv_R.dot(Atar_assume) / (Atar_assume.conj().dot(inv_R).dot(Atar_assume))
    return w
    
# optimum SINR
def SINR_opt(SNR, Atar_assume, R_in):
    # Atar_assume: assumed target steering vector
    # R_in: correlation matrix of interference-plus-noise
    SINR = 10**(SNR/10) * (Atar_assume.conj().dot(np.linalg.inv(R_in)).dot(Atar_assume)).real
    return SINR

# practical SINR
def SINR(SNR, Atar_assume, w, R_in):
    # w: beamforming weight vector
    SINR = 10**(SNR/10) * abs(w.conj().dot(Atar_assume))**2 / (w.conj().dot(R_in).dot(w)).real
    return SINR
    
# beam power
def beam_power(R, w):
    P = w.conj().dot(R).dot(w)
    return P.real



    