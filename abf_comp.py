# -*- Beampattern synthesization -*-
"""
Created on Sat Apr 16 18:03:36 2016

Minimize beamwidth of an array with arbitrary 2-D geometry

@author: biajia
"""
import beamform as bm
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
# Select array geometry:
#ARRAY_GEOMETRY = '2D_RANDOM'
ARRAY_GEOMETRY = '1D_UNIFORM_LINE'
#ARRAY_GEOMETRY = '2D_UNIFORM_LATTICE'

#
# Problem specs.
#
lambda_wl = 1         # wavelength
theta_tar = 93        # target direction
theta_tar_assume = 93
SNR = -10
theta_int = np.array([[120], [140]])
INR = np.array([[15], [15]])
N_snaps = 100 # maximum data snapshots
n_int = theta_int.size # number of interference
epsilon = 3
N_monte = 50 # Monte-carlo simulations
#
# 2D_RANDOM: 
#     n randomly located elements in 2D.
#
if ARRAY_GEOMETRY == '2D_RANDOM':
    # Set random seed for repeatable experiments.
    # Uniformly distributed on [0,L]-by-[0,L] square.
    np.random.seed(1)
    n = 36
    L = 5
    max_beam_ang = 360
    loc = L*np.random.random((n,2))

#
# 1D_UNIFORM_LINE:
#     Uniform 1D array with n elements with inter-element spacing d.
#
elif ARRAY_GEOMETRY == '1D_UNIFORM_LINE':
    n = 10
    d = 0.5*lambda_wl
    max_beam_ang = 180
#    loc = np.append(d * np.arange(n).reshape(n, 1), np.zeros((n,1)), axis = 1 )
#    loc = np.matrix(loc)
    loc = np.c_[d * np.arange(n), np.zeros(n)]                      
#
# 2D_UNIFORM_LATTICE:
#     Uniform 2D array with m-by-m element with d spacing.
#
elif ARRAY_GEOMETRY == '2D_UNIFORM_LATTICE':
    m = 6
    n = m**2
    d = 0.45*lambda_wl
    max_beam_ang = 360
    loc = np.zeros((n, 2))
    loc[:, 0] = np.tile(range(m), m).reshape(n, 1)
    loc[:, 1] = np.repeat(range(m), m).reshape(n, 1)
    loc = loc*d

else:
    raise Exception('Undefined array geometry')
#
# Construct optimization data.
#

# Build matrix A that relates w and y(theta), ie, y = A*w.
theta = np.arange(max_beam_ang)
A = np.kron(loc[:, 0], np.cos(np.pi * theta/180)) \
  + np.kron(loc[:, 1], np.sin(np.pi * theta/180))
A = np.exp(1j * 2 * np.pi / lambda_wl * A).reshape(n, max_beam_ang)

# Target constraint matrix.
ind_closest = np.argmin(np.abs(theta - theta_tar))
Atar = A[:, ind_closest] # target steering vector

ind_closest = np.argmin(np.abs(theta - theta_tar_assume))
Atar_assume = A[:, ind_closest] # target steering vector

A_int = np.zeros((n, n_int), dtype = np.complex128)
for i in range(n_int):
    ind_closest = np.argmin(np.abs(theta - theta_int[i, 0]))
    A_int[:, i] = A[:, ind_closest]

# start Monte-Carlo analysis
SINR = np.zeros((4, N_snaps))
i_monte = 0
while i_monte < N_monte:
    print('Current iteration {} of total {}'.format(i_monte, N_monte))
    # simulate target data
    data_tar = 1/np.sqrt(2) * 10**(SNR/20) * Atar.reshape(n, 1) * \
        (np.random.randn(1, N_snaps) + 1j*np.random.randn(1, N_snaps))
    # simulate interference-plus-noise
    data_in = 1/np.sqrt(2) * A_int.dot(10**(INR/20) * \
        (np.random.randn(2, N_snaps) + 1j*np.random.randn(2, N_snaps))) + \
        1/np.sqrt(2) * (np.random.randn(n, N_snaps) + 1j*np.random.randn(n, N_snaps))
    # interference-plus-noise correlation matrix
    R_in = data_in.dot(data_in.conj().transpose()) / N_snaps
 
    for i in range(N_snaps):
        R = (data_tar[:, :i +1] + data_in[:, :i + 1]).dot((data_tar[:, :i + 1] + data_in[:, :i + 1]).conj().transpose()) / (i +1)
    #    R_in = data_in[:, :i +1].dot(data_in[:, :i + 1].conj().transpose()) / (i + 1)
    
        if i < n - 1:    
            U = np.linalg.cholesky(R + 0.5 * np.eye(n)).conj().transpose()
        else:
            U = np.linalg.cholesky(R).conj().transpose()
    # Iterate bisection until 1 angular degree of uncertainty.
        # As of this writing (2014/05/14) cvxpy does not do complex valued math,
        # so the real and complex values must be stored seperately as reals
        # and operated on as follows:
        #     Let any vector or matrix be represented as a+bj, or A+Bj.
        #     Vectors are stored [a; b] and matrices as [A -B; B A]:
        
        # Atar as [A -B; B A]
        Atar_assume_R = Atar_assume.real
        Atar_assume_I = Atar_assume.imag
        Atar_assume_RI = np.r_[Atar_assume_R, Atar_assume_I]
        Atar_assume_bar = np.r_[Atar_assume_I, -Atar_assume_R]
    
        U_R, U_I = U.real, U.imag
        U_RI = np.r_[np.c_[U_R, -U_I], np.c_[U_I, U_R]]
        w_ri = bm.socp_abf(U_RI, epsilon, Atar_assume_RI, Atar_assume_bar, solver_opt = 1)
        w_socp = np.array(w_ri.value[:n] + 1j * w_ri.value[-n:])
        w_socp = w_socp[:,0]
        SINR[1, i] += bm.SINR(SNR, Atar_assume, w_socp, R_in)
        
    #    if i >= n - 1:
        w_SMI = bm.w_SMI(Atar_assume, R)
        SINR[3, i] += bm.SINR(SNR, Atar_assume, w_SMI, R_in)
        SINR[0, i] += bm.SINR_opt(SNR, Atar_assume, R_in)
        w_LSMI = bm.w_SMI(Atar_assume, R, 0.5)
        SINR[2, i]  += bm.SINR(SNR, Atar_assume, w_LSMI, R_in)
    i_monte += 1
SINR /= N_monte 

#%% Angular beam spectra
Beam = np.zeros((3, max_beam_ang))
for i in range(max_beam_ang):
    a_curr = A[:, i]
    a_curr_R = a_curr.real
    a_curr_I = a_curr.imag
    a_curr_RI = np.r_[a_curr_R, a_curr_I]
    a_curr_bar = np.r_[a_curr_I, -a_curr_R]
    
    w_ri_scan = bm.socp_abf(U_RI, epsilon, a_curr_RI, a_curr_bar, solver_opt = 1)
    w_socp_scan = np.array(w_ri_scan.value[:n] + 1j * w_ri_scan.value[-n:])
    w_socp_scan = w_socp_scan[:,0]    
    Beam[0, i] = bm.beam_power(R, w_socp_scan)
    
    w_LSMI = bm.w_SMI_norm(A[:, i], R, 0.5)
    Beam[1, i] = bm.beam_power(R, w_LSMI)
    w_SMI = bm.w_SMI_norm(A[:, i], R)
    Beam[2, i] = bm.beam_power(R, w_SMI)

#%%
# Show plot inline in ipython

# Plot properties.
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#%%
plt.close()
with plt.style.context('fivethirtyeight'):
    plt.plot(theta, 10 * np.log10(Beam[0:]).transpose())
    plt.legend(('SOCP-ABF', 'LSMI', 'SMI'), loc = 'best')
    plt.xlabel('Angle')
    plt.ylabel('Beam (dB)')
#    plt.ylim(ymin = -10, ymax = 1)
plt.tight_layout()
plt.show()

#%%
plt.close('all')
plt.figure(1)
x_snap = np.arange(1, N_snaps + 1)
SINR_dB = 10 * np.log10(SINR)
#plt.rc('lines', linewidth=4)
#plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
#                           cycler('linestyle', [':', '-', '-.', '--']))) 
with plt.style.context('fivethirtyeight'):
    plt.plot(x_snap, SINR_dB[0:].transpose())
    plt.legend(('Optimal SINR', 'SOCP-ABF', 'LSMI', 'SMI'), loc = 'lower right')
    plt.xlabel('Snapshots')
    plt.ylabel('Output SINR (dB)')
    plt.ylim(ymin = -10, ymax = 1)
    plt.tight_layout()
plt.show()
plt.savefig('sinr_vs_snapshots1.pdf', dpi = 150, transparent=False, bbox_inches = 'tight')
#%%
# First Figure: Antenna Locations
#
plt.figure(figsize=(6, 6))
plt.scatter(np.array(loc[:, 0]), np.array(loc[:, 1]), \
            s=30, facecolors='none', edgecolors='b')
plt.title('Antenna Locations', fontsize=16)
plt.tight_layout()
plt.show()

#
# Second Plot: Array Pattern
#

# Complex valued math to calculate y = A*w_im;
# See comments in code above regarding complex representation as reals.
#%%
y = w_socp.conj().dot(A)
#y = w_SMI.conj().dot(A)
#y = w_LSMI.conj().dot(A)
plt.figure(2)
with plt.style.context('ggplot'):
    ymin, ymax = -60, 5
    plt.plot(np.arange(max_beam_ang), 20*np.log10(np.abs(y) / np.max(np.abs(y))))
    plt.plot([theta_tar, theta_tar], [ymin, ymax], 'g--')
    for i in range(n_int):
        plt.plot([theta_int[i], theta_int[i]], [ymin, ymax], 'r--')
    plt.xlabel('Angle ($^{\circ}$)', fontsize=16)
    plt.ylabel(r'Beam (dB)', fontsize=16)
    plt.ylim(ymin, ymax)
    
    plt.tight_layout()
plt.show()
plt.savefig('beampattern.png', dpi = 150, transparent=False, bbox_inches = 'tight')
#
# Third Plot: Polar Pattern
#%%
plt.close(3)
plt.figure(3, figsize=(5, 2.5))
with plt.style.context('fivethirtyeight'):
    zerodB = 50
    dBY = 20*np.log10(np.abs(y)/np.max(np.abs(y))) + zerodB
    plt.plot(dBY * np.cos(np.pi*theta/180), \
             dBY * np.sin(np.pi*theta/180))
    plt.xlim(-zerodB, zerodB)
    if ARRAY_GEOMETRY == '1D_UNIFORM_LINE':
        plt.ylim(0, zerodB)
    else:
        plt.ylim(-zerodB, zerodB)

    plt.axis('off')
    
    # 0 dB level.
    plt.plot(zerodB * np.cos(np.pi*theta/180), \
             zerodB * np.sin(np.pi*theta/180), 'k:')
    plt.text(-zerodB,0,'0 dB', fontsize=16)
    # Max sideband level.
    min_sidelobe = -20
    m= zerodB + min_sidelobe
    plt.plot(m* np.cos(np.pi*theta/180), \
             m* np.sin(np.pi*theta/180), 'k:') 
    plt.text(-m,0,'{:.1f} dB'.format(min_sidelobe), fontsize=16)
    #Lobe center and boundaries angles.
    plt.plot([0, zerodB * np.cos(theta_tar*np.pi/180)], \
             [0, zerodB * np.sin(theta_tar*np.pi/180)], 'k:')
    plt.gca().axis('equal')         
    #Show plot.
    plt.tight_layout()
plt.show()
plt.savefig('beampattern_polar.png', dpi = 150, transparent=False, bbox_inches = 'tight')

#%% save data for Pgfplots
np.savetxt('sinr.txt', np.c_[x_snap, SINR_dB.transpose()], fmt='%g', delimiter=' ', newline='\n')
np.savetxt('beam_power.txt', np.c_[theta, 10 * np.log10(Beam[0:]).transpose()], fmt='%g', delimiter=' ', newline='\n') 
#%%




