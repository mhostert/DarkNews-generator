import numpy as np
import pandas as pd
from scipy.stats import exponnorm

from . import math_vecs as mv
from . import exp_params as ep


# COMPUTE EFFICIENCY
def efficiency(samples, weights, xmin, xmax):

    mask = samples >= xmin & samples <= xmax
    
    weights_detected = weights[mask]

    return weights.sum() / weights_detected.sum()


# SMEARING
def smear_samples(samples, mass, EXP='miniboone'):
    
    # compute kinematic quantities
    E = samples['0'].values
    px = samples['1'].values
    py = samples['2'].values
    pz = samples['3'].values

    P = mv.modulus3([E,px,py,pz])

    # compute sigmas
    sigma_E = ep.STOCHASTIC[EXP]*np.sqrt(E) + ep.NOISE[EXP]
    sigma_angle = ep.ANGULAR[EXP]

    # compute kinetic energy and spherical angles
    T = E - mass
    theta = np.arccos(pz/P)
    phi = np.arctan2(py,px)

    if EXP=='miniboone':
        # compute smeared quantities
        T = np.random.normal(T, sigma_E)
        theta = np.random.normal(theta, sigma_angle)
        phi = np.random.normal(phi, sigma_angle)
    elif EXP=='microboone':
        #apply exponentially modified gaussian with exponential rate lambda = 1/K = 1 --> K=1
        K=1
        T = exponnorm.rvs(K, loc = T, scale = sigma_E)
        theta = exponnorm.rvs(K, loc = theta, scale = sigma_angle)
        phi = exponnorm.rvs(K, loc = phi, scale = sigma_angle)

    T[T < 0] = 1e-8 # force smearing to be positive for T
    E = T + mass
    P = np.sqrt(E**2 - mass**2)

    # put data in an array and then in a DataFrame
    smeared = np.array([E,P*np.sin(theta)*np.cos(phi),P*np.sin(theta)*np.sin(phi),P*np.cos(theta)])
    
    return smeared

