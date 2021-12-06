import os
import sys
import numpy as np
import pandas as pd
import random 

from DarkNews import logger


from . import const
from . import pdg
#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )
from . import Cfourvec as Cfv

def geometry_zero_origin(size):
	# generating entries
	x = np.zeros(size)
	y = np.zeros(size)
	z = np.zeros(size)
	t = np.zeros(size)

	return t,x,y,z


def geometry_muboone(size):

	# Tube parameters
	r_t = 191.61
	z_t  = 1086.49

	# cap parameters
	r_c = 305.250694958
	theta_c = 38.8816337686*const.deg_to_rad
	ctheta_c = np.cos(theta_c)
	stheta_c = np.sin(theta_c)
	zend_c = 305.624305042
	h = r_c*(1.-ctheta_c)
	cap_gap = 0.37361008


	# SAMPLE A HUGE RECTANGLE
	xmin=-200;xmax=200. # cm
	ymin=-200.;ymax=200. # cm
	zmin=-800.;zmax=800. # cm
	tries=0
	npoints=0
	
	# arrays for accepted points
	x_accept = np.empty(0)
	y_accept = np.empty(0)
	z_accept = np.empty(0)

	while npoints < size:
	    # generating entries
	    x = xmin + (xmax - xmin)*np.random.rand(size)
	    y = ymin + (ymax - ymin)*np.random.rand(size) 
	    z = zmin + (zmax - zmin)*np.random.rand(size) 

	    # accept in tube
	    r_polar = np.sqrt(x**2+y**2)
	    mask_tube = (-z_t/2<z)&(z<z_t/2)&(r_polar < r_t)

	    # coordinates in sphere 1
	    z1 = z  - (-z_t/2-cap_gap-h + r_c)
	    r1  = np.sqrt(r_polar**2 + z1**2)
	    inc1 = np.arctan2(z1,r_polar) # inclination angle
	    # accept in front cap
	    mask_cap1 = ((r1<r_c)&(inc1<theta_c)&(z1<-r_c+h))

	    # coordinates in sphere 2
	    z2 = z  + (-z_t/2-cap_gap-h + r_c)
	    r2  = np.sqrt(r_polar**2 + z2**2)
	    inc2 = np.arctan2(-z2,r_polar) # inclination angle
	    # accept in back cap
	    mask_cap2 = ((r2<r_c)&(inc2<theta_c)&(z2>r_c-h))

	    mask_full = (mask_tube+mask_cap1+mask_cap2)

	    x_accept = np.append(x_accept, x[mask_full])
	    y_accept = np.append(y_accept, y[mask_full])
	    z_accept = np.append(z_accept, z[mask_full])
	    
	    npoints += np.size(x[mask_full])
	    tries+=size
	    if tries > 1e3*size:
	        logger.error("Geometry sampled too inefficiently. Geometry not specified correctly?")
	        sys.exit(1) 


	#######
	# Parameters from Mark -- neutrino time spill, overlapping w Genie BNB
	GlobalTimeOffset = 3125.
	RandomTimeOffset = 1600.
	time = GlobalTimeOffset + (RandomTimeOffset)*np.random.rand(size)

	# guarantee that array has number of samples asked (size)
	return np.array([time, x_accept[:size], y_accept[:size], z_accept[:size]])
	