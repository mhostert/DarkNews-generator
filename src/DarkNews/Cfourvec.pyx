#cython: boundscheck=False
#cython: language_level=3
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np

np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

#######################################
# C functions to be used
from libc.math cimport sqrt, abs, log, cos, sin, acos, pow

#######################################
# C implementation of RANDOM
from libc.stdlib cimport rand, RAND_MAX

cdef double UniformRand():
	return ( rand() + 1. )/(RAND_MAX + 1. )


#######################################
# Box muller method for NORMAL distributed random variables
cdef double NormalRand(double mean, double stddev):
	cdef double n2 = 0.0
	cdef int n2_cached = 0
	cdef double x, y, r
	cdef double result,d,n1

	if (not n2_cached):
		while (True):
			x = 2.0*UniformRand() - 1
			y = 2.0*UniformRand() - 1

			r = x*x + y*y

			if (r != 0.0 and r <= 1.0):
				break
		
		
		d = sqrt(-2.0*log(r)/r)
		n1 = x*d
		n2 = y*d
		result = n1*stddev + mean
		n2_cached = 1
		return result
	else:
		n2_cached = 0;
		return n2*stddev + mean

#######################################
# RANDOM FUNCTIONS
#######################################

#******************************
def random_generator(int size, double min, double max):
	cdef np.ndarray[DTYPE_t,ndim=1] s = np.empty((size), dtype=DTYPE)
	for i in range(size):
		s[i] = (max-min)*UniformRand()+min
	return s

#******************************
def random_normal(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] sigma):
	cdef int size = x.shape[0]
	cdef np.ndarray[DTYPE_t,ndim=1] s = np.empty((size), dtype=DTYPE)
	for i in range(size):
		s[i] = NormalRand(x[i], sigma[i])
	return s


#######################################
# FOURVECTOR FUNCTIONS
#######################################

#******************************
def build_fourvec(np.ndarray[DTYPE_t, ndim=1] E, np.ndarray[DTYPE_t, ndim=1] p, np.ndarray[DTYPE_t, ndim=1] cost, np.ndarray[DTYPE_t, ndim=1] phi):

	cdef int i,m
	m = phi.shape[0]
	cdef np.ndarray[DTYPE_t,ndim=2] s = np.empty((m,4), dtype=DTYPE)

	with nogil:
		for i in range(m):
			s[i,0] = E[i]
			s[i,1] = p[i]*cos(phi[i])*sqrt(1.0-cost[i]*cost[i])
			s[i,2] = p[i]*sin(phi[i])*sqrt(1.0-cost[i]*cost[i])
			s[i,3] = p[i]*cost[i]

	return s

#******************************
def momentum_scalar(np.ndarray[DTYPE_t, ndim=1] E, double mass):

	cdef int i,m
	m = E.shape[0]
	cdef np.ndarray[DTYPE_t,ndim=1] s = np.empty((m), dtype=DTYPE)

	with nogil:
		for i in range(m):
			s[i] = sqrt(E[i]*E[i] - mass*mass)
	return s

#******************************
def get_theta_3vec(np.ndarray[DTYPE_t, ndim=2] r):

	cdef int i,m
	m = r.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=1] s = np.empty((m), dtype=DTYPE)

	with nogil:
		for i in range(m):
			s[i] = acos(r[i,3]/sqrt(r[i,1]*r[i,1]+r[i,2]*r[i,2]+r[i,3]*r[i,3]))
	return s

#******************************
def mass(np.ndarray[DTYPE_t, ndim=2] x):
	cdef int i,m
	m= x.shape[0]
	cdef np.ndarray[DTYPE_t,ndim=1] s = np.empty((m), dtype=DTYPE)
	with nogil:
		for i in range(m):
			s[i] = x[i,0]*x[i,0] - x[i,1]*x[i,1] - x[i,2]*x[i,2] - x[i,3]*x[i,3]
		if s[i] <= 0.0:
			s[i]=0
		else:
			s[i]=sqrt(s[i])
	return  s

#******************************
def inv_mass(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=2] y):
	cdef int i,m
	m= x.shape[0]
	cdef np.ndarray[DTYPE_t,ndim=1] s = np.empty((m), dtype=DTYPE)
	with nogil:
		for i in range(m):
			s[i] = x[i,0]*y[i,0] - x[i,1]*y[i,1] - x[i,2]*y[i,2] - x[i,3]*y[i,3]
			if s[i] <= 0.0:
				s[i]=0
			else:
				s[i]=sqrt(s[i])
	return  s

#******************************
def dot4(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=2] y):
	cdef int i,m
	m= x.shape[0]
	cdef np.ndarray[DTYPE_t,ndim=1] s = np.empty((m), dtype=DTYPE)
	with nogil:
		for i in range(m):
			s[i] = x[i,0]*y[i,0] - x[i,1]*y[i,1] - x[i,2]*y[i,2] - x[i,3]*y[i,3]
	return  s

#******************************
def dot3(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=2] y):
	cdef int i,m
	m= x.shape[0]
	cdef np.ndarray[DTYPE_t,ndim=1] s = np.empty((m), dtype=DTYPE)
	with nogil:
		for i in range(m):
			s[i] = x[i,1]*y[i,1] + x[i,2]*y[i,2] + x[i,3]*y[i,3]
	return  s

#******************************
def dotXY(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=2] y):
	cdef int i,m
	m= x.shape[0]
	cdef np.ndarray[DTYPE_t,ndim=1] s = np.empty((m), dtype=DTYPE)
	with nogil:
		for i in range(m):
			s[i] =  x[i,1]*y[i,1] + x[i,2]*y[i,2]
	return  s

#******************************
def dotXY_vec(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=2] y):
	cdef int i,m
	m= x.shape[0]
	cdef np.ndarray[DTYPE_t,ndim=1] s = np.empty((m), dtype=DTYPE)
	with nogil:
		for i in range(m):
			s[i] =  x[i,0]*y[i,0] + x[i,1]*y[i,1]
	return  s

#******************************
def getXYnorm(np.ndarray[DTYPE_t, ndim=2] x):
	cdef int i,m
	m= x.shape[0]
	cdef np.ndarray[DTYPE_t,ndim=1] s = np.empty((m), dtype=DTYPE)
	with nogil:
		for i in range(m):
			s[i] =  sqrt(x[i,1]*x[i,1] + x[i,2]*x[i,2])
	return  s
#******************************
def getXYnorm_3vec(np.ndarray[DTYPE_t, ndim=2] x):
	cdef int i,m
	m= x.shape[0]
	cdef np.ndarray[DTYPE_t,ndim=1] s = np.empty((m), dtype=DTYPE)
	with nogil:
		for i in range(m):
			s[i] =  sqrt(x[i,0]*x[i,0] + x[i,1]*x[i,1])
	return  s

#******************************
def get_vec_norm(np.ndarray[DTYPE_t, ndim=2] x):
	cdef int i,m
	m= x.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=1] s = np.empty((m), dtype=DTYPE)
	with nogil:
		for i in range(m):
			s[i] = sqrt(x[i,0]*x[i,0] + x[i,1]*x[i,1] + x[i,2]*x[i,2])
	return s

#******************************
def get_3vec_norm(np.ndarray[DTYPE_t,ndim=2] x):
	cdef int i,m
	m= x.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=1] s = np.empty((m), dtype=DTYPE)
	with nogil:
		for i in range(m):
			s[i] = sqrt(x[i,1]*x[i,1]+x[i,2]*x[i,2]+x[i,3]*x[i,3])
	return s

#******************************
def get_3norm_vec(np.ndarray[DTYPE_t,ndim=2] x):
	cdef int i,m
	m= x.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=1] s = np.empty((m), dtype=DTYPE)
	with nogil:
		for i in range(m):
			s[i] = sqrt(x[i,0]*x[i,0]+x[i,1]*x[i,1]+x[i,2]*x[i,2])
	return s

#******************************
def get_3direction_3vec(np.ndarray[DTYPE_t, ndim=2] x):
	cdef int i,m
	m= x.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=2] s = np.empty((m,3), dtype=DTYPE)
	with nogil:
		for i in range(m):
			s[i,0] = x[i,0]/sqrt(x[i,0]*x[i,0]+x[i,1]*x[i,1]+x[i,2]*x[i,2])
			s[i,1] = x[i,1]/sqrt(x[i,0]*x[i,0]+x[i,1]*x[i,1]+x[i,2]*x[i,2])
			s[i,2] = x[i,2]/sqrt(x[i,0]*x[i,0]+x[i,1]*x[i,1]+x[i,2]*x[i,2])
	return s

#******************************
def get_cosTheta(np.ndarray[DTYPE_t, ndim=2] x):
	cdef int i,m
	m= x.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=1] s = np.empty((m), dtype=DTYPE)
	with nogil:
		for i in range(m):
			s[i] = x[i,3]/sqrt(x[i,1]*x[i,1] + x[i,2]*x[i,2] + x[i,3]*x[i,3])
	return s

#******************************
def get_cos_opening_angle(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=2] y):
	cdef int i,m
	m= x.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=1] s = np.empty((m), dtype=DTYPE)
	with nogil:
		for i in range(m):
			s[i] = (x[i,1]*y[i,1] + x[i,2]*y[i,2] + x[i,3]*y[i,3])/sqrt(x[i,1]*x[i,1] + x[i,2]*x[i,2] + x[i,3]*x[i,3])/sqrt(y[i,1]*y[i,1] + y[i,2]*y[i,2] + y[i,3]*y[i,3])
	return s

#******************************
def get_3direction(np.ndarray[DTYPE_t, ndim=2] x):
	cdef int i,m
	m= x.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=2] s = np.empty((m,3), dtype=DTYPE)
	with nogil:
		for i in range(m):
			s[i,0] = x[i,1]/sqrt(x[i,1]*x[i,1]+x[i,2]*x[i,2]+x[i,3]*x[i,3])
			s[i,1] = x[i,2]/sqrt(x[i,1]*x[i,1]+x[i,2]*x[i,2]+x[i,3]*x[i,3])
			s[i,2] = x[i,3]/sqrt(x[i,1]*x[i,1]+x[i,2]*x[i,2]+x[i,3]*x[i,3])
	return s

#******************************
def put_in_z_axis(np.ndarray[DTYPE_t, ndim=1] x):
	cdef int i,m
	m= x.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=2] s = np.empty((m,3), dtype=DTYPE)
	with nogil:
		for i in range(m):
			s[i,0] = 0.0
			s[i,1] = 0.0
			s[i,2] = x[i]
	return s

#******************************
def rotationx(np.ndarray[DTYPE_t, ndim=2] v4, np.ndarray[DTYPE_t, ndim=1] theta):

	cdef int i, m;

	m = v4.shape[0]

	cdef np.ndarray[DTYPE_t, ndim=2] res = np.empty((m,4), dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=3] R = np.empty((m,4,4), dtype=DTYPE)

	with nogil:
		for i in range(m):
			R[i,0,0] = 1.0
			R[i,0,1] = 0.0
			R[i,0,2] = 0.0
			R[i,0,3] = 0.0

			R[i,1,0] = 0.0
			R[i,1,1] = 1.0
			R[i,1,2] = 0.0
			R[i,1,3] = 0.0

			R[i,2,0] = 0.0
			R[i,2,1] = 0.0
			R[i,2,2] = cos(theta[i])
			R[i,2,3] = -sin(theta[i])

			R[i,3,0] = 0.0
			R[i,3,1] = 0.0
			R[i,3,2] = sin(theta[i])
			R[i,3,3] = cos(theta[i])

			res[i,0] = R[i,0,0]*v4[i,0] + R[i,0,1]*v4[i,1] + R[i,0,2]*v4[i,2] + R[i,0,3]*v4[i,3]
			res[i,1] = R[i,1,0]*v4[i,0] + R[i,1,1]*v4[i,1] + R[i,1,2]*v4[i,2] + R[i,1,3]*v4[i,3]
			res[i,2] = R[i,2,0]*v4[i,0] + R[i,2,1]*v4[i,1] + R[i,2,2]*v4[i,2] + R[i,2,3]*v4[i,3]
			res[i,3] = R[i,3,0]*v4[i,0] + R[i,3,1]*v4[i,1] + R[i,3,2]*v4[i,2] + R[i,3,3]*v4[i,3]
		    
	return res

#******************************
def rotationy(np.ndarray[DTYPE_t, ndim=2] v4, np.ndarray[DTYPE_t, ndim=1] theta):

	cdef int i, m;
	m = v4.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=2] res = np.empty((m,4), dtype=DTYPE)

	with nogil:
		for i in range(m):
			res[i,0] = v4[i,0]
			res[i,1] = cos(theta[i])*v4[i,1] - sin(theta[i])*v4[i,3]
			res[i,2] = v4[i,2]
			res[i,3] = sin(theta[i])*v4[i,1] + cos(theta[i])*v4[i,3]
		    
	return res
#******************************
def rotationy_sin(np.ndarray[DTYPE_t, ndim=2] v4, np.ndarray[DTYPE_t, ndim=1] stheta):

	cdef int i, m;
	m = v4.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=2] res = np.empty((m,4), dtype=DTYPE)

	with nogil:
		for i in range(m):
			res[i,0] = v4[i,0]
			res[i,1] = sqrt(1.0-stheta[i]*stheta[i])*v4[i,1] - stheta[i]*v4[i,3]
			res[i,2] = v4[i,2]
			res[i,3] = stheta[i]*v4[i,1] + sqrt(1.0-stheta[i]*stheta[i])*v4[i,3]
		    
	return res
#******************************
def rotationy_cos(np.ndarray[DTYPE_t, ndim=2] v4, np.ndarray[DTYPE_t, ndim=1] ctheta, int sign=1):

	cdef int i, m;
	m = v4.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=2] res = np.empty((m,4), dtype=DTYPE)

	with nogil:
		for i in range(m):
			res[i,0] = v4[i,0]
			res[i,1] = ctheta[i]*v4[i,1] - sign*sqrt(1.0-ctheta[i]*ctheta[i])*v4[i,3]
			res[i,2] = v4[i,2]
			res[i,3] = sign*sqrt(1.0-ctheta[i]*ctheta[i])*v4[i,1] + ctheta[i]*v4[i,3]
		    
	return res

#******************************
def rotationz(np.ndarray[DTYPE_t, ndim=2] v4, np.ndarray[DTYPE_t, ndim=1] theta):

	cdef int i, m;
	m = v4.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=2] res = np.empty((m,4), dtype=DTYPE)
	with nogil:
		for i in range(m):
			res[i,0] = v4[i,0]
			res[i,1] = cos(theta[i])*v4[i,1] - sin(theta[i])*v4[i,2] 
			res[i,2] = sin(theta[i])*v4[i,1] + cos(theta[i])*v4[i,2] 
			res[i,3] = v4[i,3]
		    
	return res
#******************************
def rotationz_sin(np.ndarray[DTYPE_t, ndim=2] v4, np.ndarray[DTYPE_t, ndim=1] stheta):

	cdef int i, m;
	m = v4.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=2] res = np.empty((m,4), dtype=DTYPE)
	with nogil:
		for i in range(m):
			res[i,0] = v4[i,0]
			res[i,1] = sqrt(1.0-stheta[i]*stheta[i])*v4[i,1] - stheta[i]*v4[i,2] 
			res[i,2] = stheta[i]*v4[i,1] + sqrt(1.0-stheta[i]*stheta[i])*v4[i,2] 
			res[i,3] = v4[i,3]
		    
	return res
#******************************
def rotationz_cos(np.ndarray[DTYPE_t, ndim=2] v4, np.ndarray[DTYPE_t, ndim=1] ctheta, int sign = 1):

	cdef int i, m;
	m = v4.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=2] res = np.empty((m,4), dtype=DTYPE)
	with nogil:
		for i in range(m):
			res[i,0] = v4[i,0]
			res[i,1] = ctheta[i]*v4[i,1] - sign*sqrt(1.0-ctheta[i]*ctheta[i])*v4[i,2] 
			res[i,2] = sign*sqrt(1.0-ctheta[i]*ctheta[i])*v4[i,1] + ctheta[i]*v4[i,2] 
			res[i,3] = v4[i,3]
		    
	return res

#******************************
def L(np.ndarray[DTYPE_t, ndim=2] v4, np.ndarray[DTYPE_t, ndim=1] beta):
	cdef int i, m;
	m = beta.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=2] res = np.empty((m,4), dtype=DTYPE)
	with nogil:
		for i in range(m):
			res[i,0] = 1.0/sqrt(1.0 - beta[i]*beta[i])*v4[i,0] - beta[i]/sqrt(1.0 - beta[i]*beta[i])*v4[i,3]
			res[i,1] = v4[i,1] 
			res[i,2] = v4[i,2] 
			res[i,3] = -beta[i]/sqrt(1.0 - beta[i]*beta[i])*v4[i,0] + 1.0/sqrt(1.0 - beta[i]*beta[i])*v4[i,3]
		    
	return res

#******************************
def T(np.ndarray[DTYPE_t, ndim=2] v4, np.ndarray[DTYPE_t, ndim=1] beta, np.ndarray[DTYPE_t, ndim=1] theta, np.ndarray[DTYPE_t, ndim=1] phi):
	return L( rotationy( rotationz(v4,-phi), theta), -beta)

#******************************
def Tinv(np.ndarray[DTYPE_t, ndim=2] v4, np.ndarray[DTYPE_t, ndim=1] beta, np.ndarray[DTYPE_t, ndim=1] ctheta, np.ndarray[DTYPE_t, ndim=1] phi):
	return rotationz( rotationy_cos( L(v4, beta), ctheta, sign=-1), phi)

