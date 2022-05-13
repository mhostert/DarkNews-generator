import numpy as np

from ToyAnalysis import toy_logger

#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )
from . import Cfourvec as Cfv

X = 0 
Y = 1
Z = 2

kz = [0,0,0,1]
kx = [0,1,0,0]
k0 = [1,0,0,0]

##############################
# numpy functions
def dot4(x,y):
	if (np.size(x)!=4 or np.size(y)!=4):
		print("ERROR!")
		return 0
	return x[0]*y[0] - x[1]*y[1] - x[2]*y[2] - x[3]*y[3]

def dot3(x,y):
	if (np.size(x)!=4 or np.size(y)!=4):
		print("ERROR!")
	return x[1]*y[1] + x[2]*y[2] + x[3]*y[3]

def cos_opening_angle(x,y):
	return Cfv.get_cos_opening_angle(x,y)

def cos_azimuthal(x):
	return Cfv.get_cosTheta(x)

def inv_mass(x,y):
	return np.sqrt(Cfv.dot4(x,y))

def get_3vec(x):
	return x[1:]

def get_direction(x):
	if (np.size(x)!=4):
		print("ERROR! Wrong size np.shape(x):", np.shape(x))
		return 0
	return get_3vec(x)/np.sqrt(dot3(x,x))

##############################
# dataframe functions
def df_dot4(dx,dy):
	return Cfv.dot4(dx.to_numpy(), dy.to_numpy())

def df_dot3(dx,dy):
	return Cfv.dot3(dx.to_numpy(), dy.to_numpy())

def df_cos_opening_angle(dx,dy):
	return Cfv.get_cos_opening_angle(dx.to_numpy(),dy.to_numpy())

def df_cos_azimuthal(dx):
	return cos_azimuthal(dx.to_numpy())

def df_inv_mass(dx,dy):
	return inv_mass(dx.to_numpy(),dy.to_numpy())


##############################
# transformation functions
def R(v4, theta, i):
	c, s = np.cos(theta), np.sin(theta)
	if i ==X:
		R = np.array( ((1.0, 0, 0, 0), (0, 1.0, 0, 0), (0, 0, c, -s), (0, 0, s, c)) )
	if i ==Y:
		R = np.array( ((1.0, 0, 0, 0), (0, c, 0, -s), (0, 0, 1.0, 0), (0, s, 0, c)) )
	if i ==Z:
		R = np.array( ((1.0, 0, 0, 0), (0, c, -s, 0), (0, s, c, 0), (0, 0, 0, 1.0)) )
	return R.dot(v4)

def L(v4,beta):
	gamma = 1.0/np.sqrt(1.0 - beta*beta)
	R = np.array( ((gamma, 0, 0, -gamma*beta), (0, 1, 0, 0), (0, 0, 1, 0), (-gamma*beta, 0, 0, gamma)) )
	return R.dot(v4)

def T(v4, beta, theta, phi):
	return L( R( R(v4,-phi, Z), theta, Y), -beta)

def Tinv(v4, beta, theta, phi):
	return R( R( L(v4, beta), -theta, Y), phi, Z)
