import numpy as np
import pandas as pd

# SCALAR PRODUCTS
def dot4(v1, v2):
    return v1[0]*v2[0] - v1[1]*v2[1] - v1[2]*v2[2] - v1[3]*v2[3]

def dot3(v1, v2):
    return v1[1]*v2[1] + v1[2]*v2[2] + v1[3]*v2[3]

# MODULUS OF VECTORS
def modulus4(v):
    return np.sqrt(np.abs(dot4(v,v)))

def modulus3(v):
    return np.sqrt(dot3(v,v))