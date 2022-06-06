import numpy as np
import pandas as pd


# CONSTANTS
THRESHOLD = {'miniboone': 0.03, 'microboone': 0.01}
ANGLE_MAX = {'miniboone': 13., 'microboone': 5.}
ENU_MIN = {'miniboone': 0.14, 'microboone': 0.14}
ENU_MAX = {'miniboone': 1.5, 'microboone': 1.5}
EVIS_MIN = {'miniboone': 0.14, 'microboone': 0.14}
EVIS_MAX = {'miniboone': 3., 'microboone': 1.4}
STOCHASTIC = {'miniboone': 0.12, 'microboone': 0.12}
NOISE = {'miniboone': 0.01, 'microboone': 0.01}
ANGULAR = {'miniboone': 2.*np.pi/180.0, 'microboone': 2.*np.pi/180.0}
Q2 = {'miniboone': 1.e10, 'microboone': None}
ANALYSIS_TH = {'miniboone': 0.02, 'microboone': None}