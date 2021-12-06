""" Constant modules

Here are defined the most common constants used in darknews.

We use PDG2020 values for constants and SM masses.

Some low-level auxialiary functions are defined.

"""
import sys
import subprocess
import numpy as np
from numpy import sqrt
import math
from scipy import interpolate
import os
import itertools
import logging

from DarkNews import logger
from DarkNews import local_dir

#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )
from . import Cfourvec as Cfv


#####################################
# particle names and props
from particle import Particle
from particle import literals as lp

N4 = 'N4'
N5 = 'N5'
N6 = 'N6'

################################################
# Masses 
m_proton = 0.93827208816 # GeV
m_neutron = 0.93956542052 # GeV
m_avg = (m_proton+m_neutron)/2. # GeV

m_W = 80.37912 # GeV
m_Z = 91.187621 # GeV

m_e =  0.5109989500015e-3 # GeV
m_mu =  0.1134289257 # GeV
m_tau =  1.77682 # GeV

# charged hadrons
m_charged_pion = 0.1396
m_charged_rho = 0.7758

# neutral hadrons
m_neutral_pion = 0.135
m_neutral_eta = 0.5478
m_neutral_rho = 0.7755

m_neutral_B = 5.27958
m_charged_B = 5.27958

m_neutral_kaon = 0.497611
m_charged_kaon = 0.4937
m_charged_kaonstar = 0.892

################################################
# QED
alphaQED = 1.0/137.03599908421 # Fine structure constant at q2 -> 0
eQED = np.sqrt((4*np.pi)*alphaQED)

# get running alphaQED
Q, inv_alphaQED = np.genfromtxt(f'{local_dir}/aux_data/alphaQED/alpha_QED_running_posQ2.dat',unpack=True)
runningAlphaQED = interpolate.interp1d(Q,1.0/inv_alphaQED)


################################################
# Weak sector
Gf =1.16637876e-5 # Fermi constant (GeV^-2)
gweak = np.sqrt(Gf*m_W**2*8/np.sqrt(2))
s2w = 0.22343 # On-shell
sw = np.sqrt(s2w)
cw = np.sqrt(1. - s2w)

################################################
# Higgs -- https://pdg.lbl.gov/2019/reviews/rpp2018-rev-higgs-boson.pdf
vev_EW = 1/np.sqrt(np.sqrt(2)*Gf)
m_H = 125.10
lamb_quartic = (m_H/vev_EW)**2/2
m_h_potential = - lamb_quartic*vev_EW**2

################################################
# Mesons
fcharged_pion = 0.1307
fcharged_kaon = 0.1598
fcharged_rho = 0.220

fneutral_pion = 0.130
fneutral_kaon = 0.164
fneutral_B = 0.1909
fcharged_B = 0.1871
fneutral_Bs = 0.2272
fneutral_eta = 0.210

Fneutral_pion = fneutral_pion/np.sqrt(2.0)
Fneutral_kaon = fneutral_kaon/np.sqrt(2.0)
Fneutral_B = fneutral_B/np.sqrt(2.0)
Fcharged_B = fcharged_B/np.sqrt(2.0)
Fneutral_Bs = fneutral_Bs/np.sqrt(2.0)
Fneutral_eta = fneutral_eta/np.sqrt(2.0)


################################################
# CKM elements
# PDG2019
lamCKM = 0.22453;
ACKM = 0.836;
rhoBARCKM = 0.122;
etaBARCKM = 0.355;
rhoCKM = rhoBARCKM/(1-lamCKM*lamCKM/2.0);
etaCKM = etaBARCKM/(1-lamCKM*lamCKM/2.0);

s12 = lamCKM;
s23 = ACKM*lamCKM**2;
s13e = ACKM*lamCKM**3*(rhoCKM + 1j*etaCKM)*np.sqrt(1.0 - ACKM**2*lamCKM**4)/(np.sqrt(1.0 - lamCKM**2)*(1.0 - ACKM**2*lamCKM**4*(rhoCKM + 1j*etaCKM) ))
c12 = np.sqrt(1 - s12**2);
c23 = np.sqrt(1 - s23**2);
c13 = np.sqrt(1 - abs(s13e)**2);

Vud = c12*c13;
Vus = s12*c13;
Vub = np.conj(s13e);
Vcs = c12*c23-s12*s23*s13e;
Vcd = -s12*c23-c12*s23*s13e;
Vcb = s23*c13;
Vts = -c12*s23-s12*c23*s13e;
Vtd = s12*s23-c12*c23*s13e;
Vtb = c23*c13;


################################################
# speed of light cm/s
c_LIGHT = 29979245800

################################################
# constants for normalization
invm2_to_incm2=1e-4
fb_to_cm2 = 1e-39
NAvo = 6.02214076*1e23
tons_to_nucleons = NAvo*1e6/m_avg
rad_to_deg = 180.0/np.pi
deg_to_rad= 1/rad_to_deg

invGeV2_to_cm2 = 3.89379372e-28 # hbar c = 197.3269804e-16 GeV.cm
cm2_to_attobarn = 1e42
attobarn_to_cm2 = 1e-42
invGeV2_to_attobarn = invGeV2_to_cm2*cm2_to_attobarn

invGeV_to_cm = np.sqrt(invGeV2_to_cm2)
fm_to_GeV = 1/invGeV_to_cm*1e-15*1e2
invGeV_to_s = invGeV_to_cm/c_LIGHT
hb = 6.582119569e-25 # hbar in Gev s


################
# DM velocity at FREEZOUT
V_DM_FREEZOUT = 0.3 # c 
# SIGMAV_FREEZOUT = 4.8e-26 # cm^3/s
SIGMAV_FREEZOUT = 6e-26 # cm^3/s
GeV2_to_cm3s = invGeV2_to_cm2*c_LIGHT*1e2



################################################
# low-level auxiliary functions
def is_odd(num):
    return num & 0x1
np_is_odd = np.vectorize(is_odd)

# https://newbedev.com/how-do-i-annotate-with-power-of-ten-formatting vvv
# Define function for string formatting of scientific notation
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if num != 0:
        if exponent is None:
            exponent = int(math.floor(math.log10(abs(num))))
        coeff = round(num / float(10**exponent), decimal_digits)
        if precision is None:
            precision = decimal_digits

        return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)
    else:
        return r"0"
# ^^^



def ConfigureLogger(logger, level=logging.INFO, prettyprinter = None, logfile = None, verbose=False):
    ''' 
    Configure the DarkNews logger 
        
        logger --> main logger of DarkNews. It handles all debug, info, warning, and error messages

        prettyprint --> secondary logger for pretty print messages. Cannot override the main logger level

    '''
    
    logger.setLevel(level)

    if logfile:
        # log to files with max 1 MB with up to 4 files of backup
        handler = logging.handlers.RotatingFileHandler(f"{logfile}", maxBytes=1000000, backupCount=4)

    else:
        # stdout only
        handler = logging.StreamHandler(stream=sys.stdout)
        if prettyprinter:
            pretty_handler = logging.StreamHandler(stream=sys.stdout)
            pretty_handler.setLevel(level)
            pretty_handler.setFormatter(logging.Formatter('\n%(message)s'))
            # update pretty printer 
            if (prettyprinter.hasHandlers()):
                prettyprinter.handlers.clear()
            prettyprinter.addHandler(pretty_handler)

    handler.setLevel(level)
    if verbose:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s:\n\t%(message)s\n', datefmt='%H:%M:%S'))
    else:
        handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

# run shell commands from notebook
def subprocess_cmd(command, verbose=2):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout,stderr = process.communicate()
    if verbose==2:  
        print(command)
        print(stdout.decode("utf-8"))
        print(stderr.decode("utf-8"))
    elif verbose==1:
        if len(stderr.decode("utf-8"))>2:
            print(command)
            print('n',stderr.decode("utf-8"),'m')


def get_decay_rate_in_s(G):
    return 1.0/G*invGeV_to_s
def get_decay_rate_in_cm(G):
    return 1.0/G*invGeV_to_cm

# phase space function
def kallen(a,b,c):
    return (a-b-c)**2 - 4*b*c
def kallen_sqrt(a,b,c):
    return np.sqrt(kallen(a,b,c))

################################################
# New flags
THREEPLUSONE = 89
THREEPLUSTWO = 90
THREEPLUSTHREE = 91

MAJORANA = 'majorana'
DIRAC    = 'dirac'

#### Shorthands used in MATHEMATICA
MZBOSON = m_Z
MW = m_W
def Power(x,n):
    return x**n
def Sqrt(x):
    return sqrt(x)
Pi = np.pi