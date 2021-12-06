import numpy as np
import numpy.ma as ma
from scipy import interpolate
import os
import itertools
from DarkNews import logger
from DarkNews import local_dir

from .const_dics import fourier_bessel_dic
from .const import *

'''
Here we define nuclear form factors following:
http://discovery.phys.virginia.edu/research/groups/ncd/index.html

When available, we use data from Nuclear Data Tables (74, 87, and 95), stored in "aux_data/mass20.txt":
    "The Ame2020 atomic mass evaluation (I)"   by W.J.Huang, M.Wang, F.G.Kondev, G.Audi and S.Naimi
           Chinese Physics C45, 030002, March 2021.
    "The Ame2020 atomic mass evaluation (II)"  by M.Wang, W.J.Huang, F.G.Kondev, G.Audi and S.Naimi
           Chinese Physics C45, 030003, March 2021.

Element properties are stored in elements_dic.To access individual elements we use the format:
    
    key = 'name+A', e.g. key = 'Pb208' or 'C12'.
'''

def assign_form_factors(target):

    # Nucleus
    if target.is_nucleus:
        try:
            a=fourier_bessel_dic[target.name.lower()] ## stored with lower case formatting
            fcoh = lambda x: nuclear_F1_fourier_bessel_EM(x,a)
        except KeyError:
            logger.warning(f'Warning: nuclear density for {target.name} not tabulated in Nuclear Data Table. Using symmetrized Fermi form factor instead.')
            fcoh = lambda x: nuclear_F1_Fsym_EM(x, target.A)
        except:
            logger.warning(f'Warning: could not compute the nuclear form factor for {target.name}. Taking it to be vanishing.')
            fcoh = lambda x: 0

        ### FIX ME -- No nuclear magnetic moments so far
        target.F1_EM = fcoh # Dirac FF
        target.F2_EM = lambda x: 0.0 # Pauli FF
        
        ### FIX ME -- need to find the correct NC form factor
        target.F1_NC = lambda x: nuclear_F1_FHelmz_NC(x, target.A) # Dirac FF
        target.F2_NC = lambda x: 0.0 # Pauli FF
        target.F3_NC = lambda x: 0.0 # Axial FF
    
    # Nucleons
    elif target.is_nucleon:

        target.F1_EM = lambda x: nucleon_F1_EM(x, target.tau3) # Dirac FF
        target.F2_EM = lambda x: nucleon_F2_EM(x, target.tau3) # Pauli FF
        
        target.F1_NC = lambda x: nucleon_F1_NC(x, target.tau3) # Dirac FF
        target.F2_NC = lambda x: nucleon_F2_NC(x, target.tau3) # Pauli FF
        target.F3_NC = lambda x: nucleon_F3_NC(x, target.tau3) # Axial FF

    else:
        logger.error(f"Could not find hadronic target {target.name}.")
        exit(0)



# covering the zero case 0 
def sph_bessel_0(x):
    x = np.array(x)
    tolerance=(x>0.0)
    j0 = ma.masked_array(data=np.ma.sin(x)/x,
                    mask=~tolerance,
                    fill_value=1.0)
    return j0.filled()

# all units in fm
def fourier_bessel_form_factor_terms(q,R,n):
    x = q*R
    return sph_bessel_0(x)*R**3/((n*np.pi)**2 - x**2)*(-1)**n

# all units in fm
def fourier_bessel_integral_terms(R,n):
    return R**3*(-1)**n/np.pi**2/n**2

# Q2 in GeV^2 and other units in fm
def nuclear_F1_fourier_bessel_EM(Q2, array_coeff):
    q=np.sqrt(Q2)*fm_to_GeV
    R = array_coeff[-1]
    ai = np.array(array_coeff[:-1])
    nonzero_terms = np.nonzero(ai)[0]
    # compute expansion for non-zero terms
    expansion=np.zeros((np.size(q)))
    for i in nonzero_terms:
        n=i+1
        expansion += fourier_bessel_form_factor_terms(q,R,n)*ai[i]
    Q_total = np.sum(fourier_bessel_integral_terms(R,nonzero_terms+1)*ai[nonzero_terms])
    return expansion/Q_total

################################################
# NUCLEAR DATA -- AME 2020
# All units in GeV

# approximate formula in D. Lunney, J.M. Pearson and C. Thibault, Rev. Mod. Phys.75, 1021 (2003)
# eV
def electron_binding_energy(Z):
    return (14.4381*Z**2.39 + 1.55468e-6*Z**5.35)

# Nested dic containing all elements
elements_dic = {}
hydrogen_Eb = 13.5981e-9 # GeV
atomic_unit = 0.9314941024228 # mass of Carbon12 / 12
mass_file = os.path.join(local_dir, 'aux_data/mass20.txt')
with open(mass_file, 'r') as ame:
    # Read lines in file starting at line 40
    for line in itertools.islice(ame, 36, None):
        
        Z = int(line[12:15])
        
        if Z < 93: ## no support for heavier elements due to the 
            name = '{}{}'.format(line[20:22].strip(), int(line[16:19]))
            elements_dic[name] = {}
            
            elements_dic[name]['name'] = name
            elements_dic[name]['Z'] = Z
            elements_dic[name]['N'] = int(line[6:10])
            elements_dic[name]['A'] = int(line[16:20])

            elements_dic[name]['atomic_Eb'] = float(line[56:61] + '.' + line[62:67])*1e-6
            
            elements_dic[name]['electronic_Eb'] = electron_binding_energy(Z)*1e-9

            elements_dic[name]['nuclear_Eb'] = elements_dic[name]['atomic_Eb'] + Z*hydrogen_Eb - elements_dic[name]['electronic_Eb']

            # micro-mu = 1e-6 mu in GeV*u
            elements_dic[name]['atomic_mass'] = (float(line[106:109]) + 1e-6*float(line[110:116] + '.' + line[117:124]))*atomic_unit
            elements_dic[name]['excess_mass'] = float(line[30:35] + '.' + line[36:42])*1e-6

            # nuclear mass correct for electron mass but neglecting electron binding energy (Eb_e << MeV)
            elements_dic[name]['nuclear_mass'] =  (elements_dic[name]['atomic_mass'] 
                                                               - Z*m_e + elements_dic[name]['electronic_Eb'])

            decay = line[82:88] + '.' + line[89:94]
            if '*' in decay:
                decay = np.nan
            elements_dic[name]['beta_decay_energy'] = float(decay)*1e-6




################################################
# NUCLEAR AND NUCLEON FORM FACTORS
MAG_N = -1.913
MAG_P = 2.792
## dipole parametrization
def D(Q2):
    MV = 0.843 # GeV
    return 1.0/((1+Q2/MV**2)**2)

## nucleon
def nucleon_F1_EM(Q2, tau3):
    # pick nucleon mag moment
    MAG = ( MAG_P+MAG_N + tau3*(MAG_P-MAG_N))/2.0
    tau = -Q2/4.0/m_proton**2
    return (D(Q2) - tau*MAG*D(Q2))/(1-tau)

def nucleon_F2_EM(Q2, tau3):
    # pick nucleon mag moment
    MAG = ( MAG_P+MAG_N + tau3*(MAG_P-MAG_N))/2.0
    tau = -Q2/4.0/m_proton**2
    return (MAG*D(Q2) - D(Q2))/(1-tau)



def nucleon_F1_NC(Q2, tau3):
    tau = -Q2/4.0/m_proton**2
    f = (0.5 - s2w)*(tau3)*(1-tau*(1+MAG_P-MAG_N))/(1-tau) - s2w*(1-tau*(1+MAG_P+MAG_N))/(1-tau)  
    return f*D(Q2)

def nucleon_F2_NC(Q2, tau3):
    tau = -Q2/4.0/m_proton**2
    f = (0.5 - s2w)*(tau3)*(MAG_P-MAG_N)/(1-tau) - s2w*(MAG_P+MAG_N)/(1-tau)  
    return f*D(Q2)

def nucleon_F3_NC(Q2, tau3):
    MA = 1.02 # GeV
    gA = 1.26
    f = gA*tau3/2.0/(1+Q2/MA**2)**2
    return f

## symmetrized fermi nuclear
def nuclear_F1_Fsym_EM(Q2,A):
    Q=np.sqrt(Q2)
    a = 0.523*fm_to_GeV # GeV^-1
    r0 = 1.03*(A**(1.0/3.0))*fm_to_GeV # GeV^-1
    tolerance=(Q>0.0)
    clean_FF = ma.masked_array(data=3.0*np.pi*a/(r0**2 + np.pi**2 * a**2) * (np.pi*a *(1.0/np.tanh(np.pi*a*Q))*np.sin(Q*r0) - r0*np.cos(Q*r0))/(Q*r0*np.sinh(np.pi*Q*a)),
                    mask=~tolerance,
                    fill_value=1.0)
    return clean_FF.filled()
def j1(z):
    z = np.array(z)
    tolerance=(z>0.0)
    clean_j1 = ma.masked_array(data=np.sin(z)/z/z - np.cos(z)/z,
                    mask=~tolerance,
                    fill_value=0.0)
    return clean_j1.filled()

def nuclear_F1_FHelmz_NC(Q2,A):
    Q=np.sqrt(Q2)
    a = 0.523*fm_to_GeV # GeV^-1
    s = 0.9*fm_to_GeV # fm to GeV^-1
    R = 3.9*fm_to_GeV*(A/40.0)**(1.0/3.0) # fm to GeV^-1
    return (3*np.abs(j1(Q*R)/Q/R))*np.exp(-Q*Q*s*s/2)


        