""" Constant modules

Here are defined the most common constants used in DarkNews.

We use PDG2020 values for constants and SM masses.

Some low-level auxiliary functions are defined.

"""

import numpy as np
from numpy import sqrt
from scipy import interpolate

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

################################################
# constants of light cm/s
c_LIGHT = 29_979_245_800
hb = 6.582119569e-25  # hbar in Gev s

m_proton_in_kg = 1.6726219236951e-27
m_proton_in_g = m_proton_in_kg * 1e3
m_proton_in_t = m_proton_in_kg * 1e-3

################################################
# constants for convertion
MeV_to_GeV = 1e-3

invm2_to_incm2 = 1e-4
fb_to_cm2 = 1e-39
NAvo = 6.02214076 * 1e23
rad_to_deg = 180.0 / np.pi
deg_to_rad = 1 / rad_to_deg

s_in_year = 60 * 60 * 24 * 365.25

invGeV_to_cm = hb * c_LIGHT  # hbar c = 197.3269804e-16 GeV.cm
invGeV2_to_cm2 = invGeV_to_cm**2

fm_to_GeV = 1 / invGeV_to_cm * 1e-15 * 1e2
invGeV_to_s = invGeV_to_cm / c_LIGHT

cm2_to_attobarn = 1e42
attobarn_to_cm2 = 1e-42

invGeV2_to_attobarn = invGeV2_to_cm2 * cm2_to_attobarn

# cosmo definitions
GeV2_to_cm3s = invGeV2_to_cm2 * c_LIGHT
invcm3_to_eV3 = invGeV2_to_cm2 ** (3 / 2) * 1e27
Mplanck = 1.22e19  # GeV

g_to_eV = 5.6095886031e32  # eV
kg_to_eV = 5.6095886031e35  # eV
t_to_eV = 5.6095886031e38  # eV

g_to_GeV = g_to_eV * 1e-9  # GeV
kg_to_GeV = kg_to_eV * 1e-9  # GeV
t_to_GeV = t_to_eV * 1e-9  # GeV

#####################################
# indices of neutral leptons
ind_e = 0
ind_mu = 1
ind_tau = 2

# list of active neutrino indices
inds_active = range(ind_tau + 1)

N4 = "N4"
N5 = "N5"
N6 = "N6"

################################################
# Masses

# quarks
m_u = 2.16e-3  # GeV
m_d = 4.67e-3  # GeV
m_s = 93e-3  # GeV
m_c = 1.27  # GeV
m_b = 4.18  # GeV
m_t = 172.76  # GeV

# nucleons
m_proton = 0.93827208816  # GeV
m_neutron = 0.93956542052  # GeV
m_avg = (m_proton + m_neutron) / 2.0  # GeV

# leptons
m_W = 80.37912  # GeV
m_Z = 91.187621  # GeV

m_e = 0.5109989500015e-3  # GeV
m_mu = 0.1056583755  # GeV
m_tau = 1.77686  # GeV

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
alphaQED = 1.0 / 137.03599908421  # Fine structure constant at q2 -> 0
eQED = np.sqrt((4 * np.pi) * alphaQED)

# get running alphaQED
Q, inv_alphaQED = np.genfromtxt(files("DarkNews.include.aux_data").joinpath("alpha_QED_running_posQ2.dat").open(), unpack=True)
runningAlphaQED = interpolate.interp1d(Q, 1.0 / inv_alphaQED)

################################################
# Weak sector
Gf = 1.16637876e-5  # Fermi constant (GeV^-2)
gweak = np.sqrt(Gf * m_W**2 * 8 / np.sqrt(2))
s2w = 0.22343  # On-shell
sw = np.sqrt(s2w)
cw = np.sqrt(1.0 - s2w)

################################################
# Higgs -- https://pdg.lbl.gov/2019/reviews/rpp2018-rev-higgs-boson.pdf
vev_EW = 1 / np.sqrt(np.sqrt(2) * Gf)
m_H = 125.10
lamb_quartic = (m_H / vev_EW) ** 2 / 2
m_h_potential = -lamb_quartic * vev_EW**2

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

Fneutral_pion = fneutral_pion / np.sqrt(2.0)
Fneutral_kaon = fneutral_kaon / np.sqrt(2.0)
Fneutral_B = fneutral_B / np.sqrt(2.0)
Fcharged_B = fcharged_B / np.sqrt(2.0)
Fneutral_Bs = fneutral_Bs / np.sqrt(2.0)
Fneutral_eta = fneutral_eta / np.sqrt(2.0)


################################################
# CKM elements
# PDG2019
lamCKM = 0.22453
ACKM = 0.836
rhoBARCKM = 0.122
etaBARCKM = 0.355
rhoCKM = rhoBARCKM / (1 - lamCKM * lamCKM / 2.0)
etaCKM = etaBARCKM / (1 - lamCKM * lamCKM / 2.0)

s12 = lamCKM
s23 = ACKM * lamCKM**2
s13e = (
    ACKM
    * lamCKM**3
    * (rhoCKM + 1j * etaCKM)
    * np.sqrt(1.0 - ACKM**2 * lamCKM**4)
    / (np.sqrt(1.0 - lamCKM**2) * (1.0 - ACKM**2 * lamCKM**4 * (rhoCKM + 1j * etaCKM)))
)
c12 = np.sqrt(1 - s12**2)
c23 = np.sqrt(1 - s23**2)
c13 = np.sqrt(1 - abs(s13e) ** 2)
Vud = c12 * c13
Vus = s12 * c13
Vub = np.conj(s13e)
Vcd = -s12 * c23 - c12 * s23 * s13e
Vcs = c12 * c23 - s12 * s23 * s13e
Vcb = s23 * c13
Vtd = s12 * s23 - c12 * c23 * s13e
Vts = -c12 * s23 - s12 * c23 * s13e
Vtb = c23 * c13
UCKM = np.matrix([[Vud, Vus, Vub], [Vcd, Vcs, Vcb], [Vtd, Vts, Vtb]], dtype=complex)

################################################
# PMNS parameters
# NuFit Oct 2021 -- http://www.nu-fit.org/?q=node/238#label85

s12 = np.sqrt(0.304)
s23 = np.sqrt(0.450)
s13 = np.sqrt(0.02246)
c12 = np.sqrt(1 - s12**2)
c23 = np.sqrt(1 - s23**2)
c13 = np.sqrt(1 - s13**2)
delta = 230 * deg_to_rad

Ue1 = c12 * c13
Ue2 = s12 * c13
Ue3 = s13 * np.exp(-delta * 1j)
Umu1 = -s12 * c23 - c12 * s23 * s13 * np.exp(delta * 1j)
Umu2 = c12 * c23 - s12 * s23 * s13 * np.exp(delta * 1j)
Umu3 = s23 * c13
Utau1 = s12 * s23 - c12 * c23 * s13 * np.exp(delta * 1j)
Utau2 = -c12 * s23 - s12 * c23 * s13 * np.exp(delta * 1j)
Utau3 = c23 * c13
UPMNS = np.matrix([[Ue1, Ue2, Ue3], [Umu1, Umu2, Umu3], [Utau1, Utau2, Utau3]], dtype=complex)


################################################
# low-level auxiliary functions
################################################
def get_decay_rate_in_s(G):
    return 1.0 / G * invGeV_to_s


def get_decay_rate_in_cm(G):
    return 1.0 / G * invGeV_to_cm


# phase space function
def kallen(a, b, c):
    return (a - b - c) ** 2 - 4 * b * c


def kallen_sqrt(a, b, c):
    return np.sqrt(kallen(a, b, c))


def rng_interval(size, a, b, rng):
    return rng(size) * (a - b) + b


# aliases used in MATHEMATICA copy-paste
MZBOSON = m_Z
MW = m_W

# from CForm to pythonic syntax
Power = lambda x, n: x**n
Sqrt = lambda x: sqrt(x)
Pi = np.pi
