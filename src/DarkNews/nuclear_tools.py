import numpy as np
import numpy.ma as ma
from itertools import islice
from functools import partial

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from particle import literals as lp

import logging

logger = logging.getLogger("logger." + __name__)

from DarkNews.const_dics import fourier_bessel_dic
from DarkNews import const


# to replace certain lambda functions
def zero_func(x):
    return 0.0


class NuclearTarget:
    def __init__(self, name):
        """
        Main DarkNews class for the nucleus to be used in a neutrino scattering event.

        It contains target properties like number of protons, neutrons, informations on the mass, as well
        as all the Nuclear Data Table information on:
            nuclear_Eb,
            atomic_Eb,
            atomic_mass,
            excess_mass,
            nuclear_mass,
            beta_decay_energy.

        It also provides the nuclear form factors as functions to

        Its daughter class, BoundNucleon(), handles the nucleons inside the nucleus.

        Args:
            name (str): name of the target. Can be either "electron" or the name of the element
            following the Nuclear Data Table format (e.g. C12, Pb208).

        """

        self.name = name

        #####################################
        # free electron
        if name == "electron":
            self.is_hadron = False
            self.is_nucleus = False
            self.is_proton = False
            self.is_neutron = False
            self.is_free_nucleon = False

            self.mass = const.m_e
            self.charge = 1
            self.Z = 0
            self.N = 0
            self.A = 0
            self.pdgid = 11
        #####################################
        # hadronic *nuclear* target
        else:

            # Using the global dictionary of elements defined in nuclear_tools
            # Set all items as attributes of the class
            for k, v in elements_dic[name].items():
                setattr(self, k, v)
            self.mass = self.nuclear_mass
            self.charge = self.Z

            self.is_hadron = True
            self.is_nucleus = self.A > 1
            self.is_proton = self.Z == 1 and self.A == 1
            self.is_neutron = self.N == 1 and self.A == 1

            self.is_nucleon = self.is_neutron or self.is_proton
            self.is_free_nucleon = self.A == 1
            self.is_bound_nucleon = False

            if self.is_nucleus:
                # no hyperons and always ground state
                self.pdgid = int(f"100{self.Z:03d}{self.A:03d}0")
            elif self.is_neutron:
                self.pdgid = lp.neutron.pdgid
            elif self.is_proton:
                self.pdgid = lp.proton.pdgid
            else:
                logger.error(f"Error. Could not find the PDG ID of {self.name}.")
                raise ValueError

            if self.is_neutron and self.is_free_nucleon:
                logger.error(f"Error. Target {self.name} is a free neutron.")
                raise ValueError

            self.tau3 = self.Z * 2 - 1  # isospin +1 proton / -1 neutron

            assign_form_factors(self)

    # hadronic *constituent* target
    def get_constituent_nucleon(self, name):
        return self.BoundNucleon(self, name)

    class BoundNucleon:
        """for scattering on bound nucleon in the nuclear target

        Inner Class

        Args:
            nucleus: nucleus of which this particle is bound into (always the outer class)
            name: 'proton' or 'neutron'
        """

        def __init__(self, nucleus, name):

            self.nucleus = nucleus
            self.A = int(name == "proton" or name == "neutron")
            self.Z = int(name == "proton")
            self.N = int(name == "neutron")
            self.charge = self.Z
            self.mass = const.m_proton
            self.name = f"{name}_in_{nucleus.name}"

            self.is_hadron = True
            self.is_nucleus = False
            self.is_neutron = self.N == 1 and self.A == 1
            self.is_proton = self.Z == 1 and self.A == 1

            self.is_nucleon = self.is_neutron or self.is_proton
            self.is_false_nucleon = False
            self.is_bound_nucleon = True

            if self.is_neutron:
                self.pdgid = lp.neutron.pdgid
            elif self.is_proton:
                self.pdgid = lp.proton.pdgid
            else:
                logger.error(f"Error. Could not find the PDG ID of {self.name}.")
                raise ValueError

            self.tau3 = self.Z * 2 - 1  # isospin +1 proton / -1 neutron

            assign_form_factors(self)


def assign_form_factors(target):
    """
    Here we define nuclear form factors following:
    http://discovery.phys.virginia.edu/research/groups/ncd/index.html

    When available, we use data from Nuclear Data Tables (74, 87, and 95), stored in "include/aux_data/mass20.txt":
        "The Ame2020 atomic mass evaluation (I)"   by W.J.Huang, M.Wang, F.G.Kondev, G.Audi and S.Naimi
            Chinese Physics C45, 030002, March 2021.
        "The Ame2020 atomic mass evaluation (II)"  by M.Wang, W.J.Huang, F.G.Kondev, G.Audi and S.Naimi
            Chinese Physics C45, 030003, March 2021.

    Element properties are stored in elements_dic.To access individual elements we use the format:

        key = 'name+A', e.g. key = 'Pb208' or 'C12'.

    All units in GeV, except otherwise specified.

    Args:
        target (DarkNews.nuclear_tools.Target): instance of main DarkNews target object.
    """

    # Nucleus
    if target.is_nucleus:
        try:
            a = fourier_bessel_dic[target.name.lower()]  ## stored with lower case formatting
            fcoh = partial(nuclear_F1_fourier_bessel_EM, array_coeff=a)
        except KeyError:
            logger.warning(f"Warning: nuclear density for {target.name} not tabulated in Nuclear Data Table. Using symmetrized Fermi form factor instead.")
            fcoh = partial(nuclear_F1_Fsym_EM, A=target.A)
        except:
            logger.warning(f"Warning: could not compute the nuclear form factor for {target.name}. Taking it to be vanishing.")
            fcoh = zero_func

        ### FIX ME -- No nuclear magnetic moments so far
        target.F1_EM = fcoh  # Dirac FF
        target.F2_EM = zero_func  # Pauli FF

        ### FIX ME -- need to find the correct NC form factor
        # target.F1_NC = partial(nuclear_F1_FHelmz_NC, target.A) # Dirac FF
        target.F1_NC = fcoh  # Dirac FF
        target.F2_NC = zero_func  # Pauli FF
        target.F3_NC = zero_func  # Axial FF

    # Nucleons
    elif target.is_nucleon:

        target.F1_EM = partial(nucleon_F1_EM, tau3=target.tau3)  # Dirac FF
        target.F2_EM = partial(nucleon_F2_EM, tau3=target.tau3)  # Pauli FF

        target.F1_NC = partial(nucleon_F1_NC, tau3=target.tau3)  # Dirac FF
        target.F2_NC = partial(nucleon_F2_NC, tau3=target.tau3)  # Pauli FF
        target.F3_NC = partial(nucleon_F3_NC, tau3=target.tau3)  # Axial FF

    else:
        logger.error(f"Could not find hadronic target {target.name}.")
        exit(0)


# covering the zero case 0
def sph_bessel_0(x):
    x = np.array(x)
    tolerance = x > 0.0
    j0 = ma.masked_array(data=np.ma.sin(x) / x, mask=~tolerance, fill_value=1.0)
    return j0.filled()


# all units in fm
def fourier_bessel_form_factor_terms(q, R, n):
    x = q * R
    return sph_bessel_0(x) * R**3 / ((n * np.pi) ** 2 - x**2) * (-1) ** n


# all units in fm
def fourier_bessel_integral_terms(R, n):
    return R**3 * (-1) ** n / np.pi**2 / n**2


# Q2 in GeV^2 and other units in fm
def nuclear_F1_fourier_bessel_EM(Q2, array_coeff):
    q = np.sqrt(Q2) * const.fm_to_GeV
    R = array_coeff[-1]
    ai = np.array(array_coeff[:-1])
    nonzero_terms = np.nonzero(ai)[0]
    # compute expansion for non-zero terms
    expansion = np.zeros((np.size(q)))
    for i in nonzero_terms:
        n = i + 1
        expansion += fourier_bessel_form_factor_terms(q, R, n) * ai[i]
    Q_total = np.sum(fourier_bessel_integral_terms(R, nonzero_terms + 1) * ai[nonzero_terms])
    return np.abs(expansion / Q_total)


#####################################
# Nested dic containing all elements
# approximate formula in D. Lunney, J.M. Pearson and C. Thibault, Rev. Mod. Phys.75, 1021 (2003)
def electron_binding_energy(Z):
    return (14.4381 * Z**2.39 + 1.55468e-6 * Z**5.35) * 1e-9  # GeV


elements_dic = {}
hydrogen_Eb = 13.5981e-9  # GeV
atomic_unit = 0.9314941024228  # mass of Carbon12 in GeV / 12

with files("DarkNews.include.aux_data").joinpath("mass20_1.txt").open() as ame:
    # Read lines in file starting at line 36
    for line in islice(ame, 36, None):

        Z = int(line[12:15])

        if Z < 93:  # no support for heavier elements
            name = "{}{}".format(line[20:22].strip(), int(line[16:19]))
            elements_dic[name] = {}

            elements_dic[name]["name"] = name
            elements_dic[name]["Z"] = Z
            elements_dic[name]["N"] = int(line[6:10])
            elements_dic[name]["A"] = int(line[16:20])

            # elements_dic[name]['atomic_Eb'] = float(line[56:61] + '.' + line[62:67])*1e-6*elements_dic[name]['A']
            elements_dic[name]["atomic_Eb"] = electron_binding_energy(Z)

            elements_dic[name]["nuclear_Eb"] = float(line[56:61] + "." + line[62:67]) * 1e-6 * elements_dic[name]["A"]
            # elements_dic[name]['atomic_Eb'] + Z*hydrogen_Eb - elements_dic[name]['electronic_Eb']

            # micro-mu = 1e-6 mu in GeV*u
            elements_dic[name]["atomic_mass"] = (float(line[106:109]) + 1e-6 * float(line[110:116] + "." + line[117:124])) * atomic_unit
            elements_dic[name]["excess_mass"] = float(line[30:35] + "." + line[36:42]) * 1e-6

            # nuclear mass corrected for electron mass and electron binding energy (note that Eb_e << MeV)
            elements_dic[name]["nuclear_mass"] = elements_dic[name]["atomic_mass"] - Z * const.m_e + elements_dic[name]["atomic_Eb"]

            decay = line[82:88] + "." + line[89:94]
            if "*" in decay:
                decay = np.nan
            elements_dic[name]["beta_decay_energy"] = float(decay) * 1e-6


################################################
# NUCLEAR AND NUCLEON FORM FACTORS
MAG_N = -1.913
MAG_P = 2.792


## dipole parametrization
def D(Q2):
    MV = 0.843  # GeV
    return 1.0 / ((1 + Q2 / MV**2) ** 2)


## nucleon
def nucleon_F1_EM(Q2, tau3):
    # pick nucleon mag moment
    MAG = (MAG_P + MAG_N + tau3 * (MAG_P - MAG_N)) / 2.0
    tau = -Q2 / 4.0 / const.m_proton**2
    return (D(Q2) - tau * MAG * D(Q2)) / (1 - tau)


def nucleon_F2_EM(Q2, tau3):
    # pick nucleon mag moment
    MAG = (MAG_P + MAG_N + tau3 * (MAG_P - MAG_N)) / 2.0
    tau = -Q2 / 4.0 / const.m_proton**2
    return (MAG * D(Q2) - D(Q2)) / (1 - tau)


def nucleon_F1_NC(Q2, tau3):
    tau = -Q2 / 4.0 / const.m_proton**2
    f = (0.5 - const.s2w) * (tau3) * (1 - tau * (1 + MAG_P - MAG_N)) / (1 - tau) - const.s2w * (1 - tau * (1 + MAG_P + MAG_N)) / (1 - tau)
    return f * D(Q2)


def nucleon_F2_NC(Q2, tau3):
    tau = -Q2 / 4.0 / const.m_proton**2
    f = (0.5 - const.s2w) * (tau3) * (MAG_P - MAG_N) / (1 - tau) - const.s2w * (MAG_P + MAG_N) / (1 - tau)
    return f * D(Q2)


def nucleon_F3_NC(Q2, tau3):
    MA = 1.02  # GeV
    gA = 1.26
    f = gA * tau3 / 2.0 / (1 + Q2 / MA**2) ** 2
    return f


# symmetrized fermi nuclear
def f(Q, a, r0):
    return (
        3.0
        * np.pi
        * a
        / (r0**2 + np.pi**2 * a**2)
        * (np.pi * a * (1.0 / np.tanh(np.pi * a * Q)) * np.sin(Q * r0) - r0 * np.cos(Q * r0))
        / (Q * r0 * np.sinh(np.pi * Q * a))
    )


def nuclear_F1_Fsym_EM(Q2, A):
    Q = np.sqrt(Q2)
    a = 0.523 * const.fm_to_GeV  # GeV^-1
    r0 = 1.03 * (A ** (1.0 / 3.0)) * const.fm_to_GeV  # GeV^-1
    # tolerance = Q < 5
    # clean_FF = ma.masked_array(
    #     data=3.0 * np.pi * a / (r0 ** 2 + np.pi ** 2 * a ** 2) *
    #         (np.pi * a * (1.0 / np.tanh(np.pi * a * Q)) * np.sin(Q * r0) - r0 * np.cos(Q * r0)) /
    #         (Q * r0 * np.sinh(np.pi * Q * a)),
    #     mask=~tolerance,
    #     fill_value=0.0,
    # )
    # return clean_FF.filled()
    return np.piecewise(Q, [Q < 5, Q >= 5], [partial(f, a=a, r0=r0), zero_func])


def j1(z):
    z = np.array(z)
    tolerance = z > 0.0
    clean_j1 = ma.masked_array(data=np.sin(z) / z / z - np.cos(z) / z, mask=~tolerance, fill_value=0.0)
    return clean_j1.filled()


def nuclear_F1_FHelmz_NC(Q2, A):
    Q = np.sqrt(Q2)
    a = 0.523 * const.fm_to_GeV  # GeV^-1
    s = 0.9 * const.fm_to_GeV  # fm to GeV^-1
    R = 3.9 * const.fm_to_GeV * (A / 40.0) ** (1.0 / 3.0)  # fm to GeV^-1
    return (3 * np.abs(j1(Q * R) / Q / R)) * np.exp(-Q * Q * s * s / 2)
