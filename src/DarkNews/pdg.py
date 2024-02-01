import numpy as np
import logging
logger = logging.getLogger('logger.' + __name__)

#####################################
# particle names and props
from particle import Particle
from particle import literals as lp

################################################
# auxiliary functions -- scikit-hep particles
def in_same_doublet(p1, p2):
    if p1.pdgid in [11, 13, 15]:
        return p2.pdgid - p1.pdgid == 1 or p2.pdgid - p1.pdgid == 0
    elif p1.pdgid in [12, 14, 16]:
        return p1.pdgid - p2.pdgid == 1 or p1.pdgid - p2.pdgid == 0
    else:
        return None


def in_e_doublet(p):
    return np.abs(p.pdgid) in [11, 12]


def in_mu_doublet(p):
    return np.abs(p.pdgid) in [13, 14]


def in_tau_doublet(p):
    return np.abs(p.pdgid) in [15, 16]


def get_doublet(p):
    if in_e_doublet(p):
        return 0
    elif in_mu_doublet(p):
        return 1
    elif in_tau_doublet(p):
        return 2
    else:
        logger.error(f"Could not find doublet of {p.name}.")
        return 0


def same_doublet(p1, p2):
    return get_doublet(p1) == get_doublet(p2)


def same_particle(p1, p2):
    return p1.pdgid == p2.pdgid


def is_particle(p):
    return p.pdgid > 0


def is_antiparticle(p):
    return p.pdgid < 0


#####
# generational indices
# 0 - e, 1 -mu, 2 - tau, 3 - N4, 4 - N5, ...
def get_HNL_index(particle):
    return int(particle.name.strip("nuN")) - 1


def get_lepton_index(particle):
    return get_doublet(particle)


# mediators
photon = Particle.from_pdgid(22)

# Leptons
electron = Particle.from_pdgid(11)
positron = Particle.from_pdgid(-11)

muon = Particle.from_pdgid(13)
antimuon = Particle.from_pdgid(-13)

tau = Particle.from_pdgid(15)
antitau = Particle.from_pdgid(-15)

nue = Particle.from_pdgid(12)
nuebar = Particle.from_pdgid(-12)

numu = Particle.from_pdgid(14)
numubar = Particle.from_pdgid(-14)

nutau = Particle.from_pdgid(16)
nutaubar = Particle.from_pdgid(-16)

# Baryons
proton = Particle.from_pdgid(2212)
neutron = Particle.from_pdgid(2112)

# Nuclei
Argon40 = Particle.from_pdgid(1000180400)
Carbon12 = Particle.from_pdgid(1000060120)


########################################################################################
# Define new particles
"""
PDG code convention for new particles

As advised by the PDG, we start the new particle system with 59. The full identifier is:
    
     PDGID  =  59(particle spin code: 0-scalar 1-fermion 2-vector)(generation number)

"""


def new_particle(name, pdgid, charge=0, mass=0, **kwargs):
    return Particle(pdg_name=name, pdgid=pdgid, three_charge=3*charge, mass=mass, **kwargs)
    """ 
        Particle class definition:
            https://github.com/scikit-hep/particle/blob/dd3c71e0b4319f729533ff0fc2e1e8cfa49684dd/src/particle/particle/particle.py#L91
    """


"""
PDG code convention for new particles

As advised by the PDG, we start the new system with 59. The full identifier is:
    
     PDGID  =  59(particle spin code: 0-scalar 1-fermion 2-vector)(generation number)

"""

# pseudoparticle denoting all light neutrinos states
nulight = new_particle(name="nu_light", pdgid=5910, latex_name="\nu_{\rm light}")

# light neutrino mass states
neutrino1 = new_particle(name="nu1", pdgid=5911, latex_name="\nu_1")
neutrino2 = new_particle(name="nu2", pdgid=5912, latex_name="\nu_2")
neutrino3 = new_particle(name="nu3", pdgid=5913, latex_name="\nu_3")

# heavy neutrinos
neutrino4 = new_particle(name="N4", pdgid=5914, latex_name="N_4")
neutrino5 = new_particle(name="N5", pdgid=5915, latex_name="N_5")
neutrino6 = new_particle(name="N6", pdgid=5916, latex_name="N_6")

# dark photon
zprime = new_particle(name="zprime", pdgid=5921, latex_name="Z^\prime")

# three kind of new scalar particles
hprime = new_particle(name="hprime", pdgid=5901, latex_name="h^\prime")
phi = new_particle(name="phi", pdgid=5902, latex_name="\varphi")
alp = new_particle(name="alp", pdgid=5903, latex_name="a")

########################################################################################
