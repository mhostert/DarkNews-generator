import numpy as np

import logging

logger = logging.getLogger("logger." + __name__)

from DarkNews import const
from DarkNews.const import Pi, MZBOSON, MW, gweak, eQED


def tau_GeV_to_s(decay_rate):
    try:
        return 1.0 / decay_rate / 1.52 / 1e24
    except ZeroDivisionError("Lifetime of stable particle is infinite"):
        return np.inf


def L_GeV_to_cm(decay_rate):
    return tau_GeV_to_s(decay_rate) * const.c_LIGHT


# #################################################
# Special functions for phase space integrals
# fmt: off
# def I1_2body(x,y):
#     return ((1+x-y)*(1+x) - 4*x)*const.kallen_sqrt(1.0,x,y)

# def F_3nu_decay(a,b,c):
#     # all effectively massless
#     if a+b+c < 1e-6:
#         return 1.0
#     else:

#         def DGammaDuDt(sab,sac,a,b,c):
#             rtot = 1 + a + b + c
#             sbc = 1 - sab - sac + a**2 + b**2 + c**2
#             I =  sab*(rtot - sab) + sac*(rtot - sac) + 2*(a+b*c)*(sbc - (a+b*c)) - (1+a)**2*(b+c)**2
#             return 6*I
        
#         # check
#         sab_min = lambda sac: a**2 + b**2 - ((-1 + sac + b**2)*(a**2 + sac - c**2))/(2.*sac) - (np.sqrt((-1 + sac)**2 - 2*(1 + sac)*c**2 + b**4)*np.sqrt(a**4 + (sac - c**2)**2 - 2*(a**2)*(sac + c**2)))/(2.*sac)
#         sab_max = lambda sac: a**2 + b**2 - ((-1 + sac + b**2)*(a**2 + sac - c**2))/(2.*sac) + (np.sqrt((-1 + sac)**2 - 2*(1 + sac)*c**2 + b**4)*np.sqrt(a**4 + (sac - c**2)**2 - 2*(a**2)*(sac + c**2)))/(2.*sac)
        
#         integral, error = dblquad(  DGammaDuDt, (a+c)**2, (1-b)**2,  sab_min, sab_max, args=(a,b,c))
#         return integral

# def G_3nu_decay(a,b,c):
#     # all effectively massless
#     if a+b+c < 1e-6:
#         return 1.0

#     else:

#         def DGammaDuDt(sab,sac,a,b,c):
#             rtot = 1 + a + b + c
#             sbc = 1 - sab - sac + a**2 + b**2 + c**2
#             I = sab*(rtot+(c+a*b)-sab)+sbc*(a+b*c)(b+a*c)(sac-(b+a*c)) -(1+b)(1+c)(a+b)(a+c)
#             return 12*I
        
#         # check
#         sab_min = lambda sac: a**2 + b**2 - ((-1 + sac + b**2)*(a**2 + sac - c**2))/(2.*sac) - (np.sqrt((-1 + sac)**2 - 2*(1 + sac)*c**2 + b**4)*np.sqrt(a**4 + (sac - c**2)**2 - 2*(a**2)*(sac + c**2)))/(2.*sac)
#         sab_max = lambda sac: a**2 + b**2 - ((-1 + sac + b**2)*(a**2 + sac - c**2))/(2.*sac) + (np.sqrt((-1 + sac)**2 - 2*(1 + sac)*c**2 + b**4)*np.sqrt(a**4 + (sac - c**2)**2 - 2*(a**2)*(sac + c**2)))/(2.*sac)
        
#         integral, error = dblquad(  DGammaDuDt, (a+c)**2, (1-b)**2, sab_min, sab_max, args=(a,b,c))
#         return integral



# ############################################################################
'''
    Scalar and vector boson decay rates -- always summed over final state hels
'''

# S -> ell ell
def gamma_S_to_ell_ell(vertex, mS, m_ell):
    r = m_ell/mS
    gamma = np.abs(vertex)**2*mS/(8*np.pi) * np.sqrt(1-4*r**2)**(3./2.)
    return gamma

# V -> ell ell
def gamma_V_to_ell_ell(vertex, mV, m_ell):
    r = m_ell/mV
    gamma = np.abs(vertex)**2*mV/(12*np.pi) * np.sqrt(1-4*r**2)*(1 + 2*r**2)
    return gamma



# # S -> Ni Nj -- assuming ordering of i and j matters (S -> Ni Nj) != ( S -> Ni Nj)
# # includes symmetry factor 
# def gamma_S_to_Ni_Nj(vertex, mS, mi, mj, HNLtype = 'majorana'):
    
#     r1 = mi/mS
#     r2 = mj/mS

#     gamma = mS/(4*np.pi)
#     gamma *= const.kallen_sqrt(1, r1**2, r2**2)

#     if HNLtype == 'majorana':
#         # Majorana -- dependence real and imag part of vertex 
#         gamma *= r1*r2*( np.conjugate(vertex)**2 + vertex**2) + np.abs(vertex)**2*(1 - r1**2 - r2**2)
    
#     elif HNLtype == 'dirac':
#         # Dirac -- dependence only on abs value of vertex
#         gamma *=  np.abs(vertex)**2*(1 - r1**2 - r2**2)
#     else: 
#         logger.error(f"HNL type {HNLtype} not recognized.")
#         return 0.0 

#     return gamma

# # V -> Ni Nj -- assuming ordering of i and j matters (Z' -> Ni Nj) != ( Z' -> Ni Nj)
# # includes symmetry factor 
# def gamma_V_to_Ni_Nj(vertex, mV, mi, mj, HNLtype = 'majorana'):
    
#     r1 = mi/mV
#     r2 = mj/mV  

#     gamma = mV/(48*np.pi)
#     gamma *= const.kallen_sqrt(1, r1**2, r2**2)

#     if HNLtype == 'majorana':
#         # Majorana -- dependence real and imag part of vertex 
#         gamma *= 3*r1*r2*( np.conjugate(vertex)**2 + vertex**2) + np.abs(vertex)**2*(2 - r1**4 - (1 - 2*r2**2)*r1**2 - r2**4 - r2**2)
    
#     elif HNLtype == 'dirac':
#         # Dirac -- dependence only on abs value of vertex
#         gamma *=  np.abs(vertex)**2*(2 - r1**4 - (1 - 2*r2**2)*r1**2 - r2**4 - r2**2)
#     else: 
#         logger.error(f"HNL type {HNLtype} not recognized.")
#         return 0.0 

#     return gamma


# ###########################################
'''
Heavy neutral lepton decay rates -- polarized (h=+1 or -1). 
Averaged rates are found by summing (gamma_h=1 + gamma_h=-1)/2
'''

# d(gamma)/d(cos(theta))   Ni(k) -> Nj(k1) S(k2)
def diff_gamma_Ni_to_Nj_S(cost, vertex_ij, mi, mj, mS, HNLtype = 'majorana', h = -1):
    
    r1 = np.full_like(cost, mj/mi)
    r2 = np.full_like(cost, mS/mi)
    
    diff_gamma = mi / 32 / np.pi
    diff_gamma *= const.kallen_sqrt(1, r1**2, r2**2)

    if HNLtype == 'majorana':
        # Majorana -- independent of cost 
        diff_gamma *= np.conjugate(vertex_ij)**2 * r1 - vertex_ij**2 *r1 + np.abs(vertex_ij)**2*(1 + r1**2-r2**2)
    
    elif HNLtype == 'dirac':
        # Dirac -- helicity dependent
        kCM = mi*const.kallen_sqrt(1.0, r1**2, r2**2)/2.0
        diff_gamma *= 1/2 * np.abs(vertex_ij)**2 *(2*cost*h*kCM/mi + (1 + r1**2-r2**2) ) 
    else: 
        logger.error(f"HNL type {HNLtype} not recognized.")
        return 0.0 

    return diff_gamma

# gamma(Ni -> Nj V)  Ni(k) -> Nj(k1) S(k2)
def gamma_Ni_to_Nj_S(vertex_ij, mi, mj, mS, HNLtype = 'majorana'):
    
    r1 = mj/mi
    r2 = mS/mi

    gamma = mi / 16 / np.pi
    gamma *= const.kallen_sqrt(1, r1**2, r2**2)

    if HNLtype == 'majorana':
        # Majorana -- dependence real and imag part of vertex 
        gamma *= np.conjugate(vertex_ij)**2 * r1 - vertex_ij**2 *r1 + np.abs(vertex_ij)**2*(1 + r1**2 - r2**2)
    
    elif HNLtype == 'dirac':
        # Dirac -- dependence only on abs value of vertex
        gamma *= 1/2 *np.abs(vertex_ij)**2 *(1 + r1**2 - r2**2)

    return gamma


def diff_gamma_Ni_to_Nj_V(cost, vertex_ij, mi, mj, mV, HNLtype = 'majorana', h = -1):
    """diff_gamma_Ni_to_Nj_V 
        
        Differential decay rate

            dGamma/dcos(theta_V) (Ni(k) -> Nj(k1) V(k2))

    Parameters
    ----------
    cost : float or np.ndarray
        cosine of theta -- the angle between the vector boson and the z axis.
    vertex_ij : float
        the coupling vertex at the amplitude level (usually kinetic mixing * dark coupling * U_di U_dj or V_ij, or C_ij)
    mi : float
        mass of parent HNL
    mj : float
        mass of daughter HNL
    mV : float
        mass of the vector boson
    HNLtype : string, optional
        by default 'majorana'
    h : int, optional
        helicity of the parent HNL, by default -1

    Returns
    -------
    float or np.ndarray
        the differential decay rate in GeV
    """
    r1 = np.full_like(cost, mj/mi)
    r2 = np.full_like(cost, mV/mi)
    
    diff_gamma = mi**3 / mV**2 / 32 / np.pi
    diff_gamma *= const.kallen_sqrt(1, r1**2, r2**2)

    if HNLtype == 'majorana':
        # Majorana -- independent of cost 
        diff_gamma *=  3*r1*np.conjugate(vertex_ij)**2*r2**2 + 3*vertex_ij**2*r1*r2**2 + np.abs(vertex_ij)**2*(r1**4 + (r2**2-2)*r1**2 - 2*r2**4 + r2**2 + 1)
    
    elif HNLtype == 'dirac':
        # Dirac -- helicity dependent
        kCM = mi*const.kallen_sqrt(1.0, r1**2, r2**2)/2.0
        diff_gamma *= 2*cost*h*kCM/mi*(r1**2 + 2*r2**2 - 1) + (r1**4 + (r2**2 - 2)*r1**2 - 2*r2**4 + r2**2 + 1)
        diff_gamma *= 1/2 * np.abs(vertex_ij)**2
    else: 
        logger.error(f"HNL type {HNLtype} not recognized.")
        return 0.0 

    return diff_gamma

# gamma(Ni -> Nj V)  Ni(k) -> Nj(k1) V(k2)
def gamma_Ni_to_Nj_V(vertex_ij, mi, mj, mV, HNLtype = 'majorana'):
    
    r1 = mj/mi
    r2 = mV/mi

    gamma = mi / 16 / np.pi
    gamma *= const.kallen_sqrt(1, r1**2, r2**2)

    if HNLtype == 'majorana':
        # Majorana -- dependence real and imag part of vertex 
        gamma *= 3*r1*np.conjugate(vertex_ij)**2*r2**2 + 3*vertex_ij**2*r1*r2**2 + np.abs(vertex_ij)**2*(r1**4 + (r2**2-2)*r1**2 - 2*r2**4 + r2**2 + 1)
    
    elif HNLtype == 'dirac':
        # Dirac -- dependence only on abs value of vertex
        gamma *= 1/2 *np.abs(vertex_ij)**2 * (r1**4 + (r2**2 - 2)*r1**2 - 2*r2**4 + r2**2 + 1)

    return gamma


# d(gamma)/d(cos(theta))   Ni(k) -> Nj(k1) gamma(k2)
def diff_gamma_Ni_to_Nj_gamma(cost, vertex_ij, mi, mj, HNLtype = 'majorana', h = -1):
    
    rj = np.full_like(cost, mj/mi)
    k1CM = mi/2.0 * const.kallen_sqrt(1, rj**2, 0.0)
    
    diff_gamma = mi**3 / 16 / np.pi
    diff_gamma *= (1 - rj**2)**2

    if HNLtype == 'majorana':
        # Majorana -- independent of cost 
        diff_gamma *=  np.abs(vertex_ij)**2 * (1 - rj**2)
    
    elif HNLtype == 'dirac':
        # Dirac -- helicity dependent
        diff_gamma *= np.abs(vertex_ij)**2 * (1 - rj**2 - h*cost*k1CM/(mi/2.0))
    else: 
        logger.error(f"HNL type {HNLtype} not recognized.")
        return 0.0 

    return diff_gamma


# gamma(Ni -> Nj V)  Ni(k) -> Nj(k1) gamma(k2)
def gamma_Ni_to_Nj_gamma(vertex_ij, mi, mj, HNLtype = 'majorana'):
    
    rj = mj/mi

    gamma = mi**3 / 8 / np.pi
    gamma *= (1-rj**2)**3

    if HNLtype == 'majorana':
        # Majorana -- dependence real and imag part of vertex 
        gamma *= 2 * np.abs(vertex_ij)**2 

    elif HNLtype == 'dirac':
        # Dirac -- dependence only on abs value of vertex
        gamma *= np.abs(vertex_ij)**2 

    return gamma
# fmt: on


#
def diff_gamma_Ni_to_Nj_ell_ell(PS, process, diagrams=["total"]):
    """
        diff_gamma_Ni_to_Nj_ell_ell

        d(gamma)/dPS_3 (Ni (k1) --> ell-(k2)  ell+(k3)  Nj(k4))

        the differential decay rate in GeV for HNL decays to dileptons under all interactions assumptions.
        Phase space is parametrized as

            dPS_3 = (dm23^2 dm24^2 dOmega_4 dphi_34)/(32 m1^2 (2pi)^5)

        where dm23 == t and dm24 == u. Note that phi_4 can be integrated over trivially.


    Parameters
    ----------
    PS : list
        [t,u,v,c3,phi34] with mandelstam and angular phase space variables

    process : DarkNews.processes.FermionDileptonDecay()
        the main decay rate class with all model and scope parameters.

    diagrams : list, optional
        Diagrams to include when calculating the decay rate.
        If ["All"], returns a dictionary with all the rates for non-zero diagrams.
        Choices are:

                CC_SQR
                CC_NC_inter
                NC_SQR

                KinMix_SQR
                KinMix_NC_inter
                CC_KinMix_inter

                Scalar_SQR
                CC_Scalar_inter
                KinMix_Scalar_inter
                Scalar_NC_inter

                TMM_SQR

                TMM_NC_inter -- set to 0.0
                TMM_KinMix_inter -- set to 0.0

        By default diagrams=["total"], which will return the total decay rate summing all non-zero contributions.

    Returns
    -------
    float, np.ndarray, dict
        depending on the diagrams chosen and whether PS is a list of floats or np.ndarrays
        containing the differential decay rate in GeV.

    """
    # 3 body phase space
    t, u, v, c3, phi34 = PS

    #######################
    m1 = process.m_parent
    m2 = process.mm
    m3 = process.mp
    m4 = process.m_daughter

    cphi34 = np.cos(phi34)
    E3CM_decay = (m1**2 + m3**2 - u) / 2.0 / m1
    E4CM_decay = (m1**2 + m4**2 - t) / 2.0 / m1
    k3CM = const.kallen_sqrt(m1**2, u, m3**2) / 2.0 / m1
    k4CM = const.kallen_sqrt(m1**2, t, m4**2) / 2.0 / m1
    c34 = (t + u - m2**2 - m1**2 + 2 * E3CM_decay * E4CM_decay) / (2 * k3CM * k4CM)
    #
    c4 = c3 * c34 - np.sqrt(1.0 - c3 * c3) * np.sqrt(1.0 - c34 * c34) * cphi34

    #######################
    CCflag1 = process.CC_mixing1
    CCflag2 = process.CC_mixing2
    NCflag = 1  # NC always allowed, but controlled by Cih.
    Sflag = 1  # Include scalar
    DIP = 1  # Include dipole

    Cih = process.Cih
    Dih = process.Dih
    Sih = process.Sih
    dij = process.Tih / 2

    Cv = process.Cv
    Ca = process.Ca
    Dv = process.Dv
    Da = process.Da
    Ds = process.Ds
    Dp = process.Dp

    h = process.h_parent

    MZPRIME = process.mzprime
    MSCALAR = process.mhprime

    # k1k2 = (-v + m1**2 + m2**2)/2
    # k1k3 = -(u - m1**2 - m3**2)/2
    # k1k4 = (-t + m1**2 + m4**2)/2
    k2k3 = (t - m2**2 - m3**2) / 2
    k2k4 = (u - m4**2 - m2**2) / 2
    k3k4 = (v - m4**2 - m3**2) / 2

    # fmt: off
    # Dirac
    if process.HNLtype == "dirac":

        # SM CC SQR
        def Amp_CC_SQR():
            return (2*(CCflag1*CCflag1)*(gweak*gweak*gweak*gweak)*k2k4*(k2k3 + k3k4 + c3*h*k3CM*m1 + m3*m3))/((2*k3k4 + m3*m3 + m4*m4 - MW*MW)*(2*k3k4 + m3*m3 + m4*m4 - MW*MW))
        # SM CC NC interference
        def Amp_CC_NC_inter():
            return (-4*CCflag1*Cih*(gweak*gweak)*(Ca*(2*k2k3*k2k4 + k2k4*(2*k3k4 + 2*c3*h*k3CM*m1 - m2*m3 + 2*(m3*m3)) - m2*m3*(k3k4 + c4*h*k4CM*m1 + m4*m4)) + Cv*(2*k2k3*k2k4 + k2k4*(2*k3k4 + 2*c3*h*k3CM*m1 + m3*(m2 + 2*m3)) + m2*m3*(k3k4 + c4*h*k4CM*m1 + m4*m4)))*NCflag)/((2*k3k4 + m3*m3 + m4*m4 - MW*MW)*(2*k2k3 + m2*m2 + m3*m3 - MZBOSON*MZBOSON))
        #  SM CC KinMix interference
        def Amp_CC_KinMix_inter():
            return (-4*CCflag1*Dih*(gweak*gweak)*(Da*(2*k2k3*k2k4 + k2k4*(2*k3k4 + 2*c3*h*k3CM*m1 - m2*m3 + 2*(m3*m3)) - m2*m3*(k3k4 + c4*h*k4CM*m1 + m4*m4)) + Dv*(2*k2k3*k2k4 + k2k4*(2*k3k4 + 2*c3*h*k3CM*m1 + m3*(m2 + 2*m3)) + m2*m3*(k3k4 + c4*h*k4CM*m1 + m4*m4)))*NCflag)/((2*k3k4 + m3*m3 + m4*m4 - MW*MW)*(2*k2k3 + m2*m2 + m3*m3 - MZPRIME*MZPRIME))
        # SM CC scalar interference
        def Amp_CC_Scalar_inter():
            return (-2*CCflag1*Ds*(gweak*gweak)*((k2k3 + k3k4 + c3*h*k3CM*m1)*m2 - (k2k3 + k2k4 - h*(c3*k3CM + c4*k4CM)*m1 + m2*m2)*m3 + m2*(m3*m3))*m4*Sflag*Sih)/((2*k2k3 + m2*m2 + m3*m3 - MSCALAR*MSCALAR)*(2*k3k4 + m3*m3 + m4*m4 - MW*MW))
        # SM NC/CC SQR
        def Amp_NC_SQR():
            return (8*(Cih*Cih)*(2*Ca*Cv*(k2k3*(k2k4 - k3k4) + c3*h*k3CM*(k2k4 + k3k4)*m1 + c4*h*k3k4*k4CM*m1 - k3k4*(m2*m2) + k2k4*(m3*m3)) + Ca*Ca*(k2k3*(k2k4 + k3k4) - c3*h*k3CM*k3k4*m1 - c4*h*k3k4*k4CM*m1 + k3k4*(m2*m2) - k3k4*m2*m3 - c4*h*k4CM*m1*m2*m3 + k2k4*(2*k3k4 + c3*h*k3CM*m1 - m2*m3 + m3*m3) - m2*m3*(m4*m4)) + Cv*Cv*(k2k3*(k2k4 + k3k4) - c3*h*k3CM*k3k4*m1 - c4*h*k3k4*k4CM*m1 + k3k4*(m2*m2) + k3k4*m2*m3 + c4*h*k4CM*m1*m2*m3 + k2k4*(2*k3k4 + c3*h*k3CM*m1 + m3*(m2 + m3)) + m2*m3*(m4*m4)))*(NCflag*NCflag))/((2*k2k3 + m2*m2 + m3*m3 - MZBOSON*MZBOSON)*(2*k2k3 + m2*m2 + m3*m3 - MZBOSON*MZBOSON))
        # kinetic mixing term SQR
        def Amp_KinMix_SQR():
            return (8*(Dih*Dih)*(2*Da*Dv*(k2k3*(k2k4 - k3k4) + c3*h*k3CM*(k2k4 + k3k4)*m1 + c4*h*k3k4*k4CM*m1 - k3k4*(m2*m2) + k2k4*(m3*m3)) + Da*Da*(k2k3*(k2k4 + k3k4) - c3*h*k3CM*k3k4*m1 - c4*h*k3k4*k4CM*m1 + k3k4*(m2*m2) - k3k4*m2*m3 - c4*h*k4CM*m1*m2*m3 + k2k4*(2*k3k4 + c3*h*k3CM*m1 - m2*m3 + m3*m3) - m2*m3*(m4*m4)) + Dv*Dv*(k2k3*(k2k4 + k3k4) - c3*h*k3CM*k3k4*m1 - c4*h*k3k4*k4CM*m1 + k3k4*(m2*m2) + k3k4*m2*m3 + c4*h*k4CM*m1*m2*m3 + k2k4*(2*k3k4 + c3*h*k3CM*m1 + m3*(m2 + m3)) + m2*m3*(m4*m4)))*(NCflag*NCflag))/((2*k2k3 + m2*m2 + m3*m3 - MZPRIME*MZPRIME)*(2*k2k3 + m2*m2 + m3*m3 - MZPRIME*MZPRIME))
        # kinetic mixing + SM NC interference
        def Amp_KinMix_NC_inter():
            return (16*Cih*Dih*(Cv*Da*(k2k3*(k2k4 - k3k4) + c3*h*k3CM*(k2k4 + k3k4)*m1 + c4*h*k3k4*k4CM*m1 - k3k4*(m2*m2) + k2k4*(m3*m3)) + Ca*Dv*(k2k3*(k2k4 - k3k4) + c3*h*k3CM*(k2k4 + k3k4)*m1 + c4*h*k3k4*k4CM*m1 - k3k4*(m2*m2) + k2k4*(m3*m3)) + Ca*Da*(k2k3*(k2k4 + k3k4) - c3*h*k3CM*k3k4*m1 - c4*h*k3k4*k4CM*m1 + k3k4*(m2*m2) - k3k4*m2*m3 - c4*h*k4CM*m1*m2*m3 + k2k4*(2*k3k4 + c3*h*k3CM*m1 - m2*m3 + m3*m3) - m2*m3*(m4*m4)) + Cv*Dv*(k2k3*(k2k4 + k3k4) - c3*h*k3CM*k3k4*m1 - c4*h*k3k4*k4CM*m1 + k3k4*(m2*m2) + k3k4*m2*m3 + c4*h*k4CM*m1*m2*m3 + k2k4*(2*k3k4 + c3*h*k3CM*m1 + m3*(m2 + m3)) + m2*m3*(m4*m4)))*(NCflag*NCflag))/((2*k2k3 + m2*m2 + m3*m3 - MZBOSON*MZBOSON)*(2*k2k3 + m2*m2 + m3*m3 - MZPRIME*MZPRIME))
        # KinMix + Scalar interference
        def Amp_KinMix_Scalar_inter():
            return (8*Dih*Ds*Dv*((k2k3 + k3k4 + c3*h*k3CM*m1)*m2 - (k2k3 + k2k4 - h*(c3*k3CM + c4*k4CM)*m1 + m2*m2)*m3 + m2*(m3*m3))*m4*NCflag*Sflag*Sih)/((2*k2k3 + m2*m2 + m3*m3 - MSCALAR*MSCALAR)*(2*k2k3 + m2*m2 + m3*m3 - MZPRIME*MZPRIME))
        # transition magnetic moment SQR
        def Amp_TMM_SQR():
            return (16*(dij*dij)*(DIP*DIP)*(eQED*eQED)*(4*(k2k3*k2k3)*(k2k4 + k3k4) + 4*(k3k4*k3k4)*m2*(m2 + m3) + k3k4*m2*(4*c3*h*k3CM*m1*m2 + m3*(8*k2k4 - 4*c4*h*k4CM*m1 + 3*(m2*m2) + 2*m2*m3 + 3*(m3*m3))) + k2k3*(4*(k2k4*k2k4) + 4*(k3k4*k3k4) + 4*c3*h*k3CM*k3k4*m1 - 4*h*k2k4*(c3*k3CM + c4*k4CM)*m1 + 3*k2k4*((m2 + m3)*(m2 + m3)) + (m2 + m3)*(m2 + m3)*(3*k3k4 - c4*h*k4CM*m1 - m4*m4)) + m3*(4*(k2k4*k2k4)*(m2 + m3) + k2k4*(3*(m2*m2*m2) - 4*c3*h*k3CM*m1*m3 + 2*(m2*m2)*m3 + 3*m2*(m3*m3) - 4*c4*h*k4CM*m1*(m2 + m3)) - m2*((m2 + m3)*(m2 + m3))*(c4*h*k4CM*m1 + m4*m4))))/((2*k2k3 + m2*m2 + m3*m3)*(2*k2k3 + m2*m2 + m3*m3))
        # transition magnetic moment and NC interference
        def Amp_TMM_NC_inter():
            return 0.0
        # transition magnetic moment and Z' interference
        def Amp_TMM_KinMix_inter():
            return 0.0
        # scalar term SQR 
        def Amp_Scalar_SQR():
            return (-4*(Ds*Ds*(-k2k3 + m2*m3) + Dp*Dp*(k2k3 + m2*m3))*(k2k4 + k3k4 + c4*h*k4CM*m1 + m4*m4)*(Sflag*Sflag)*(Sih*Sih))/((2*k2k3 + m2*m2 + m3*m3 - MSCALAR*MSCALAR)*(2*k2k3 + m2*m2 + m3*m3 - MSCALAR*MSCALAR))
        # scalar + SM NC term
        def Amp_Scalar_NC_inter():
            return (8*Cih*Cv*Ds*((k2k3 + k3k4 + c3*h*k3CM*m1)*m2 - (k2k3 + k2k4 - h*(c3*k3CM + c4*k4CM)*m1 + m2*m2)*m3 + m2*(m3*m3))*m4*NCflag*Sflag*Sih)/((2*k2k3 + m2*m2 + m3*m3 - MSCALAR*MSCALAR)*(2*k2k3 + m2*m2 + m3*m3 - MZBOSON*MZBOSON))
        # scalar + TMM interference
        # def Amp_Scalar_TMM_inter():
            # return 
            
    elif process.HNLtype == "majorana":
        
        # SM CC SQR
        def Amp_CC_SQR():
            return 2*(gweak*gweak*gweak*gweak)*((CCflag2*CCflag2*k2k3*(k2k4 + k3k4 - c4*h*k4CM*m1 + m4*m4))/((2*k2k4 + m2*m2 + m4*m4 - MW*MW)*(2*k2k4 + m2*m2 + m4*m4 - MW*MW)) + (CCflag1*CCflag1*k2k4*(k2k3 + k3k4 + c3*h*k3CM*m1 + m3*m3))/((2*k3k4 + m3*m3 + m4*m4 - MW*MW)*(2*k3k4 + m3*m3 + m4*m4 - MW*MW)) + (CCflag1*CCflag2*m2*(k3k4*m1 - c4*h*k4CM*(k2k3 + k3k4 + m3*m3) + c3*h*k3CM*(k2k4 + k3k4 + m4*m4)))/((2*k2k4 + m2*m2 + m4*m4 - MW*MW)*(2*k3k4 + m3*m3 + m4*m4 - MW*MW)))
        # SM CC NC interference
        def Amp_CC_NC_inter():
            return (-4*Cih*(gweak*gweak)*(Ca*CCflag1*(2*k2k3*k2k4 + 2*k2k4*k3k4 + 2*c3*h*k2k4*k3CM*m1 - k2k4*m2*m3 - k3k4*m2*m3 - c4*h*k4CM*m1*m2*m3 + 2*k2k4*(m3*m3) + k2k3*(2*c3*h*k3CM + c4*h*k4CM + m1)*m4 + (-2*m1*m2*m3 + c4*h*k4CM*(k3k4 + m3*m3) + c3*h*k3CM*(k2k4 + k3k4 + m2*m2 + m3*m3))*m4 - m2*m3*(m4*m4))*(2*k2k4 + m2*m2 + m4*m4 - MW*MW) + CCflag1*Cv*(2*k2k3*k2k4 + 2*k2k4*k3k4 + 2*c3*h*k2k4*k3CM*m1 + k2k4*m2*m3 + k3k4*m2*m3 + c4*h*k4CM*m1*m2*m3 + 2*k2k4*(m3*m3) + k2k3*(2*c3*h*k3CM + c4*h*k4CM + m1)*m4 + (2*m1*m2*m3 + c4*h*k4CM*(k3k4 + m3*m3) + c3*h*k3CM*(k2k4 + k3k4 + m2*m2 + m3*m3))*m4 + m2*m3*(m4*m4))*(2*k2k4 + m2*m2 + m4*m4 - MW*MW) + CCflag2*Cv*(k3k4*m1*m2 - k2k4*m1*m3 - k2k3*m2*m4 - k3k4*m2*m4 + k2k3*m3*m4 + k2k4*m3*m4 + m2*m2*m3*m4 - m2*(m3*m3)*m4 + c3*h*k3CM*(m2 + m3)*(k2k4 + k3k4 + m4*(m1 + m4)) + c4*h*k4CM*(k2k3*(-m2 + m3) + k3k4*(-m2 + m3) + m3*(2*k2k4 + m2*(m2 - m3) + m4*(m1 + m4))))*(2*k3k4 + m3*m3 + m4*m4 - MW*MW) + Ca*CCflag2*(m1*(k3k4*m2 + k2k4*m3) - (k3k4*m2 + k2k3*(m2 + m3) + m3*(k2k4 + m2*(m2 + m3)))*m4 + c3*h*k3CM*(m2 - m3)*(k2k4 + k3k4 + m4*(m1 + m4)) - c4*h*k4CM*(k2k3*(m2 + m3) + k3k4*(m2 + m3) + m3*(2*k2k4 + m2*(m2 + m3) + m4*(m1 + m4))))*(2*k3k4 + m3*m3 + m4*m4 - MW*MW))*NCflag)/((2*k2k4 + m2*m2 + m4*m4 - MW*MW)*(2*k3k4 + m3*m3 + m4*m4 - MW*MW)*(2*k2k3 + m2*m2 + m3*m3 - MZBOSON*MZBOSON))
        #  SM CC KinMix interference
        def Amp_CC_KinMix_inter():
            return (4*Dih*(gweak*gweak)*((CCflag2*(-((Da + Dv)*k3k4*m1*m2) + (-Da + Dv)*k2k4*m1*m3 + ((Da + Dv)*(k2k3 + k3k4)*m2 + (Da - Dv)*(k2k3 + k2k4 + m2*m2)*m3 + (Da + Dv)*m2*(m3*m3))*m4 + c4*h*k4CM*(Dv*(k2k3*(m2 - m3) + k3k4*(m2 - m3) + m3*(-2*k2k4 + m2*(-m2 + m3) - m4*(m1 + m4))) + Da*(k2k3*(m2 + m3) + k3k4*(m2 + m3) + m3*(2*k2k4 + m2*(m2 + m3) + m4*(m1 + m4))))))/(2*k2k4 + m2*m2 + m4*m4 - MW*MW) + (CCflag1*(-2*(Da + Dv)*k2k4*(k2k3 + k3k4) + (Da - Dv)*(k2k4 + k3k4 + c4*h*k4CM*m1)*m2*m3 - 2*(Da + Dv)*k2k4*(m3*m3) - ((Da + Dv)*k2k3*m1 + 2*(-Da + Dv)*m1*m2*m3 + c4*(Da + Dv)*h*k4CM*(k2k3 + k3k4 + m3*m3))*m4 + (Da - Dv)*m2*m3*(m4*m4)))/(2*k3k4 + m3*m3 + m4*m4 - MW*MW) + c3*h*k3CM*(-((CCflag2*(Da*(m2 - m3) + Dv*(m2 + m3))*(k2k4 + k3k4 + m4*(m1 + m4)))/(2*k2k4 + m2*m2 + m4*m4 - MW*MW)) - (CCflag1*(Da + Dv)*(2*k2k4*m1 + (2*k2k3 + k2k4 + k3k4 + m2*m2 + m3*m3)*m4))/(2*k3k4 + m3*m3 + m4*m4 - MW*MW)))*NCflag)/(2*k2k3 + m2*m2 + m3*m3 - MZPRIME*MZPRIME)
        # SM CC scalar interference
        def Amp_CC_Scalar_inter():
            return (2*(gweak*gweak)*m4*((2*CCflag2*m1*((Dp - Ds)*k2k3 + (Dp + Ds)*m2*m3))/(2*k2k4 + m2*m2 + m4*m4 - MW*MW) + (CCflag1*((Dp - Ds)*(k2k3 + k3k4 + c3*h*k3CM*m1)*m2 + (Dp + Ds)*(k2k3 + k2k4 - h*(c3*k3CM + c4*k4CM)*m1 + m2*m2)*m3 + (Dp - Ds)*m2*(m3*m3)))/(2*k3k4 + m3*m3 + m4*m4 - MW*MW))*Sflag*Sih)/(2*k2k3 + m2*m2 + m3*m3 - MSCALAR*MSCALAR)
        # SM NC/CC SQR
        def Amp_NC_SQR():
            return (16*(Cih*Cih)*(Cv*Cv*(2*k2k4*k3k4 + k3k4*m2*(m2 + m3) + k2k4*m3*(m2 + m3) + 2*m1*m2*m3*m4 + m2*m3*(m4*m4) + k2k3*(k2k4 + k3k4 + m1*m4)) + Ca*Ca*(2*k2k4*k3k4 + k3k4*m2*(m2 - m3) + k2k4*m3*(-m2 + m3) - m2*m3*m4*(2*m1 + m4) + k2k3*(k2k4 + k3k4 + m1*m4)) + 2*Ca*Cv*h*(c4*k4CM*(k3k4*m1 + (k2k3 + k3k4 + m3*m3)*m4) + c3*k3CM*((k2k4 + k3k4)*m1 + (2*k2k3 + k2k4 + k3k4 + m2*m2 + m3*m3)*m4)))*(NCflag*NCflag))/((2*k2k3 + m2*m2 + m3*m3 - MZBOSON*MZBOSON)*(2*k2k3 + m2*m2 + m3*m3 - MZBOSON*MZBOSON))
        # kinetic mixing term SQR
        def Amp_KinMix_SQR():
            return (16*(Dih*Dih)*(Dv*Dv*(2*k2k4*k3k4 + k3k4*m2*(m2 + m3) + k2k4*m3*(m2 + m3) + 2*m1*m2*m3*m4 + m2*m3*(m4*m4) + k2k3*(k2k4 + k3k4 + m1*m4)) + Da*Da*(2*k2k4*k3k4 + k3k4*m2*(m2 - m3) + k2k4*m3*(-m2 + m3) - m2*m3*m4*(2*m1 + m4) + k2k3*(k2k4 + k3k4 + m1*m4)) + 2*Da*Dv*h*(c4*k4CM*(k3k4*m1 + (k2k3 + k3k4 + m3*m3)*m4) + c3*k3CM*((k2k4 + k3k4)*m1 + (2*k2k3 + k2k4 + k3k4 + m2*m2 + m3*m3)*m4)))*(NCflag*NCflag))/((2*k2k3 + m2*m2 + m3*m3 - MZPRIME*MZPRIME)*(2*k2k3 + m2*m2 + m3*m3 - MZPRIME*MZPRIME))
        # kinetic mixing + SM NC interference
        def Amp_KinMix_NC_inter():
            return (32*Cih*Dih*(Cv*Dv*(2*k2k4*k3k4 + k3k4*m2*(m2 + m3) + k2k4*m3*(m2 + m3) + 2*m1*m2*m3*m4 + m2*m3*(m4*m4) + k2k3*(k2k4 + k3k4 + m1*m4)) + Ca*Da*(2*k2k4*k3k4 + k3k4*m2*(m2 - m3) + k2k4*m3*(-m2 + m3) - m2*m3*m4*(2*m1 + m4) + k2k3*(k2k4 + k3k4 + m1*m4)) + Cv*Da*h*(c4*k4CM*(k3k4*m1 + (k2k3 + k3k4 + m3*m3)*m4) + c3*k3CM*((k2k4 + k3k4)*m1 + (2*k2k3 + k2k4 + k3k4 + m2*m2 + m3*m3)*m4)) + Ca*Dv*h*(c4*k4CM*(k3k4*m1 + (k2k3 + k3k4 + m3*m3)*m4) + c3*k3CM*((k2k4 + k3k4)*m1 + (2*k2k3 + k2k4 + k3k4 + m2*m2 + m3*m3)*m4)))*(NCflag*NCflag))/((2*k2k3 + m2*m2 + m3*m3 - MZBOSON*MZBOSON)*(2*k2k3 + m2*m2 + m3*m3 - MZPRIME*MZPRIME))
        # KinMix + Scalar interference
        def Amp_KinMix_Scalar_inter():
            return (8*Dih*(m1*(Ds*Dv*(-(k3k4*m2) + k2k4*m3) + Da*Dp*(k3k4*m2 + k2k4*m3)) - ((Da*Dp - Ds*Dv)*(k2k3 + k3k4)*m2 + (Da*Dp + Ds*Dv)*(k2k3 + k2k4 + m2*m2)*m3 + (Da*Dp - Ds*Dv)*m2*(m3*m3))*m4 - c3*h*k3CM*(Da*Dp*(m2 - m3) - Ds*Dv*(m2 + m3))*(k2k4 + k3k4 + m4*(m1 + m4)) + c4*h*k4CM*(Ds*Dv*(k2k3*(-m2 + m3) + k3k4*(-m2 + m3) + m3*(2*k2k4 + m2*(m2 - m3) + m4*(m1 + m4))) + Da*Dp*(k2k3*(m2 + m3) + k3k4*(m2 + m3) + m3*(2*k2k4 + m2*(m2 + m3) + m4*(m1 + m4)))))*NCflag*Sflag*Sih)/((2*k2k3 + m2*m2 + m3*m3 - MSCALAR*MSCALAR)*(2*k2k3 + m2*m2 + m3*m3 - MZPRIME*MZPRIME))
        # transition magnetic moment SQR
        def Amp_TMM_SQR():
            return (16*(dij*dij)*(DIP*DIP)*(eQED*eQED)*(4*(k2k3*k2k3)*(k2k4 + k3k4) + 4*(k3k4*k3k4)*m2*(m2 + m3) + k3k4*m2*(4*c3*h*k3CM*m1*m2 + m3*(8*k2k4 - 4*c4*h*k4CM*m1 + 3*(m2*m2) + 2*m2*m3 + 3*(m3*m3))) + k2k3*(4*(k2k4*k2k4) + 4*(k3k4*k3k4) + 4*c3*h*k3CM*k3k4*m1 - 4*h*k2k4*(c3*k3CM + c4*k4CM)*m1 + 3*k2k4*((m2 + m3)*(m2 + m3)) + (m2 + m3)*(m2 + m3)*(3*k3k4 - c4*h*k4CM*m1 - m4*m4)) + m3*(4*(k2k4*k2k4)*(m2 + m3) + k2k4*(3*(m2*m2*m2) - 4*c3*h*k3CM*m1*m3 + 2*(m2*m2)*m3 + 3*m2*(m3*m3) - 4*c4*h*k4CM*m1*(m2 + m3)) - m2*((m2 + m3)*(m2 + m3))*(c4*h*k4CM*m1 + m4*m4))))/((2*k2k3 + m2*m2 + m3*m3)*(2*k2k3 + m2*m2 + m3*m3))
        # transition magnetic moment and NC interference
        def Amp_TMM_NC_inter():
            return 0.0 #(16*Cih*dij*DIP*eQED*h*(2*Ca*(k2k4 + k3k4) + Cv*(-2*k2k4 + 2*k3k4 - m2*m2 + m3*m3))*NCflag*Eps(Momentum(k2),Momentum(k3),Momentum(k4),Momentum(s)))/((2*k2k3 + m2*m2 + m3*m3)*(2*k2k3 + m2*m2 + m3*m3 - MZBOSON*MZBOSON))
        # transition magnetic moment and Z' interference
        def Amp_TMM_KinMix_inter():
            return 0.0 #(16*Dih*dij*DIP*eQED*h*(2*Da*(k2k4 + k3k4) + Dv*(-2*k2k4 + 2*k3k4 - m2*m2 + m3*m3))*NCflag*Eps(Momentum(k2),Momentum(k3),Momentum(k4),Momentum(s)))/((2*k2k3 + m2*m2 + m3*m3)*(2*k2k3 + m2*m2 + m3*m3 - MZPRIME*MZPRIME))
        # scalar term SQR 
        def Amp_Scalar_SQR():
            return (4*(Ds*Ds*(k2k3 - m2*m3) + Dp*Dp*(k2k3 + m2*m3))*(k2k4 + k3k4 + c4*h*k4CM*m1 + m4*m4)*(Sflag*Sflag)*(Sih*Sih))/((2*k2k3 + m2*m2 + m3*m3 - MSCALAR*MSCALAR)*(2*k2k3 + m2*m2 + m3*m3 - MSCALAR*MSCALAR))
        # scalar + SM NC term
        def Amp_Scalar_NC_inter():
            return (8*Cih*(m1*(Cv*Ds*(-(k3k4*m2) + k2k4*m3) + Ca*Dp*(k3k4*m2 + k2k4*m3)) - ((Ca*Dp - Cv*Ds)*(k2k3 + k3k4)*m2 + (Ca*Dp + Cv*Ds)*(k2k3 + k2k4 + m2*m2)*m3 + (Ca*Dp - Cv*Ds)*m2*(m3*m3))*m4 - c3*h*k3CM*(Ca*Dp*(m2 - m3) - Cv*Ds*(m2 + m3))*(k2k4 + k3k4 + m4*(m1 + m4)) + c4*h*k4CM*(Cv*Ds*(k2k3*(-m2 + m3) + k3k4*(-m2 + m3) + m3*(2*k2k4 + m2*(m2 - m3) + m4*(m1 + m4))) + Ca*Dp*(k2k3*(m2 + m3) + k3k4*(m2 + m3) + m3*(2*k2k4 + m2*(m2 + m3) + m4*(m1 + m4)))))*NCflag*Sflag*Sih)/((2*k2k3 + m2*m2 + m3*m3 - MSCALAR*MSCALAR)*(2*k2k3 + m2*m2 + m3*m3 - MZBOSON*MZBOSON))
        # scalar + TMM interference
        # def Amp_Scalar_TMM_inter():
            # return 
    # fmt: on

    # Dict with all amplitude terms of interest
    Amp = {}

    # SM-like diagrams
    Amp["CC_SQR"] = Amp_CC_SQR
    Amp["CC_NC_inter"] = Amp_CC_NC_inter
    Amp["NC_SQR"] = Amp_NC_SQR

    if process.TheoryModel.has_vector_coupling:
        Amp["KinMix_SQR"] = Amp_KinMix_SQR
        Amp["KinMix_NC_inter"] = Amp_KinMix_NC_inter
        Amp["CC_KinMix_inter"] = Amp_CC_KinMix_inter

        if process.TheoryModel.has_scalar_coupling:
            Amp["KinMix_Scalar_inter"] = Amp_KinMix_Scalar_inter

    if process.TheoryModel.has_scalar_coupling:
        Amp["Scalar_SQR"] = Amp_Scalar_SQR
        Amp["Scalar_NC_inter"] = Amp_Scalar_NC_inter
        Amp["CC_Scalar_inter"] = Amp_CC_Scalar_inter
        # Amp['Scalar_TMM_inter'] = Amp_Scalar_TMM_inter

    if process.TheoryModel.has_TMM:
        Amp["TMM_SQR"] = Amp_TMM_SQR
        Amp["TMM_NC_inter"] = Amp_TMM_NC_inter

        if process.TheoryModel.has_vector_coupling:
            Amp["TMM_KinMix_inter"] = Amp_TMM_KinMix_inter

    # phase space is trivially integrated over phi4, but not over c4 and phi34.
    phase_space = 1 / (32 * m1**2 * (2 * np.pi) ** 4)
    flux_factor = 1 / (2 * m1)

    # note that there is no spin average factor for polarized decay.
    prefactor = flux_factor * phase_space

    # from amplitude to differential decay rate:
    diff_dr_terms = {}
    diff_dr_terms["total"] = 0.0
    for diagram, amplitude in Amp.items():
        if (diagram in diagrams) or ("all" in diagrams) or ("total" in diagrams):
            # all diff xsec terms
            diff_dr_terms[diagram] = amplitude() * prefactor
            # summing all contributions (Z,Z',S,interferences,etc)
            diff_dr_terms["total"] += diff_dr_terms[diagram]

    # raise warning for any requested diagram not picked up here and setting to zero
    for missing_diagram in list(set(diagrams) - set(diff_dr_terms.keys())):
        logger.warning(f"Warning: Diagram not found. Either not implemented or misspelled. Setting amplitude it to zero: {missing_diagram}")
        diff_dr_terms[missing_diagram] = prefactor * 0.0

    if "all" in diagrams:
        # return all individual diagrams in a dictionary
        return diff_dr_terms
    else:
        # return the sum of all diagrams requested
        return diff_dr_terms["total"]


# class HeavyNu:
#     def __init__(self,params,particle):

#         self.params=params
#         self.particle = particle

#         self.R_total = 0.0

#         self.R_nu_nu_nu = 0.0
#         self.R_nu4_nu_nu = 0.0
#         self.R_nu4_nu4_nu = 0.0
#         self.R_nu4_nu4_nu4 = 0.0
#         self.R_nu5_nu_nu = 0.0
#         self.R_nu5_nu5_nu = 0.0
#         self.R_nu5_nu5_nu5 = 0.0
#         self.R_nu_e_e = 0.0
#         self.R_nu_e_e_SM = 0.0
#         self.R_nu_e_mu = 0.0
#         self.R_nu_mu_mu = 0.0
#         self.R_e_pi = 0.0
#         self.R_e_K = 0.0
#         self.R_mu_pi = 0.0
#         self.R_mu_K = 0.0
#         self.R_nu_pi = 0.0
#         self.R_nu_eta = 0.0
#         self.R_nu_rho = 0.0
#         self.R_nu4_e_e = 0.0
#         self.R_nu4_e_mu = 0.0
#         self.R_nu4_mu_mu = 0.0
#         self.R_nu4_pi = 0.0
#         self.R_nu4_eta = 0.0
#         self.R_nu4_rho = 0.0
#         self.R_nu5_e_e = 0.0
#         self.R_nu5_e_mu = 0.0
#         self.R_nu5_mu_mu = 0.0
#         self.R_nu5_pi = 0.0
#         self.R_nu5_eta = 0.0
#         self.R_nu5_rho = 0.0
#         self.R_nu_gamma = 0.0
#         self.R_nu4_gamma = 0.0
#         self.R_nu5_gamma = 0.0

#         # self.BR_nu_nu_nu = 0.0
#         # self.BR_nu4_nu_nu = 0.0
#         # self.BR_nu4_nu4_nu = 0.0
#         # self.BR_nu4_nu4_nu4 = 0.0
#         # self.BR_nu_e_e = 0.0
#         # self.BR_nu_e_e_SM = 0.0
#         # self.BR_nu_e_mu = 0.0
#         # self.BR_nu_mu_mu = 0.0
#         # self.BR_e_pi = 0.0
#         # self.BR_e_K = 0.0
#         # self.BR_mu_pi = 0.0
#         # self.BR_mu_K = 0.0
#         # self.BR_nu_pi = 0.0
#         # self.BR_nu_eta = 0.0
#         # self.BR_nu_rho = 0.0
#         # self.BR_nu4_e_e = 0.0
#         # self.BR_nu4_e_mu = 0.0
#         # self.BR_nu4_mu_mu = 0.0
#         # self.BR_nu4_pi = 0.0
#         # self.BR_nu4_eta = 0.0
#         # self.BR_nu4_rho = 0.0
#         # self.BR_nu_gamma = 0.0
#         # self.BR_nu4_gamma = 0.0


#     def compute_rates(self):
#         params = self.params
#         ##################
#         # Neutrino 4
#         if self.particle==pdg.neutrino4:
#             mh = self.params.m4

#             # nuSM limit does not match -- FIX ME
#             self.R_nu_nu_nu = nuh_nui_nuj_nuk(params, pdg.neutrino4, const.nulight, const.nulight, const.nulight)
#             self.R_nu_gamma = nui_nu_gamma(params, pdg.neutrino4, const.nulight)

#             # dileptons -- already contain the Delta L = 2 channel
#             if mh > 2*const.m_e:
#                 self.R_nu_e_e = nui_nuj_ell1_ell2(params, pdg.neutrino4, const.nulight, const.electron, const.electron)
#                 self.R_nu_e_e_SM = nui_nuj_ell1_ell2(params, pdg.neutrino4, const.nulight, const.electron, const.electron, SM=True)
#             if mh > const.m_e + const.m_mu:
#                 self.R_nu_e_mu = nui_nuj_ell1_ell2(params, pdg.neutrino4, const.nulight, const.electron, const.muon)
#             if mh > 2*const.m_mu:
#                 self.R_nu_mu_mu = nui_nuj_ell1_ell2(params, pdg.neutrino4, const.nulight, const.muon, const.muon)
#             # pseudoscalar -- factor of 2 for delta L=2 channel
#             if mh > const.m_e+const.Mcharged_pion:
#                 self.R_e_pi = 2*nui_l_P(params, pdg.neutrino4, const.electron, const.charged_pion)
#             if mh > const.m_e+const.charged_kaon:
#                 self.R_e_K = 2*nui_l_P(params, pdg.neutrino4, const.electron, const.charged_kaon)
#             # pseudoscalar -- already contain the Delta L = 2 channel
#             if mh > const.m_mu+const.Mcharged_pion:
#                 self.R_mu_pi = 2*nui_l_P(params, pdg.neutrino4, const.muon, const.charged_pion)
#             if mh > const.m_mu+const.charged_kaon:
#                 self.R_mu_K = 2*nui_l_P(params, pdg.neutrino4, const.muon, const.charged_kaon)
#             if mh > const.Mneutral_pion:
#                 self.R_nu_pi = nui_nu_P(params, pdg.neutrino4, const.nulight, const.neutral_pion)
#             if mh > const.neutral_eta:
#                 self.R_nu_eta = nui_nu_P(params, pdg.neutrino4, const.nulight, const.neutral_eta)

#             # vector mesons
#             if mh > const.Mneutral_rho:
#                 self.R_nu_rho = nui_nu_V(params, pdg.neutrino5, const.nulight, const.neutral_rho)


#         ##################
#         # Neutrino 5
#         if self.particle==pdg.neutrino5:
#             mh = self.params.m5

#             # nuSM limit does not match -- FIX ME
#             self.R_nu_nu_nu = nuh_nui_nuj_nuk(params, pdg.neutrino5, const.nulight, const.nulight, const.nulight)
#             self.R_nu_gamma = nui_nu_gamma(params, pdg.neutrino5, const.nulight)

#             if mh > self.params.m4:
#                     self.R_nu4_nu_nu = nuh_nui_nuj_nuk(params, pdg.neutrino5, pdg.neutrino4, const.nulight, const.nulight)
#                     self.R_nu4_gamma = nui_nu_gamma(params, pdg.neutrino5, pdg.neutrino4)
#             if mh > 2*self.params.m4:
#                     self.R_nu4_nu4_nu = nuh_nui_nuj_nuk(params, pdg.neutrino5, pdg.neutrino4, pdg.neutrino4, const.nulight)
#             if mh > 3*self.params.m4:
#                     self.R_nu4_nu4_nu4 = nuh_nui_nuj_nuk(params, pdg.neutrino5, pdg.neutrino4, pdg.neutrino4, pdg.neutrino4)

#             # dileptons -- already contain the Delta L = 2 channel
#             if mh > 2*const.m_e:
#                 self.R_nu_e_e = nui_nuj_ell1_ell2(params, pdg.neutrino5, const.nulight, const.electron, const.electron)
#                 self.R_nu_e_e_SM = nui_nuj_ell1_ell2(params, pdg.neutrino5, const.nulight, const.electron, const.electron, SM=True)
#             if mh > const.m_e + const.m_mu:
#                 self.R_nu_e_mu = nui_nuj_ell1_ell2(params, pdg.neutrino5, const.nulight, const.electron, const.muon)
#             if mh > 2*const.m_mu:
#                 self.R_nu_mu_mu = nui_nuj_ell1_ell2(params, pdg.neutrino5, const.nulight, const.muon, const.muon)
#             # pseudoscalar -- factor of 2 for delta L=2 channel
#             if mh > const.m_e+const.Mcharged_pion:
#                 self.R_e_pi = 2*nui_l_P(params, pdg.neutrino5, const.electron, const.charged_pion)
#             if mh > const.m_e+const.charged_kaon:
#                 self.R_e_K = 2*nui_l_P(params, pdg.neutrino5, const.electron, const.charged_kaon)

#             # pseudoscalar -- already contain the Delta L = 2 channel
#             if mh > const.m_mu+const.Mcharged_pion:
#                 self.R_mu_pi = 2*nui_l_P(params, pdg.neutrino5, const.muon, const.charged_pion)
#             if mh > const.m_mu+const.charged_kaon:
#                 self.R_mu_K = 2*nui_l_P(params, pdg.neutrino5, const.muon, const.charged_kaon)
#             if mh > const.Mneutral_pion:
#                 self.R_nu_pi = nui_nu_P(params, pdg.neutrino5, const.nulight, const.neutral_pion)
#             if mh > const.neutral_eta:
#                 self.R_nu_eta = nui_nu_P(params, pdg.neutrino5, const.nulight, const.neutral_eta)

#             # vector mesons
#             if mh > const.Mneutral_rho:
#                 self.R_nu_rho = nui_nu_V(params, pdg.neutrino5, const.nulight, const.neutral_rho)


#             # dileptons -- already contain the Delta L = 2 channel
#             if mh > params.m4+2*const.m_e:
#                 self.R_nu4_e_e = nui_nuj_ell1_ell2(params, pdg.neutrino5, pdg.neutrino4, const.electron, const.electron)
#             if mh > params.m4+const.m_e + const.m_mu:
#                 self.R_nu4_e_mu = nui_nuj_ell1_ell2(params, pdg.neutrino5, pdg.neutrino4, const.electron, const.muon)
#             if mh > params.m4+2*const.m_mu:
#                 self.R_nu4_mu_mu = nui_nuj_ell1_ell2(params, pdg.neutrino5, pdg.neutrino4, const.muon, const.muon)
#             if mh > params.m4+const.Mneutral_pion:
#                 self.R_nu4_pi = nui_nu_P(params, pdg.neutrino5, pdg.neutrino4, const.neutral_pion)
#             if mh > params.m4+const.Mneutral_eta:
#                 self.R_nu4_eta = nui_nu_P(params, pdg.neutrino5, pdg.neutrino4, const.neutral_eta)
#             if mh > params.m4+const.Mneutral_rho:
#                 self.R_nu4_rho = nui_nu_V(params, pdg.neutrino5, pdg.neutrino4, const.neutral_rho)

#         self.array_R = [   self.R_nu_nu_nu,
#                             self.R_nu4_nu_nu,
#                             self.R_nu4_nu4_nu,
#                             self.R_nu4_nu4_nu4,
#                             self.R_nu_e_e,
#                             self.R_nu_e_mu,
#                             self.R_nu_mu_mu,
#                             self.R_e_pi,
#                             self.R_e_K,
#                             self.R_mu_pi,
#                             self.R_mu_K,
#                             self.R_nu_pi,
#                             self.R_nu_eta,
#                             self.R_nu4_e_e,
#                             self.R_nu4_e_mu,
#                             self.R_nu4_mu_mu,
#                             self.R_nu4_pi,
#                             self.R_nu4_eta,
#                             self.R_nu_rho,
#                             self.R_nu4_rho,
#                             self.R_nu_gamma,
#                             self.R_nu4_gamma,
#                             self.R_nu_e_e_SM]

#         ##################
#         # Neutrino 5
#         if self.particle==pdg.neutrino6:
#             mh = self.params.m6

#             # nuSM limit does not match -- FIX ME
#             self.R_nu_nu_nu = nuh_nui_nuj_nuk(params, pdg.neutrino6, const.nulight, const.nulight, const.nulight)
#             self.R_nu_gamma = nui_nu_gamma(params, pdg.neutrino6, const.nulight)

#             if mh > self.params.m5:
#                     self.R_nu5_nu_nu = nuh_nui_nuj_nuk(params, pdg.neutrino6, pdg.neutrino5, const.nulight, const.nulight)
#                     self.R_nu5_gamma = nui_nu_gamma(params, pdg.neutrino6, pdg.neutrino5)
#             if mh > 2*self.params.m5:
#                     self.R_nu5_nu5_nu = nuh_nui_nuj_nuk(params, pdg.neutrino6, pdg.neutrino5, pdg.neutrino5, const.nulight)
#             if mh > 3*self.params.m5:
#                     self.R_nu5_nu5_nu5 = nuh_nui_nuj_nuk(params, pdg.neutrino6, pdg.neutrino5, pdg.neutrino5, pdg.neutrino5)


#             if mh > self.params.m4:
#                     self.R_nu4_nu_nu = nuh_nui_nuj_nuk(params, pdg.neutrino6, pdg.neutrino4, const.nulight, const.nulight)
#                     self.R_nu4_gamma = nui_nu_gamma(params, pdg.neutrino6, pdg.neutrino4)
#             if mh > 2*self.params.m4:
#                     self.R_nu4_nu4_nu = nuh_nui_nuj_nuk(params, pdg.neutrino6, pdg.neutrino4, pdg.neutrino4, const.nulight)
#             if mh > 3*self.params.m4:
#                     self.R_nu4_nu4_nu4 = nuh_nui_nuj_nuk(params, pdg.neutrino6, pdg.neutrino4, pdg.neutrino4, pdg.neutrino4)


#             ###################################3
#             # FIX ME
#             # NEED TO IMPLEMENT MIXED FINAL STATE DECAYS!!!!!!!

#             # dileptons -- already contain the Delta L = 2 channel
#             if mh > 2*const.m_e:
#                 self.R_nu_e_e = nui_nuj_ell1_ell2(params, pdg.neutrino6, const.nulight, const.electron, const.electron)
#                 self.R_nu_e_e_SM = nui_nuj_ell1_ell2(params, pdg.neutrino6, const.nulight, const.electron, const.electron, SM=True)
#             if mh > const.m_e + const.m_mu:
#                 self.R_nu_e_mu = nui_nuj_ell1_ell2(params, pdg.neutrino6, const.nulight, const.electron, const.muon)
#             if mh > 2*const.m_mu:
#                 self.R_nu_mu_mu = nui_nuj_ell1_ell2(params, pdg.neutrino6, const.nulight, const.muon, const.muon)
#             # pseudoscalar -- factor of 2 for delta L=2 channel
#             if mh > const.m_e+const.Mcharged_pion:
#                 self.R_e_pi = 2*nui_l_P(params, pdg.neutrino6, const.electron, const.charged_pion)
#             if mh > const.m_e+const.charged_kaon:
#                 self.R_e_K = 2*nui_l_P(params, pdg.neutrino6, const.electron, const.charged_kaon)

#             # pseudoscalar -- already contain the Delta L = 2 channel
#             if mh > const.m_mu+const.Mcharged_pion:
#                 self.R_mu_pi = 2*nui_l_P(params, pdg.neutrino6, const.muon, const.charged_pion)
#             if mh > const.m_mu+const.charged_kaon:
#                 self.R_mu_K = 2*nui_l_P(params, pdg.neutrino6, const.muon, const.charged_kaon)
#             if mh > const.Mneutral_pion:
#                 self.R_nu_pi = nui_nu_P(params, pdg.neutrino6, const.nulight, const.neutral_pion)
#             if mh > const.neutral_eta:
#                 self.R_nu_eta = nui_nu_P(params, pdg.neutrino6, const.nulight, const.neutral_eta)

#             # vector mesons
#             if mh > const.Mneutral_rho:
#                 self.R_nu_rho = nui_nu_V(params, pdg.neutrino6, const.nulight, const.neutral_rho)


#             # dileptons -- already contain the Delta L = 2 channel
#             if mh > params.m5+2*const.m_e:
#                 self.R_nu5_e_e = nui_nuj_ell1_ell2(params, pdg.neutrino6, pdg.neutrino5, const.electron, const.electron)
#             if mh > params.m5+const.m_e + const.m_mu:
#                 self.R_nu5_e_mu = nui_nuj_ell1_ell2(params, pdg.neutrino6, pdg.neutrino5, const.electron, const.muon)
#             if mh > params.m5+2*const.m_mu:
#                 self.R_nu5_mu_mu = nui_nuj_ell1_ell2(params, pdg.neutrino6, pdg.neutrino5, const.muon, const.muon)
#             if mh > params.m5+const.Mneutral_pion:
#                 self.R_nu5_pi = nui_nu_P(params, pdg.neutrino6, pdg.neutrino5, const.neutral_pion)
#             if mh > params.m5+const.Mneutral_eta:
#                 self.R_nu5_eta = nui_nu_P(params, pdg.neutrino6, pdg.neutrino5, const.neutral_eta)
#             if mh > params.m5+const.Mneutral_rho:
#                 self.R_nu5_rho = nui_nu_V(params, pdg.neutrino6, pdg.neutrino5, const.neutral_rho)

#             # dileptons -- already contain the Delta L = 2 channel
#             if mh > params.m4+2*const.m_e:
#                 self.R_nu4_e_e = nui_nuj_ell1_ell2(params, pdg.neutrino6, pdg.neutrino4, const.electron, const.electron)
#             if mh > params.m4+const.m_e + const.m_mu:
#                 self.R_nu4_e_mu = nui_nuj_ell1_ell2(params, pdg.neutrino6, pdg.neutrino4, const.electron, const.muon)
#             if mh > params.m4+2*const.m_mu:
#                 self.R_nu4_mu_mu = nui_nuj_ell1_ell2(params, pdg.neutrino6, pdg.neutrino4, const.muon, const.muon)
#             if mh > params.m4+const.Mneutral_pion:
#                 self.R_nu4_pi = nui_nu_P(params, pdg.neutrino6, pdg.neutrino4, const.neutral_pion)
#             if mh > params.m4+const.Mneutral_eta:
#                 self.R_nu4_eta = nui_nu_P(params, pdg.neutrino6, pdg.neutrino4, const.neutral_eta)
#             if mh > params.m4+const.Mneutral_rho:
#                 self.R_nu4_rho = nui_nu_V(params, pdg.neutrino6, pdg.neutrino4, const.neutral_rho)

#         self.array_R = np.array([   self.R_nu_nu_nu,
#                             self.R_nu4_nu_nu,
#                             self.R_nu4_nu4_nu,
#                             self.R_nu4_nu4_nu4,
#                             self.R_nu5_nu_nu,
#                             self.R_nu5_nu5_nu,
#                             self.R_nu5_nu5_nu5,
#                             self.R_nu_e_e,
#                             self.R_nu_e_mu,
#                             self.R_nu_mu_mu,
#                             self.R_e_pi,
#                             self.R_e_K,
#                             self.R_mu_pi,
#                             self.R_mu_K,
#                             self.R_nu_pi,
#                             self.R_nu_eta,
#                             self.R_nu4_e_e,
#                             self.R_nu4_e_mu,
#                             self.R_nu4_mu_mu,
#                             self.R_nu4_pi,
#                             self.R_nu4_eta,
#                             self.R_nu5_e_e,
#                             self.R_nu5_e_mu,
#                             self.R_nu5_mu_mu,
#                             self.R_nu5_pi,
#                             self.R_nu5_eta,
#                             self.R_nu_rho,
#                             self.R_nu4_rho,
#                             self.R_nu5_rho,
#                             self.R_nu_gamma,
#                             self.R_nu4_gamma,
#                             self.R_nu5_gamma,
#                             self.R_nu_e_e_SM])

#     def total_rate(self):
#         # self.R_total =  self.R_nu_nu_nu+self.R_nu4_nu_nu+self.R_nu4_nu4_nu+self.R_nu4_nu4_nu4+self.R_nu_e_e+self.R_nu_e_mu\
#         #                               +self.R_nu_mu_mu+self.R_e_pi+self.R_e_K+self.R_mu_pi+self.R_mu_K+self.R_nu_pi+self.R_nu_eta+self.R_nu4_e_e\
#         #                               +self.R_nu4_e_mu+self.R_nu4_mu_mu+self.R_nu4_pi+self.R_nu4_eta+self.R_nu4_rho+self.R_nu_rho+self.R_nu_gamma+self.R_nu4_gamma
#         self.R_total =  np.sum(self.array_R[:-1])

#         return self.R_total

#     def compute_BR(self):

#         self.BR_nu_nu_nu = self.R_nu_nu_nu/self.R_total
#         self.BR_nu4_nu_nu = self.R_nu4_nu_nu/self.R_total
#         self.BR_nu4_nu4_nu = self.R_nu4_nu4_nu/self.R_total
#         self.BR_nu4_nu4_nu4 = self.R_nu4_nu4_nu4/self.R_total
#         self.BR_nu_e_e = self.R_nu_e_e/self.R_total
#         self.BR_nu_e_mu = self.R_nu_e_mu/self.R_total
#         self.BR_nu_mu_mu = self.R_nu_mu_mu/self.R_total
#         self.BR_e_pi = self.R_e_pi/self.R_total
#         self.BR_e_K = self.R_e_K/self.R_total
#         self.BR_mu_pi = self.R_mu_pi/self.R_total
#         self.BR_mu_K = self.R_mu_K/self.R_total
#         self.BR_nu_pi = self.R_nu_pi/self.R_total
#         self.BR_nu_eta = self.R_nu_eta/self.R_total
#         self.BR_nu4_e_e = self.R_nu4_e_e/self.R_total
#         self.BR_nu4_e_mu = self.R_nu4_e_mu/self.R_total
#         self.BR_nu4_mu_mu = self.R_nu4_mu_mu/self.R_total
#         self.BR_nu4_pi = self.R_nu4_pi/self.R_total
#         self.BR_nu4_eta = self.R_nu4_eta/self.R_total
#         self.BR_nu4_rho = self.R_nu4_rho/self.R_total
#         self.BR_nu_rho = self.R_nu_rho/self.R_total
#         self.BR_nu4_gamma = self.R_nu4_gamma/self.R_total
#         self.BR_nu_gamma = self.R_nu_gamma/self.R_total

#         # self.array_BR = [   self.BR_nu_nu_nu,
#         #                   self.BR_nu4_nu_nu,
#         #                   self.BR_nu4_nu4_nu,
#         #                   self.BR_nu4_nu4_nu4,
#         #                   self.BR_nu_e_e,
#         #                   self.BR_nu_e_mu,
#         #                   self.BR_nu_mu_mu,
#         #                   self.BR_e_pi,
#         #                   self.BR_e_K,
#         #                   self.BR_mu_pi,
#         #                   self.BR_mu_K,
#         #                   self.BR_nu_pi,
#         #                   self.BR_nu_eta,
#         #                   self.BR_nu4_e_e,
#         #                   self.BR_nu4_e_mu,
#         #                   self.BR_nu4_mu_mu,
#         #                   self.BR_nu4_pi,
#         #                   self.BR_nu4_eta,
#         #                   self.BR_nu_rho,
#         #                   self.BR_nu4_rho,
#         #                   self.BR_nu_gamma,
#         #                   self.BR_nu4_gamma]

#         self.array_BR = self.array_R[:-1]/self.R_total

#         return self.array_BR

# def nui_nu_gamma(params, initial_neutrino, final_neutrino):

#     if (initial_neutrino==pdg.neutrino6):
#         mh = params.m6
#         if (final_neutrino==const.neutrino_tau):
#             CC_mixing = params.Utau6
#             mf=0.0
#         elif(final_neutrino==const.neutrino_muon):
#             CC_mixing = params.Umu6
#             mf=0.0
#         elif(final_neutrino==const.neutrino_electron):
#             CC_mixing = params.Ue6
#             mf=0.0
#         elif(final_neutrino==const.nulight):
#             CC_mixing = np.sqrt(params.Ue6*params.Ue6+params.Umu6*params.Umu6+params.Utau6*params.Utau6)
#             mf=0.0
#         elif(final_neutrino==pdg.neutrino5):
#             CC_mixing = params.c56
#             mf=params.m5
#         elif(final_neutrino==pdg.neutrino4):
#             CC_mixing = params.c46
#             mf=params.m4

#     elif (initial_neutrino==pdg.neutrino5):
#         mh = params.m5
#         if (final_neutrino==const.neutrino_tau):
#             CC_mixing = params.Utau5
#             mf=0.0
#         elif(final_neutrino==const.neutrino_muon):
#             CC_mixing = params.Umu5
#             mf=0.0
#         elif(final_neutrino==const.neutrino_electron):
#             CC_mixing = params.Ue5
#             mf=0.0
#         elif(final_neutrino==const.nulight):
#             CC_mixing = np.sqrt(params.Ue5*params.Ue5+params.Umu5*params.Umu5+params.Utau5*params.Utau5)
#             mf=0.0
#         elif(final_neutrino==pdg.neutrino4):
#             CC_mixing = params.c45
#             mf=params.m4


#     elif(initial_neutrino==pdg.neutrino4):
#         mh=params.m4
#         mf=0.0
#         if (final_neutrino==const.neutrino_tau):
#             CC_mixing = params.Utau4
#         elif(final_neutrino==const.neutrino_muon):
#             CC_mixing = params.Umu4
#         elif(final_neutrino==const.neutrino_electron):
#             CC_mixing = params.Ue4
#         elif(final_neutrino==const.nulight):
#             CC_mixing = np.sqrt(params.Ue4*params.Ue4+params.Umu4*params.Umu4+params.Utau4*params.Utau4)
#         else:
#             print('ERROR! Wrong inital neutrino')

#     return (const.Gf*CC_mixing)*(const.Gf*CC_mixing)*mh**5/(192.0*np.pi*np.pi*np.pi)*(27.0*const.alphaQED/32.0/np.pi)*(const.kallen_sqrt(1.0, mf/mh*mf/mh,0.0))


# def nui_l_P(params, initial_neutrino, final_lepton, final_hadron):


#     mh = initial_neutrino.mass/1e3
#     ml = final_lepton.mass/1e3
#     CC_mixing = params.Ulep[pdg.get_lepton_index(final_lepton)]

#     if (initial_neutrino==pdg.neutrino6):
#         mh = params.m6
#         if (final_lepton==const.tau):
#             ml = const.m_tau
#             CC_mixing = params.Utau6
#         elif(final_lepton==const.muon):
#             ml = const.m_mu
#             CC_mixing = params.Umu6
#         elif(final_lepton==const.electron):
#             ml = const.m_e
#             CC_mixing = params.Ue6

#     elif (initial_neutrino==pdg.neutrino5):
#         mh = params.m5
#         if (final_lepton==const.tau):
#             ml = const.m_tau
#             CC_mixing = params.Utau5
#         elif(final_lepton==const.muon):
#             ml = const.m_mu
#             CC_mixing = params.Umu5
#         elif(final_lepton==const.electron):
#             ml = const.m_e
#             CC_mixing = params.Ue5

#     elif(initial_neutrino==pdg.neutrino4):
#         mh = params.m4
#         if (final_lepton==const.tau):
#             ml = const.m_tau
#             CC_mixing = params.Utau4
#         elif(final_lepton==const.muon):
#             ml = const.m_mu
#             CC_mixing = params.Umu4
#         elif(final_lepton==const.electron):
#             ml = const.m_e
#             CC_mixing = params.Ue4
#     else:
#             print('ERROR! Wrong inital neutrino')

#     if (final_hadron==const.charged_pion):
#         mp = const.Mcharged_pion
#         Vqq = const.Vud
#         fp  = const.Fcharged_pion
#     elif(final_hadron==const.charged_kaon):
#         mp = const.Mcharged_kaon
#         Vqq = const.Vus
#         fp  = const.Fcharged_kaon
#     # elif(final_hadron==const.charged_rho):
#     #   mp = const.Mcharged_rho
#     #   Vqq = params.Vud

#     return (const.Gf*fp*CC_mixing*Vqq)**2 * mh**3/(16*np.pi) * I1_2body((ml/mh)**2, (mp/mh)**2)


# def nui_nu_P(params, initial_neutrino, final_neutrino, final_hadron):

#     if (initial_neutrino==pdg.neutrino6):
#         mh = params.m6
#         if (final_neutrino==const.neutrino_tau):
#             CC_mixing = params.Utau6
#         elif(final_neutrino==const.neutrino_muon):
#             CC_mixing = params.Umu6
#         elif(final_neutrino==const.neutrino_electron):
#             CC_mixing = params.Ue6
#         elif(final_neutrino==const.nulight):
#             CC_mixing = np.sqrt(params.Ue6*params.Ue6+params.Umu6*params.Umu6+params.Utau6*params.Utau6)
#         elif(final_neutrino==pdg.neutrino5):
#             CC_mixing = params.c56
#         elif(final_neutrino==pdg.neutrino4):
#             CC_mixing = params.c46

#     elif (initial_neutrino==pdg.neutrino5):
#         mh = params.m5
#         if (final_neutrino==const.neutrino_tau):
#             CC_mixing = params.Utau5
#         elif(final_neutrino==const.neutrino_muon):
#             CC_mixing = params.Umu5
#         elif(final_neutrino==const.neutrino_electron):
#             CC_mixing = params.Ue5
#         elif(final_neutrino==const.nulight):
#             CC_mixing = np.sqrt(params.Ue5*params.Ue5+params.Umu5*params.Umu5+params.Utau5*params.Utau5)
#         elif(final_neutrino==pdg.neutrino4):
#             CC_mixing = params.c45


#     elif(initial_neutrino==pdg.neutrino4):
#         mh = params.m4
#         if (final_neutrino==const.neutrino_tau):
#             CC_mixing = params.Utau4
#         elif(final_neutrino==const.neutrino_muon):
#             CC_mixing = params.Umu4
#         elif(final_neutrino==const.neutrino_electron):
#             CC_mixing = params.Ue4
#         elif(final_neutrino==const.nulight):
#             CC_mixing = np.sqrt(params.Ue4*params.Ue4+params.Umu4*params.Umu4+params.Utau4*params.Utau4)
#         else:
#             print('ERROR! Wrong inital neutrino')

#     if (final_hadron==const.neutral_pion):
#         mp = const.Mneutral_pion
#         fp  = const.Fneutral_pion
#     elif(final_hadron==const.neutral_eta):
#         mp = const.Mneutral_eta
#         fp  = const.Fneutral_eta


#     return (const.Gf*fp*CC_mixing)**2*mh**3/(64*np.pi)*(1-(mp/mh)**2)**2


# def nui_nu_V(params, initial_neutrino, final_neutrino, final_hadron):

#     if (initial_neutrino==pdg.neutrino6):
#         mh = params.m6
#         mix = params.A6
#     elif (initial_neutrino==pdg.neutrino5):
#         mh = params.m5
#         mix = params.A5
#     elif(initial_neutrino==pdg.neutrino4):
#         mh = params.m4
#         mix = params.A4

#     if (final_hadron==const.neutral_rho):
#         mp = const.Mneutral_rho
#         fp  = const.Fneutral_rho
#     else:
#         print('ERROR! Final state not recognized.')

#     rp  = (mp/mh)*(mp/mh)
#     bsm = (const.eQED*params.chi*const.cw)**2*params.alphaD*mh*mh*mh*fp*fp*(1-rp)*(1-rp)*(0.5+rp)/4.0/params.mzprime/params.mzprime/params.mzprime/params.mzprime
#     sm  = const.Gf*const.Gf*mh*mh*mh*fp*fp*(1-rp)*(1-rp)*(0.5+rp)/16.0/np.pi
#     return (sm+bsm)*mix


# ###############################
# # New containing all terms!
# def nui_nuj_ell1_ell2(params, initial_neutrino, final_neutrino, final_lepton1, final_lepton2, SM=False):

#     ################################
#     # COUPLINGS

#     # Is neutral current possible?
#     if final_lepton2==final_lepton1:
#         NCflag=1
#         if initial_neutrino==pdg.neutrino6:
#             mh = params.m6

#             # Which outgoing neutrino?
#             if final_neutrino==const.neutrino_electron:
#                 Cih = params.ce6
#                 Dih = params.de6
#                 mf = 0.0
#             if final_neutrino==const.neutrino_muon:
#                 Cih = params.cmu6
#                 Dih = params.dmu6
#                 mf = 0.0
#             if final_neutrino==const.neutrino_tau:
#                 Cih = params.ctau6
#                 Dih = params.dtau6
#                 mf = 0.0
#             if final_neutrino==const.nulight:
#                 Cih = params.clight6
#                 Dih = params.dlight6
#                 mf = 0.0
#             if final_neutrino==pdg.neutrino5:
#                 Cih = params.c56*0
#                 Dih = params.d56*0
#                 mf = params.m5
#             if final_neutrino==pdg.neutrino4:
#                 Cih = params.c46
#                 Dih = params.d46
#                 mf = params.m4

#         elif initial_neutrino==pdg.neutrino5:
#             mh = params.m5

#             # Which outgoing neutrino?
#             if final_neutrino==const.neutrino_electron:
#                 Cih = params.ce5
#                 Dih = params.de5
#                 mf = 0.0
#             if final_neutrino==const.neutrino_muon:
#                 Cih = params.cmu5
#                 Dih = params.dmu5
#                 mf = 0.0
#             if final_neutrino==const.neutrino_tau:
#                 Cih = params.ctau5
#                 Dih = params.dtau5
#                 mf = 0.0
#             if final_neutrino==const.nulight:
#                 Cih = params.clight5
#                 Dih = params.dlight5
#                 mf = 0.0
#             if final_neutrino==pdg.neutrino4:
#                 Cih = params.c45
#                 Dih = params.d45
#                 mf = params.m4

#         elif initial_neutrino==pdg.neutrino4:
#             mh = params.m4
#             # Which outgoing neutrino?
#             if final_neutrino==const.neutrino_electron:
#                 Cih = params.ce4
#                 Dih = params.de4
#                 mf = 0.0
#             if final_neutrino==const.neutrino_muon:
#                 Cih = params.cmu4
#                 Dih = params.dmu4
#                 mf = 0.0
#             if final_neutrino==const.neutrino_tau:
#                 Cih = params.ctau4
#                 Dih = params.dtau4
#                 mf = 0.0
#             if final_neutrino==const.nulight:
#                 Cih = params.clight4
#                 Dih = params.dlight4
#                 mf = 0.0
#             if final_neutrino==pdg.neutrino4:
#                 print('ERROR! (nu4 -> nu4 l l) is kinematically not allowed!')

#         if final_neutrino/10==final_lepton1:
#             # Mixing required for CC N-like
#             if (final_lepton1==const.tau):
#                 CC_mixing1 = params.Utau4
#             elif(final_lepton1==const.muon):
#                 CC_mixing1 = params.Umu4
#             elif(final_lepton1==const.electron):
#                 CC_mixing1 = params.Ue4
#             else:
#                 print("WARNING! Unable to set CC mixing parameter for decay. Assuming 0.")
#                 CC_mixing1 = 0

#             # Mixing required for CC Nbar-like
#             if (final_lepton2==const.tau):
#                 CC_mixing2 = params.Utau4
#             elif(final_lepton2==const.muon):
#                 CC_mixing2 = params.Umu4
#             elif(final_lepton2==const.electron):
#                 CC_mixing2 = params.Ue4
#             else:
#                 print("WARNING! Unable to set CC mixing parameter for decay. Assuming 0.")
#                 CC_mixing2 = 0
#         else:
#             CC_mixing1 = 0
#             CC_mixing2 = 0

#     # Only CC is possible
#     # FIX ME!
#     # NEED TO INCLUDE ALL HNL MIXINGS
#     else:
#         NCflag=0
#         Cih = 0
#         Dih = 0
#         if initial_neutrino==pdg.neutrino6:
#             mh = params.m6
#             # Which outgoing neutrino?
#             if final_neutrino==pdg.neutrino5:
#                 mf = params.m5
#                 # Which outgoin leptons?
#                 if final_lepton1==const.electron and final_lepton2==const.muon:
#                     CC_mixing1 = params.Umu6 * params.Ue5
#                     CC_mixing2 = params.Ue6 * params.Umu5
#                 if final_lepton1==const.electron and final_lepton2==const.tau:
#                     CC_mixing1 = params.Utau6 * params.Ue5
#                     CC_mixing2 = params.Ue6 * params.Utau5
#                 if final_lepton1==const.muon and final_lepton2==const.tau:
#                     CC_mixing1 = params.Umuon6 * params.Utau5
#                     CC_mixing2 = params.Utau6 * params.Umuon5

#             elif final_neutrino==pdg.neutrino4:
#                 mf = params.m4
#                 # Which outgoin leptons?
#                 if final_lepton1==const.electron and final_lepton2==const.muon:
#                     CC_mixing1 = params.Umu6 * params.Ue4
#                     CC_mixing2 = params.Ue6 * params.Umu4
#                 if final_lepton1==const.electron and final_lepton2==const.tau:
#                     CC_mixing1 = params.Utau6 * params.Ue4
#                     CC_mixing2 = params.Ue6 * params.Utau4
#                 if final_lepton1==const.muon and final_lepton2==const.tau:
#                     CC_mixing1 = params.Umuon6 * params.Utau4
#                     CC_mixing2 = params.Utau6 * params.Umuon4


#             elif final_neutrino==const.nulight:
#                 mf = 0.0
#                 if final_lepton1==const.electron and final_lepton2==const.muon:
#                     CC_mixing1 = params.Umu5
#                     CC_mixing2 = params.Ue5
#                 if final_lepton1==const.electron and final_lepton2==const.tau:
#                     CC_mixing1 = params.Utau5
#                     CC_mixing2 = params.Ue5
#                 if final_lepton1==const.muon and final_lepton2==const.tau:
#                     CC_mixing1 = params.Umuon5
#                     CC_mixing2 = params.Utau5

#         if initial_neutrino==pdg.neutrino5:
#             mh = params.m5
#             # Which outgoing neutrino?
#             if final_neutrino==pdg.neutrino4:
#                 mf = params.m4
#                 # Which outgoin leptons?
#                 if final_lepton1==const.electron and final_lepton2==const.muon:
#                     CC_mixing1 = params.Umu5 * params.Ue4
#                     CC_mixing2 = params.Ue5 * params.Umu4
#                 if final_lepton1==const.electron and final_lepton2==const.tau:
#                     CC_mixing1 = params.Utau5 * params.Ue4
#                     CC_mixing2 = params.Ue5 * params.Utau4
#                 if final_lepton1==const.muon and final_lepton2==const.tau:
#                     CC_mixing1 = params.Umuon5 * params.Utau4
#                     CC_mixing2 = params.Utau5 * params.Umuon4
#             if final_neutrino==const.nulight:
#                 mf = 0.0
#                 if final_lepton1==const.electron and final_lepton2==const.muon:
#                     CC_mixing1 = params.Umu5
#                     CC_mixing2 = params.Ue5
#                 if final_lepton1==const.electron and final_lepton2==const.tau:
#                     CC_mixing1 = params.Utau5
#                     CC_mixing2 = params.Ue5
#                 if final_lepton1==const.muon and final_lepton2==const.tau:
#                     CC_mixing1 = params.Umuon5
#                     CC_mixing2 = params.Utau5


#         if initial_neutrino==pdg.neutrino4:
#             mh = params.m4
#             mf = 0.0
#             if final_neutrino==const.nulight:
#                 if final_lepton1==const.electron and final_lepton2==const.muon:
#                     CC_mixing1 = params.Umu4
#                     CC_mixing2 = params.Ue4
#                 if final_lepton1==const.electron and final_lepton2==const.tau:
#                     CC_mixing1 = params.Utau4
#                     CC_mixing2 = params.Ue4
#                 if final_lepton1==const.muon and final_lepton2==const.tau:
#                     CC_mixing1 = params.Umuon4
#                     CC_mixing2 = params.Utau4
#             else:
#                 print('ERROR! Unable to find outgoing neutrino.')


#     #######################################
#     #######################################
#     ### WATCH OUT FOR THE MINUS SIGN HERE -- IMPORTANT FOR INTERFERENCE
#     ## Put requires mixings in CCflags
#     CCflag1 = CC_mixing1
#     CCflag2 = -CC_mixing2

#     ##############################
#     # CHARGED LEPTON MASSES

#     if (final_lepton1==const.tau):
#         mm = const.m_tau
#     elif(final_lepton1==const.muon):
#         mm = const.m_mu
#     elif(final_lepton1==const.electron):
#         mm = const.m_e
#     else:
#         print("WARNING! Unable to set charged lepton mass. Assuming massless.")
#         mm = 0

#     if (final_lepton2==const.tau):
#         mp = const.m_tau
#     elif(final_lepton2==const.muon):
#         mp = const.m_mu
#     elif(final_lepton2==const.electron):
#         mp = const.m_e
#     else:
#         print("WARNING! Unable to set charged lepton mass. Assuming massless.")
#         mp = 0


#     # couplings and masses LEADING ORDER!
#     Cv = params.ceV
#     Ca = params.ceA
#     Dv = params.deV
#     Da = params.deA
#     MZBOSON = const.Mz
#     mzprime = params.mzprime
#     MW = const.Mw

#     xZBOSON=MZBOSON/mh
#     xZPRIME=mzprime/mh
#     xWBOSON=MW/mh
#     m1=mh
#     x2=mf/mh
#     x3=mm/mh
#     x4=mp/mh

#     gweak=const.gweak

#     if SM==True:
#         Dv=0
#         Da=0
#         mzprime=0
#         gD=0

#     def DGammaDuDt(x23,x24,m1,x2,x3,x4,NCflag,CCflag1,CCflag2,Cv,Ca,Dv,Da,Cih,Dih,MZBOSON,mzprime,MW):
#         pi = np.pi

#         # return (u*((-256*(Ca*Ca)*(Cih*Cih)*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (256*(Cih*Cih)*(Cv*Cv)*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) - (64*(Ca*Ca)*(Cih*Cih)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (64*(Cih*Cih)*(Cv*Cv)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) - (256*(Da*Da)*(Dih*Dih)*mf*mh*mm*mp*(NCflag*NCflag))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) + (256*(Dih*Dih)*(Dv*Dv)*mf*mh*mm*mp*(NCflag*NCflag))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) - (64*(Da*Da)*(Dih*Dih)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) + (64*(Dih*Dih)*(Dv*Dv)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) + (64*(Ca*Ca)*(Cih*Cih)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (64*(Cih*Cih)*(Cv*Cv)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (64*(Da*Da)*(Dih*Dih)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) + (64*(Dih*Dih)*(Dv*Dv)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) + (512*Ca*Cih*Da*Dih*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) - (512*Cih*Cv*Dih*Dv*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) + (128*Ca*Cih*Da*Dih*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) - (128*Cih*Cv*Dih*Dv*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) - (128*Ca*Cih*Da*Dih*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) - (128*Cih*Cv*Dih*Dv*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) - (32*Ca*CCflag1*Cih*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (32*CCflag1*Cih*Cv*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(MW*MW - u)) - (8*Ca*CCflag1*Cih*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Cih*Cv*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) - (32*CCflag1*Da*Dih*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((mzprime*mzprime - t)*(MW*MW - u)) + (32*CCflag1*Dih*Dv*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((mzprime*mzprime - t)*(MW*MW - u)) - (8*CCflag1*Da*Dih*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((mzprime*mzprime - t)*(MW*MW - u)) + (8*CCflag1*Dih*Dv*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((mzprime*mzprime - t)*(MW*MW - u)) + (8*Ca*CCflag1*Cih*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Cih*Cv*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Da*Dih*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((mzprime*mzprime - t)*(MW*MW - u)) + (8*CCflag1*Dih*Dv*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((mzprime*mzprime - t)*(MW*MW - u)) + (32*(Ca*Ca)*(Cih*Cih)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Cih*Cih)*(Cv*Cv)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Da*Da)*(Dih*Dih)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) + (32*(Dih*Dih)*(Dv*Dv)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) - (64*Ca*Cih*Da*Dih*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) - (64*Cih*Cv*Dih*Dv*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) + (CCflag1*CCflag1*(const.g*const.g*const.g*const.g)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/(2.*((MW*MW - u)*(MW*MW - u))) + (4*Ca*CCflag1*Cih*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Cih*Cv*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Da*Dih*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((mzprime*mzprime - t)*(MW*MW - u)) + (4*CCflag1*Dih*Dv*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((mzprime*mzprime - t)*(MW*MW - u)) + (32*(Ca*Ca)*(Cih*Cih)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Cih*Cih)*(Cv*Cv)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Da*Da)*(Dih*Dih)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) + (32*(Dih*Dih)*(Dv*Dv)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) - (64*Ca*Cih*Da*Dih*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) - (64*Cih*Cv*Dih*Dv*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) + (CCflag1*CCflag1*(const.g*const.g*const.g*const.g)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/(2.*((MW*MW - u)*(MW*MW - u))) + (4*Ca*CCflag1*Cih*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Cih*Cv*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Da*Dih*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((mzprime*mzprime - t)*(MW*MW - u)) + (4*CCflag1*Dih*Dv*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((mzprime*mzprime - t)*(MW*MW - u)) + (CCflag2*CCflag2*(const.g*const.g*const.g*const.g)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/(2.*((-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u))) + (CCflag2*CCflag2*(const.g*const.g*const.g*const.g)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/(2.*((-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u))) + (32*Ca*CCflag2*Cih*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (32*CCflag2*Cih*Cv*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*Ca*CCflag2*Cih*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Cih*Cv*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (32*CCflag2*Da*Dih*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (32*CCflag2*Dih*Dv*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Da*Dih*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Dih*Dv*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*Ca*CCflag2*Cih*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Cih*Cv*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Da*Dih*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Dih*Dv*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (2*CCflag1*CCflag2*(const.g*const.g*const.g*const.g)*mf*mh*(-(mm*mm) - mp*mp + t))/((MW*MW - u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*Ca*CCflag2*Cih*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Cih*Cv*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Da*Dih*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Dih*Dv*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*Ca*CCflag2*Cih*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Cih*Cv*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Da*Dih*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Dih*Dv*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (CCflag1*CCflag1*(const.g*const.g*const.g*const.g)*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MW*MW - u)*(MW*MW - u)) + (8*Ca*CCflag1*Cih*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Cih*Cv*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Da*Dih*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((mzprime*mzprime - t)*(MW*MW - u)) + (8*CCflag1*Dih*Dv*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((mzprime*mzprime - t)*(MW*MW - u)) - (CCflag2*CCflag2*(const.g*const.g*const.g*const.g)*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*Ca*CCflag2*Cih*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Cih*Cv*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Da*Dih*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Dih*Dv*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u))))/(512.*mh*(pi*pi*pi)*((t + u)*(t + u)))
#         return -(CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*(x2*x2))/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) + (CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*x24)/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) + (CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*(x2*x2)*x24)/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) - (CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*(x24*x24))/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) - (CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*(x2*x2)*(x3*x3))/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) + (CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*x24*(x3*x3))/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) - (CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*(x4*x4))/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) + (CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*x24*(x4*x4))/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) - (CCflag2*CCflag2*(gweak*gweak*gweak*gweak)*m1*(x3*x3)*(x4*x4))/(512.*(pi*pi*pi)*((x23 - xWBOSON*xWBOSON)*(x23 - xWBOSON*xWBOSON))) - (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*(x2*x2))/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) + (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*x23)/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) + (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*(x2*x2)*x23)/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) - (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*(x23*x23))/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) - (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*(x3*x3))/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) + (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*x23*(x3*x3))/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) - (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*(x2*x2)*(x4*x4))/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) + (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*x23*(x4*x4))/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) - (CCflag1*CCflag1*(gweak*gweak*gweak*gweak)*m1*(x3*x3)*(x4*x4))/(512.*(pi*pi*pi)*((x24 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON))) - (CCflag1*CCflag2*(gweak*gweak*gweak*gweak)*m1*x2)/(256.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON)) - (CCflag1*CCflag2*(gweak*gweak*gweak*gweak)*m1*(x2*x2*x2))/(256.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON)) + (CCflag1*CCflag2*(gweak*gweak*gweak*gweak)*m1*x2*x23)/(256.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON)) + (CCflag1*CCflag2*(gweak*gweak*gweak*gweak)*m1*x2*x24)/(256.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(x24 - xWBOSON*xWBOSON)) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x2)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x2)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x2*x2))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x2*x2))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x2*x2*x2))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x2*x2*x2))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x2*x23)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x2*x23)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x2*x2)*x23)/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x2*x2)*x23)/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x2*x24)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x2*x24)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x2*x2)*x24)/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x2*x2)*x24)/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x3*x3))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x3*x3))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x2*x2)*(x3*x3))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x2*x2)*(x3*x3))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x2*x3*x4)/(8.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x2*x3*x4)/(8.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x3*x3*x3)*x4)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x3*x3*x3)*x4)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x4*x4))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x4*x4))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x2*x2)*(x4*x4))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x2*x2)*(x4*x4))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x3*x3)*(x4*x4))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x3*x3)*(x4*x4))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x3*(x4*x4*x4))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x3*(x4*x4*x4))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x23)/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x23)/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x23*x23))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x23*x23))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x24)/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x24)/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*(x24*x24))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) - (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*(x24*x24))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x23*(x3*x3))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x23*(x3*x3))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x24*(x3*x3))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x24*(x3*x3))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x23*x3*x4)/(32.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x23*x3*x4)/(32.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) - (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x24*x3*x4)/(32.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x24*x3*x4)/(32.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x23*(x4*x4))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x23*(x4*x4))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Ca*Ca*(Cih*Cih)*m1*(NCflag*NCflag)*x24*(x4*x4))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Cih*Cih*(Cv*Cv)*m1*(NCflag*NCflag)*x24*(x4*x4))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON))) + (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x2)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x2)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*(x2*x2))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*(x2*x2))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*(x2*x2*x2))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*(x2*x2*x2))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x2*x23)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x2*x23)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x2*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x2*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*(x2*x2)*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*(x2*x2)*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*(x24*x24))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*(x24*x24))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*(x2*x2)*(x3*x3))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*(x2*x2)*(x3*x3))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x24*(x3*x3))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x24*(x3*x3))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x2*x3*x4)/(32.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x2*x3*x4)/(32.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x23*x3*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x23*x3*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x24*x3*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x24*x3*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*(x3*x3*x3)*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*(x3*x3*x3)*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x24*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x24*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*(x3*x3)*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*(x3*x3)*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag2*Cih*(gweak*gweak)*m1*NCflag*x3*(x4*x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag2*Cih*Cv*(gweak*gweak)*m1*NCflag*x3*(x4*x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x2)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x2)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*(x2*x2))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*(x2*x2))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*(x2*x2*x2))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*(x2*x2*x2))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x2*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x2*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*(x2*x2)*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*(x2*x2)*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*(x23*x23))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*(x23*x23))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x2*x24)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x2*x24)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*(x3*x3))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*(x3*x3))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x23*(x3*x3))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x23*(x3*x3))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x2*x3*x4)/(32.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x2*x3*x4)/(32.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x23*x3*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x23*x3*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x24*x3*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x24*x3*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*(x3*x3*x3)*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*(x3*x3*x3)*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*(x2*x2)*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*(x2*x2)*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x23*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x23*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*(x3*x3)*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*(x3*x3)*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) - (Ca*CCflag1*Cih*(gweak*gweak)*m1*NCflag*x3*(x4*x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (CCflag1*Cih*Cv*(gweak*gweak)*m1*NCflag*x3*(x4*x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x2)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x2)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x2*x2))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x2*x2))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x2*x2*x2))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x2*x2*x2))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x2*x23)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x2*x23)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x2*x2)*x23)/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x2*x2)*x23)/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x2*x24)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x2*x24)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x2*x2)*x24)/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x2*x2)*x24)/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x3*x3))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x3*x3))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x2*x2)*(x3*x3))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x2*x2)*(x3*x3))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x2*x3*x4)/(8.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x2*x3*x4)/(8.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x3*x3*x3)*x4)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x3*x3*x3)*x4)/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x4*x4))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x4*x4))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x2*x2)*(x4*x4))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x2*x2)*(x4*x4))/(64.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x3*x3)*(x4*x4))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x3*x3)*(x4*x4))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x3*(x4*x4*x4))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x3*(x4*x4*x4))/(32.*(pi*pi*pi)*((1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME))) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x2)/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x2)/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x2*x2))/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x2*x2))/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x2*x2*x2))/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x2*x2*x2))/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x2*x23)/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x2*x23)/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x2*x2)*x23)/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x2*x2)*x23)/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x2*x24)/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x2*x24)/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x2*x2)*x24)/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x2*x2)*x24)/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x3*x3))/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x3*x3))/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x2*x2)*(x3*x3))/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x2*x2)*(x3*x3))/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x2*x3*x4)/(4.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x2*x3*x4)/(4.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x3*x3*x3)*x4)/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x3*x3*x3)*x4)/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x4*x4))/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x4*x4))/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x2*x2)*(x4*x4))/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x2*x2)*(x4*x4))/(32.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x3*x3)*(x4*x4))/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x3*x3)*(x4*x4))/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x3*(x4*x4*x4))/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x3*(x4*x4*x4))/(16.*(pi*pi*pi)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZBOSON*xZBOSON)*(1 + x2*x2 - x23 - x24 + x3*x3 + x4*x4 - xZPRIME*xZPRIME)) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x23)/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x23)/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x23*x23))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x23*x23))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x24)/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x24)/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*(x24*x24))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) - (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*(x24*x24))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x23*(x3*x3))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x23*(x3*x3))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x24*(x3*x3))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x24*(x3*x3))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x23*x3*x4)/(32.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x23*x3*x4)/(32.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) - (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x24*x3*x4)/(32.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x24*x3*x4)/(32.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x23*(x4*x4))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x23*(x4*x4))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Da*Da*(Dih*Dih)*m1*(NCflag*NCflag)*x24*(x4*x4))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (Dih*Dih*(Dv*Dv)*m1*(NCflag*NCflag)*x24*(x4*x4))/(64.*(pi*pi*pi)*((-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))) + (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x2)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x2)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*(x2*x2))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*(x2*x2))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*(x2*x2*x2))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*(x2*x2*x2))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x2*x23)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x2*x23)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x2*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x2*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*(x2*x2)*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*(x2*x2)*x24)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*(x24*x24))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*(x24*x24))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*(x2*x2)*(x3*x3))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*(x2*x2)*(x3*x3))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x24*(x3*x3))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x24*(x3*x3))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x2*x3*x4)/(32.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x2*x3*x4)/(32.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x23*x3*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x23*x3*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x24*x3*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x24*x3*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*(x3*x3*x3)*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*(x3*x3*x3)*x4)/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x24*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x24*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*(x3*x3)*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*(x3*x3)*(x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag2*Da*Dih*(gweak*gweak)*m1*NCflag*x3*(x4*x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag2*Dih*Dv*(gweak*gweak)*m1*NCflag*x3*(x4*x4*x4))/(128.*(pi*pi*pi)*(x23 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x2)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x2)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*(x2*x2))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*(x2*x2))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*(x2*x2*x2))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*(x2*x2*x2))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x2*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x2*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*(x2*x2)*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*(x2*x2)*x23)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*(x23*x23))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*(x23*x23))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x2*x24)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x2*x24)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*(x3*x3))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*(x3*x3))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x23*(x3*x3))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x23*(x3*x3))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x2*x3*x4)/(32.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x2*x3*x4)/(32.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x23*x3*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x23*x3*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x24*x3*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x24*x3*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*(x3*x3*x3)*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*(x3*x3*x3)*x4)/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*(x2*x2)*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*(x2*x2)*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x23*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x23*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*(x3*x3)*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*(x3*x3)*(x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (CCflag1*Da*Dih*(gweak*gweak)*m1*NCflag*x3*(x4*x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (CCflag1*Dih*Dv*(gweak*gweak)*m1*NCflag*x3*(x4*x4*x4))/(128.*(pi*pi*pi)*(x24 - xWBOSON*xWBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x23)/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x23)/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x23*x23))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x23*x23))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x24)/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x24)/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*(x24*x24))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*(x24*x24))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x23*(x3*x3))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x23*(x3*x3))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x24*(x3*x3))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x24*(x3*x3))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x23*x3*x4)/(16.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x23*x3*x4)/(16.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) - (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x24*x3*x4)/(16.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x24*x3*x4)/(16.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x23*(x4*x4))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x23*(x4*x4))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Ca*Cih*Da*Dih*m1*(NCflag*NCflag)*x24*(x4*x4))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME)) + (Cih*Cv*Dih*Dv*m1*(NCflag*NCflag)*x24*(x4*x4))/(32.*(pi*pi*pi)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZBOSON*xZBOSON)*(-1 - x2*x2 + x23 + x24 - x3*x3 - x4*x4 + xZPRIME*xZPRIME))

#     def Sqrt(x):
#         return np.sqrt(x)

#     x23min = lambda x24: x2*x2 + x3*x3 - ((-1 + x24 + x3*x3)*(x2*x2 + x24 - x4*x4))/(2.*x24) - (Sqrt((-1 + x24)*(-1 + x24) - 2*(1 + x24)*(x3*x3) + x3*x3*x3*x3)*Sqrt(x2*x2*x2*x2 + (x24 - x4*x4)*(x24 - x4*x4) - 2*(x2*x2)*(x24 + x4*x4)))/(2.*x24)
#     x23max = lambda x24: x2*x2 + x3*x3 - ((-1 + x24 + x3*x3)*(x2*x2 + x24 - x4*x4))/(2.*x24) + (Sqrt((-1 + x24)*(-1 + x24) - 2*(1 + x24)*(x3*x3) + x3*x3*x3*x3)*Sqrt(x2*x2*x2*x2 + (x24 - x4*x4)*(x24 - x4*x4) - 2*(x2*x2)*(x24 + x4*x4)))/(2.*x24)


#     integral, error = integrate.dblquad(  DGammaDuDt,
#                                                 (x2+x4)**2,
#                                                 (1-x3)**2,
#                                                 x23min,
#                                                 x23max,
#                                                 args=(mh,x2,x3,x4,NCflag,CCflag1,CCflag2,Cv,Ca,Dv,Da,Cih,Dih,MZBOSON,mzprime,MW),\
#                                                 epsabs=1.49e-08, epsrel=1.49e-08)

#     return integral


# ###############################
# # New containing all terms!
# def nuh_nui_nuj_nuk(params, initial_neutrino, final_neutrinoi, final_neutrinoj, final_neutrinok):

#     ################################
#     # COUPLINGS
#     # FIX ME
#     # NEED TO RE-DO THIS WHOLE NU6 SECTION...
#     if initial_neutrino==pdg.neutrino6:
#         mh = params.m6
#         # Which outgoing neutrino?
#         Ah = params.A6
#         Dh = params.D6

#         ##############
#         # NU5 ->  nu nu nu
#         if final_neutrinoi==const.nulight:

#             aBSM = Dh*(params.A4+params.A5)*(params.A4+params.A5)*(params.A4+params.A5)/6.0
#             bBSM = aBSM
#             cBSM = aBSM
#             dBSM = aBSM
#             eBSM = aBSM
#             fBSM = aBSM

#             aSM = Ah*(1-Ah)*(1+(1-Ah)*(1-Ah))/6.0
#             bSM = aSM
#             cSM = aSM
#             dSM = aSM
#             eSM = aSM
#             fSM = aSM

#             aINT = -2*Dh *(params.D4+params.D5)*(params.A4+params.A5)*(params.A4+params.A5)/6.0
#             bINT = aINT
#             cINT = aINT
#             dINT = aINT
#             eINT = aINT
#             fINT = aINT

#             x2 = 0.0
#             x3 = 0.0
#             x4 = 0.0

#             S=6

#         ##############
#         # NU5 ->  NU4 nu nu
#         elif (final_neutrinoj==const.nulight):

#             aSM = params.C45SQR*(2+(1-params.A4-params.A5)*(1-params.A4-params.A5) +2*params.C45SQR - 2*params.A4*params.A5)
#             bSM = (params.A4-params.A4*params.A4-params.A5*params.A5)*(params.A5-params.A4*params.A4-params.A5*params.A5)
#             cSM = bSM
#             dSM = params.C45SQR*((1-params.A4-params.A5)*(1-params.A4-params.A5)+params.C45SQR-params.A4*params.A5 )
#             eSM = dSM
#             fSM = params.C45SQR*(1-params.A4-params.A5)*(1-params.A4-params.A5)

#             aBSM = Dh*params.D4*(params.A4+params.A5)*(params.A4+params.A5)
#             bBSM = aBSM
#             cBSM = aBSM
#             dBSM = aBSM
#             eBSM = aBSM
#             fBSM = aBSM

#             ##########
#             # ADD INTERFERENCE WHEN YOU CAN
#             aINT = 0.0
#             bINT = aINT
#             cINT = aINT
#             dINT = aINT
#             eINT = aINT
#             fINT = aINT

#             x2 = params.m4/mh
#             x3 = 0.0
#             x4 = 0.0

#             S=2*6

#         ##############
#         # NU5 ->  NU4 NU4 nu
#         elif (final_neutrinok==const.nulight):

#             aSM = params.C45SQR*(params.A4-params.A4*params.A4-params.A5*params.A5)
#             bSM = aSM
#             cSM = params.A4*params.A4*(params.A5-params.A4*params.A4-params.A5*params.A5)
#             dSM = aSM
#             eSM = params.C45SQR*(1-params.A4-params.A5)
#             fSM = eSM

#             aBSM = Dh*params.D4*params.D4*(params.A4+params.A5)
#             bBSM = aBSM
#             cBSM = aBSM
#             dBSM = aBSM
#             eBSM = aBSM
#             fBSM = aBSM

#             ##########
#             # ADD INTERFERENCE WHEN YOU CAN
#             aINT = 0.0
#             bINT = aINT
#             cINT = aINT
#             dINT = aINT
#             eINT = aINT
#             fINT = aINT

#             x2 = params.m4/mh
#             x3 = params.m4/mh
#             x4 = 0.0

#             S=2*6

#         ##############
#         # NU5 ->  NU4 NU4 NU4
#         elif (final_neutrinok==pdg.neutrino4):

#             aSM = params.C45SQR*params.A4*params.A4/6.0
#             bSM = aSM
#             cSM = aSM
#             dSM = aSM
#             eSM = aSM
#             fSM = aSM

#             aBSM = Dh*params.D4*params.D4*params.D4/6.0
#             bBSM = aBSM
#             cBSM = aBSM
#             dBSM = aBSM
#             eBSM = aBSM
#             fBSM = aBSM

#             ##########
#             # ADD INTERFERENCE WHEN YOU CAN
#             aINT = 0.0
#             bINT = aINT
#             cINT = aINT
#             dINT = aINT
#             eINT = aINT
#             fINT = aINT

#             x2 = params.m4/mh
#             x3 = params.m4/mh
#             x4 = params.m4/mh

#             S=6

#     ################################
#     # COUPLINGS
#     elif initial_neutrino==pdg.neutrino5:
#         mh = params.m5
#         # Which outgoing neutrino?
#         Ah = params.A5
#         Dh = params.D5

#         ##############
#         # NU5 ->  nu nu nu
#         if final_neutrinoi==const.nulight:

#             aBSM = Dh*(params.A4+params.A5)*(params.A4+params.A5)*(params.A4+params.A5)/6.0
#             bBSM = aBSM
#             cBSM = aBSM
#             dBSM = aBSM
#             eBSM = aBSM
#             fBSM = aBSM

#             aSM = Ah*(1-Ah)*(1+(1-Ah)*(1-Ah))/6.0
#             bSM = aSM
#             cSM = aSM
#             dSM = aSM
#             eSM = aSM
#             fSM = aSM

#             aINT = -2*Dh *(params.D4+params.D5)*(params.A4+params.A5)*(params.A4+params.A5)/6.0
#             bINT = aINT
#             cINT = aINT
#             dINT = aINT
#             eINT = aINT
#             fINT = aINT

#             x2 = 0.0
#             x3 = 0.0
#             x4 = 0.0

#             S=6

#         ##############
#         # NU5 ->  NU4 nu nu
#         elif (final_neutrinoj==const.nulight):

#             aSM = params.C45SQR*(2+(1-params.A4-params.A5)*(1-params.A4-params.A5) +2*params.C45SQR - 2*params.A4*params.A5)
#             bSM = (params.A4-params.A4*params.A4-params.A5*params.A5)*(params.A5-params.A4*params.A4-params.A5*params.A5)
#             cSM = bSM
#             dSM = params.C45SQR*((1-params.A4-params.A5)*(1-params.A4-params.A5)+params.C45SQR-params.A4*params.A5 )
#             eSM = dSM
#             fSM = params.C45SQR*(1-params.A4-params.A5)*(1-params.A4-params.A5)

#             aBSM = Dh*params.D4*(params.A4+params.A5)*(params.A4+params.A5)
#             bBSM = aBSM
#             cBSM = aBSM
#             dBSM = aBSM
#             eBSM = aBSM
#             fBSM = aBSM

#             ##########
#             # ADD INTERFERENCE WHEN YOU CAN
#             aINT = 0.0
#             bINT = aINT
#             cINT = aINT
#             dINT = aINT
#             eINT = aINT
#             fINT = aINT

#             x2 = params.m4/mh
#             x3 = 0.0
#             x4 = 0.0

#             S=2*6

#         ##############
#         # NU5 ->  NU4 NU4 nu
#         elif (final_neutrinok==const.nulight):

#             aSM = params.C45SQR*(params.A4-params.A4*params.A4-params.A5*params.A5)
#             bSM = aSM
#             cSM = params.A4*params.A4*(params.A5-params.A4*params.A4-params.A5*params.A5)
#             dSM = aSM
#             eSM = params.C45SQR*(1-params.A4-params.A5)
#             fSM = eSM

#             aBSM = Dh*params.D4*params.D4*(params.A4+params.A5)
#             bBSM = aBSM
#             cBSM = aBSM
#             dBSM = aBSM
#             eBSM = aBSM
#             fBSM = aBSM

#             ##########
#             # ADD INTERFERENCE WHEN YOU CAN
#             aINT = 0.0
#             bINT = aINT
#             cINT = aINT
#             dINT = aINT
#             eINT = aINT
#             fINT = aINT

#             x2 = params.m4/mh
#             x3 = params.m4/mh
#             x4 = 0.0

#             S=2*6

#         ##############
#         # NU5 ->  NU4 NU4 NU4
#         elif (final_neutrinok==pdg.neutrino4):

#             aSM = params.C45SQR*params.A4*params.A4/6.0
#             bSM = aSM
#             cSM = aSM
#             dSM = aSM
#             eSM = aSM
#             fSM = aSM

#             aBSM = Dh*params.D4*params.D4*params.D4/6.0
#             bBSM = aBSM
#             cBSM = aBSM
#             dBSM = aBSM
#             eBSM = aBSM
#             fBSM = aBSM

#             ##########
#             # ADD INTERFERENCE WHEN YOU CAN
#             aINT = 0.0
#             bINT = aINT
#             cINT = aINT
#             dINT = aINT
#             eINT = aINT
#             fINT = aINT

#             x2 = params.m4/mh
#             x3 = params.m4/mh
#             x4 = params.m4/mh

#             S=6


#     elif initial_neutrino==pdg.neutrino4:
#         mh = params.m4
#         # Which outgoing neutrino?
#         Ah = params.A4
#         Dh = params.D4

#         ##############
#         # NU4 ->  nu nu nu
#         if final_neutrinoi==const.nulight:

#             aBSM = Dh*(params.A4+params.A5)*(params.A4+params.A5)*(params.A4+params.A5)/6.0
#             bBSM = aBSM
#             cBSM = aBSM
#             dBSM = aBSM
#             eBSM = aBSM
#             fBSM = aBSM

#             aSM = Ah*(1-Ah)*(1+(1-Ah)*(1-Ah))/6.0
#             bSM = aSM
#             cSM = aSM
#             dSM = aSM
#             eSM = aSM
#             fSM = aSM

#             aINT = -2*Dh *(params.D4+params.D5)*(params.A4+params.A5)*(params.A4+params.A5)/6.0
#             bINT = aINT
#             cINT = aINT
#             dINT = aINT
#             eINT = aINT
#             fINT = aINT

#             x2 = 0.0
#             x3 = 0.0
#             x4 = 0.0

#             S=6

#     # couplings and masses  LEADING ORDER!
#     MZBOSON = const.Mz
#     mzprime = params.mzprime
#     MW = const.Mw
#     cw = const.cw
#     xZBOSON=MZBOSON/mh
#     xZPRIME=mzprime/mh
#     xWBOSON=MW/mh
#     m1=mh

#     gweak=const.gweak
#     gD=params.gD

#     pi = np.pi

#     def DGammaDuDt(x23,x24,m1,x2,x3,x4):
#         # return (u*((-256*(Ca*Ca)*(Cih*Cih)*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (256*(Cih*Cih)*(Cv*Cv)*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) - (64*(Ca*Ca)*(Cih*Cih)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (64*(Cih*Cih)*(Cv*Cv)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) - (256*(Da*Da)*(Dih*Dih)*mf*mh*mm*mp*(NCflag*NCflag))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) + (256*(Dih*Dih)*(Dv*Dv)*mf*mh*mm*mp*(NCflag*NCflag))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) - (64*(Da*Da)*(Dih*Dih)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) + (64*(Dih*Dih)*(Dv*Dv)*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) + (64*(Ca*Ca)*(Cih*Cih)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (64*(Cih*Cih)*(Cv*Cv)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (64*(Da*Da)*(Dih*Dih)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) + (64*(Dih*Dih)*(Dv*Dv)*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) + (512*Ca*Cih*Da*Dih*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) - (512*Cih*Cv*Dih*Dv*mf*mh*mm*mp*(NCflag*NCflag))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) + (128*Ca*Cih*Da*Dih*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) - (128*Cih*Cv*Dih*Dv*mm*mp*(NCflag*NCflag)*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) - (128*Ca*Cih*Da*Dih*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) - (128*Cih*Cv*Dih*Dv*mf*mh*(NCflag*NCflag)*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) - (32*Ca*CCflag1*Cih*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (32*CCflag1*Cih*Cv*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(MW*MW - u)) - (8*Ca*CCflag1*Cih*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Cih*Cv*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) - (32*CCflag1*Da*Dih*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((mzprime*mzprime - t)*(MW*MW - u)) + (32*CCflag1*Dih*Dv*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((mzprime*mzprime - t)*(MW*MW - u)) - (8*CCflag1*Da*Dih*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((mzprime*mzprime - t)*(MW*MW - u)) + (8*CCflag1*Dih*Dv*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((mzprime*mzprime - t)*(MW*MW - u)) + (8*Ca*CCflag1*Cih*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Cih*Cv*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Da*Dih*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((mzprime*mzprime - t)*(MW*MW - u)) + (8*CCflag1*Dih*Dv*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((mzprime*mzprime - t)*(MW*MW - u)) + (32*(Ca*Ca)*(Cih*Cih)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Cih*Cih)*(Cv*Cv)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Da*Da)*(Dih*Dih)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) + (32*(Dih*Dih)*(Dv*Dv)*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) - (64*Ca*Cih*Da*Dih*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) - (64*Cih*Cv*Dih*Dv*(NCflag*NCflag)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) + (CCflag1*CCflag1*(const.g*const.g*const.g*const.g)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/(2.*((MW*MW - u)*(MW*MW - u))) + (4*Ca*CCflag1*Cih*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Cih*Cv*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Da*Dih*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((mzprime*mzprime - t)*(MW*MW - u)) + (4*CCflag1*Dih*Dv*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((mzprime*mzprime - t)*(MW*MW - u)) + (32*(Ca*Ca)*(Cih*Cih)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Cih*Cih)*(Cv*Cv)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)) + (32*(Da*Da)*(Dih*Dih)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) + (32*(Dih*Dih)*(Dv*Dv)*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((mzprime*mzprime - t)*(mzprime*mzprime - t)) - (64*Ca*Cih*Da*Dih*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) - (64*Cih*Cv*Dih*Dv*(NCflag*NCflag)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)) + (CCflag1*CCflag1*(const.g*const.g*const.g*const.g)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/(2.*((MW*MW - u)*(MW*MW - u))) + (4*Ca*CCflag1*Cih*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Cih*Cv*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (4*CCflag1*Da*Dih*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((mzprime*mzprime - t)*(MW*MW - u)) + (4*CCflag1*Dih*Dv*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((mzprime*mzprime - t)*(MW*MW - u)) + (CCflag2*CCflag2*(const.g*const.g*const.g*const.g)*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/(2.*((-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u))) + (CCflag2*CCflag2*(const.g*const.g*const.g*const.g)*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/(2.*((-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u))) + (32*Ca*CCflag2*Cih*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (32*CCflag2*Cih*Cv*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*Ca*CCflag2*Cih*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Cih*Cv*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (32*CCflag2*Da*Dih*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (32*CCflag2*Dih*Dv*(const.g*const.g)*mf*mh*mm*mp*NCflag)/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Da*Dih*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Dih*Dv*(const.g*const.g)*mm*mp*NCflag*(mf*mf + mh*mh - t))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*Ca*CCflag2*Cih*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Cih*Cv*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Da*Dih*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (8*CCflag2*Dih*Dv*(const.g*const.g)*mf*mh*NCflag*(-(mm*mm) - mp*mp + t))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (2*CCflag1*CCflag2*(const.g*const.g*const.g*const.g)*mf*mh*(-(mm*mm) - mp*mp + t))/((MW*MW - u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*Ca*CCflag2*Cih*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Cih*Cv*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Da*Dih*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Dih*Dv*(const.g*const.g)*NCflag*(mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*Ca*CCflag2*Cih*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Cih*Cv*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Da*Dih*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) - (4*CCflag2*Dih*Dv*(const.g*const.g)*NCflag*(mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (CCflag1*CCflag1*(const.g*const.g*const.g*const.g)*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MW*MW - u)*(MW*MW - u)) + (8*Ca*CCflag1*Cih*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Cih*Cv*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(MW*MW - u)) + (8*CCflag1*Da*Dih*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((mzprime*mzprime - t)*(MW*MW - u)) + (8*CCflag1*Dih*Dv*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((mzprime*mzprime - t)*(MW*MW - u)) - (CCflag2*CCflag2*(const.g*const.g*const.g*const.g)*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*Ca*CCflag2*Cih*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Cih*Cv*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((MZBOSON*MZBOSON - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Da*Dih*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u)) + (8*CCflag2*Dih*Dv*(const.g*const.g)*NCflag*(-((mh*mh + mm*mm - u)*(-(mf*mf) - mp*mp + u))/2. + ((mh*mh + mp*mp - t - u)*(-(mf*mf) - mm*mm + t + u))/2.))/((mzprime*mzprime - t)*(-(mf*mf) - mh*mh - mm*mm - mp*mp + MW*MW + t + u))))/(512.*mh*(pi*pi*pi)*((t + u)*(t + u)))
#         x34=1.0-x23-x24+x2*x2+x3*x3+x4*x4
#         return (gweak*gweak*gweak*gweak*m1*(-((cSM*(x24*x24 + x3*x3 - x34 - x3*x3*x34 + x34*x34 - 2*x23*x4 + 2*(x3*x3)*x4 + 2*(x4*x4) + x3*x3*(x4*x4) - x34*(x4*x4) - x24*(1 + x3*x3 + x4*x4) + x2*x2*(1 - x24 + 2*(x3*x3) - x34 + 2*x4 + x4*x4) + 2*x2*x3*(1 - x23 + 4*x4 + x4*x4)))/((x23 - xZBOSON*xZBOSON)*(x23 - xZBOSON*xZBOSON))) - (bSM*(x23*x23 - 2*x24*x3 + 2*(x3*x3) - x34 - x3*x3*x34 + x34*x34 + 2*x2*(1 - x24 + 4*x3 + x3*x3)*x4 + x4*x4 + 2*x3*(x4*x4) + x3*x3*(x4*x4) - x34*(x4*x4) - x23*(1 + x3*x3 + x4*x4) + x2*x2*(1 - x23 + 2*x3 + x3*x3 - x34 + 2*(x4*x4))))/((x24 - xZBOSON*xZBOSON)*(x24 - xZBOSON*xZBOSON)) - (aSM*(x23*x23 - x24 + x24*x24 + x3*x3 - x24*(x3*x3) + 2*x3*x4 - 2*x3*x34*x4 + x4*x4 - x24*(x4*x4) + 2*(x3*x3)*(x4*x4) - x23*(1 + x3*x3 + x4*x4) + x2*x2*(2 - x23 - x24 + x3*x3 + 2*x3*x4 + x4*x4) + 2*x2*(x3*x3 - x34 + 4*x3*x4 + x4*x4)))/((x34 - xZBOSON*xZBOSON)*(x34 - xZBOSON*xZBOSON)) + (2*fSM*(-(x24*x3) + x3*x3 - x34 - x3*x3*x34 + x34*x34 - x23*x4 + x3*x4 + x3*x3*x4 - x3*x34*x4 + x4*x4 + x3*(x4*x4) - x34*(x4*x4) + x2*x2*(x3 + x3*x3 - x34 + x4 + x3*x4 + x4*x4) + x2*(-x34 + x4 - x24*x4 + x4*x4 + x3*x3*(1 + x4) + x3*(1 - x23 + 4*x4 + x4*x4))))/((x23 - xZBOSON*xZBOSON)*(-x24 + xZBOSON*xZBOSON)) + (2*eSM*(x24*x24 + x2*x2*(1 - x24 + x3 + x3*x3 + x4 + x3*x4) - x24*(1 + x3 + x3*x3 + x4*x4) + x4*(-x23 + x4 + x3*x3*(1 + x4) + x3*(1 - x34 + x4)) + x2*(-x34 + x3*x3*(1 + x4) + x4*(1 - x24 + x4) + x3*(1 - x23 + 4*x4 + x4*x4))))/((x23 - xZBOSON*xZBOSON)*(-x34 + xZBOSON*xZBOSON)) + (2*dSM*(x23*x23 - x23*(1 + x3*x3 + x4 + x4*x4) + x2*x2*(1 - x23 + x3 + x4 + x3*x4 + x4*x4) + x3*(-x24 + x4*(1 - x34 + x4) + x3*(1 + x4 + x4*x4)) + x2*(-x34 + x3*x3*(1 + x4) + x4*(1 - x24 + x4) + x3*(1 - x23 + 4*x4 + x4*x4))))/((x24 - xZBOSON*xZBOSON)*(-x34 + xZBOSON*xZBOSON))))/(1024.*(cw*cw*cw*cw)*(pi*pi*pi)) + (gD*gD*gD*gD*m1*(-((cBSM*(x24*x24 + x3*x3 - x34 - x3*x3*x34 + x34*x34 - 2*x23*x4 + 2*(x3*x3)*x4 + 2*(x4*x4) + x3*x3*(x4*x4) - x34*(x4*x4) - x24*(1 + x3*x3 + x4*x4) + x2*x2*(1 - x24 + 2*(x3*x3) - x34 + 2*x4 + x4*x4) + 2*x2*x3*(1 - x23 + 4*x4 + x4*x4)))/((x23 - xZPRIME*xZPRIME)*(x23 - xZPRIME*xZPRIME))) - (bBSM*(x23*x23 - 2*x24*x3 + 2*(x3*x3) - x34 - x3*x3*x34 + x34*x34 + 2*x2*(1 - x24 + 4*x3 + x3*x3)*x4 + x4*x4 + 2*x3*(x4*x4) + x3*x3*(x4*x4) - x34*(x4*x4) - x23*(1 + x3*x3 + x4*x4) + x2*x2*(1 - x23 + 2*x3 + x3*x3 - x34 + 2*(x4*x4))))/((x24 - xZPRIME*xZPRIME)*(x24 - xZPRIME*xZPRIME)) - (aBSM*(x23*x23 - x24 + x24*x24 + x3*x3 - x24*(x3*x3) + 2*x3*x4 - 2*x3*x34*x4 + x4*x4 - x24*(x4*x4) + 2*(x3*x3)*(x4*x4) - x23*(1 + x3*x3 + x4*x4) + x2*x2*(2 - x23 - x24 + x3*x3 + 2*x3*x4 + x4*x4) + 2*x2*(x3*x3 - x34 + 4*x3*x4 + x4*x4)))/((x34 - xZPRIME*xZPRIME)*(x34 - xZPRIME*xZPRIME)) + (2*fBSM*(-(x24*x3) + x3*x3 - x34 - x3*x3*x34 + x34*x34 - x23*x4 + x3*x4 + x3*x3*x4 - x3*x34*x4 + x4*x4 + x3*(x4*x4) - x34*(x4*x4) + x2*x2*(x3 + x3*x3 - x34 + x4 + x3*x4 + x4*x4) + x2*(-x34 + x4 - x24*x4 + x4*x4 + x3*x3*(1 + x4) + x3*(1 - x23 + 4*x4 + x4*x4))))/((x23 - xZPRIME*xZPRIME)*(-x24 + xZPRIME*xZPRIME)) + (2*eBSM*(x24*x24 + x2*x2*(1 - x24 + x3 + x3*x3 + x4 + x3*x4) - x24*(1 + x3 + x3*x3 + x4*x4) + x4*(-x23 + x4 + x3*x3*(1 + x4) + x3*(1 - x34 + x4)) + x2*(-x34 + x3*x3*(1 + x4) + x4*(1 - x24 + x4) + x3*(1 - x23 + 4*x4 + x4*x4))))/((x23 - xZPRIME*xZPRIME)*(-x34 + xZPRIME*xZPRIME)) + (2*dBSM*(x23*x23 - x23*(1 + x3*x3 + x4 + x4*x4) + x2*x2*(1 - x23 + x3 + x4 + x3*x4 + x4*x4) + x3*(-x24 + x4*(1 - x34 + x4) + x3*(1 + x4 + x4*x4)) + x2*(-x34 + x3*x3*(1 + x4) + x4*(1 - x24 + x4) + x3*(1 - x23 + 4*x4 + x4*x4))))/((x24 - xZPRIME*xZPRIME)*(-x34 + xZPRIME*xZPRIME))))/(64.*(pi*pi*pi)) - (gD*gD*(gweak*gweak)*m1*(aINT*(x23*x23 - x24 + x24*x24 + x3*x3 - x24*(x3*x3) + 2*x3*x4 - 2*x3*x34*x4 + x4*x4 - x24*(x4*x4) + 2*(x3*x3)*(x4*x4) - x23*(1 + x3*x3 + x4*x4) + x2*x2*(2 - x23 - x24 + x3*x3 + 2*x3*x4 + x4*x4) + 2*x2*(x3*x3 - x34 + 4*x3*x4 + x4*x4))*(x23 - xZBOSON*xZBOSON)*(x24 - xZBOSON*xZBOSON)*(x23 - xZPRIME*xZPRIME)*(x24 - xZPRIME*xZPRIME) - bINT*(-(x23*x23) + 2*x24*x3 - 2*(x3*x3) + x34 + x3*x3*x34 - x34*x34 + 2*x2*(-1 + x24 - 4*x3 - x3*x3)*x4 - x4*x4 - 2*x3*(x4*x4) - x3*x3*(x4*x4) + x34*(x4*x4) + x2*x2*(-1 + x23 - 2*x3 - x3*x3 + x34 - 2*(x4*x4)) + x23*(1 + x3*x3 + x4*x4))*(x23 - xZBOSON*xZBOSON)*(x34 - xZBOSON*xZBOSON)*(x23 - xZPRIME*xZPRIME)*(x34 - xZPRIME*xZPRIME) - cINT*(-(x24*x24) - x3*x3 + x34 + x3*x3*x34 - x34*x34 + 2*x23*x4 - 2*(x3*x3)*x4 - 2*(x4*x4) - x3*x3*(x4*x4) + x34*(x4*x4) + 2*x2*x3*(-1 + x23 - 4*x4 - x4*x4) + x2*x2*(-1 + x24 - 2*(x3*x3) + x34 - 2*x4 - x4*x4) + x24*(1 + x3*x3 + x4*x4))*(x24 - xZBOSON*xZBOSON)*(x34 - xZBOSON*xZBOSON)*(x24 - xZPRIME*xZPRIME)*(x34 - xZPRIME*xZPRIME) - fINT*(-(x24*x3) + x3*x3 - x34 - x3*x3*x34 + x34*x34 - x23*x4 + x3*x4 + x3*x3*x4 - x3*x34*x4 + x4*x4 + x3*(x4*x4) - x34*(x4*x4) + x2*x2*(x3 + x3*x3 - x34 + x4 + x3*x4 + x4*x4) + x2*(-x34 + x4 - x24*x4 + x4*x4 + x3*x3*(1 + x4) + x3*(1 - x23 + 4*x4 + x4*x4)))*(x34 - xZBOSON*xZBOSON)*(x34 - xZPRIME*xZPRIME)*(-2*(xZBOSON*xZBOSON)*(xZPRIME*xZPRIME) + x24*(xZBOSON*xZBOSON + xZPRIME*xZPRIME) + x23*(-2*x24 + xZBOSON*xZBOSON + xZPRIME*xZPRIME)) - eINT*(x24*x24 + x2*x2*(1 - x24 + x3 + x3*x3 + x4 + x3*x4) - x24*(1 + x3 + x3*x3 + x4*x4) + x4*(-x23 + x4 + x3*x3*(1 + x4) + x3*(1 - x34 + x4)) + x2*(-x34 + x3*x3*(1 + x4) + x4*(1 - x24 + x4) + x3*(1 - x23 + 4*x4 + x4*x4)))*(x24*x24 + xZBOSON*xZBOSON*(xZPRIME*xZPRIME) - x24*(xZBOSON*xZBOSON + xZPRIME*xZPRIME))*(-2*(xZBOSON*xZBOSON)*(xZPRIME*xZPRIME) + x34*(xZBOSON*xZBOSON + xZPRIME*xZPRIME) + x23*(-2*x34 + xZBOSON*xZBOSON + xZPRIME*xZPRIME)) - dINT*(x23*x23 - x23*(1 + x3*x3 + x4 + x4*x4) + x2*x2*(1 - x23 + x3 + x4 + x3*x4 + x4*x4) + x3*(-x24 + x4*(1 - x34 + x4) + x3*(1 + x4 + x4*x4)) + x2*(-x34 + x3*x3*(1 + x4) + x4*(1 - x24 + x4) + x3*(1 - x23 + 4*x4 + x4*x4)))*(x23*x23 + xZBOSON*xZBOSON*(xZPRIME*xZPRIME) - x23*(xZBOSON*xZBOSON + xZPRIME*xZPRIME))*(-2*(xZBOSON*xZBOSON)*(xZPRIME*xZPRIME) + x34*(xZBOSON*xZBOSON + xZPRIME*xZPRIME) + x24*(-2*x34 + xZBOSON*xZBOSON + xZPRIME*xZPRIME))))/(128.*(cw*cw)*(pi*pi*pi)*(x23 - xZBOSON*xZBOSON)*(-x24 + xZBOSON*xZBOSON)*(-x34 + xZBOSON*xZBOSON)*(x23 - xZPRIME*xZPRIME)*(x24 - xZPRIME*xZPRIME)*(x34 - xZPRIME*xZPRIME))
#     def Sqrt(x):
#         return np.sqrt(x)

#     x23min = lambda x24: x2*x2 + x3*x3 - ((-1 + x24 + x3*x3)*(x2*x2 + x24 - x4*x4))/(2.*x24) - (Sqrt((-1 + x24)*(-1 + x24) - 2*(1 + x24)*(x3*x3) + x3*x3*x3*x3)*Sqrt(x2*x2*x2*x2 + (x24 - x4*x4)*(x24 - x4*x4) - 2*(x2*x2)*(x24 + x4*x4)))/(2.*x24)
#     x23max = lambda x24: x2*x2 + x3*x3 - ((-1 + x24 + x3*x3)*(x2*x2 + x24 - x4*x4))/(2.*x24) + (Sqrt((-1 + x24)*(-1 + x24) - 2*(1 + x24)*(x3*x3) + x3*x3*x3*x3)*Sqrt(x2*x2*x2*x2 + (x24 - x4*x4)*(x24 - x4*x4) - 2*(x2*x2)*(x24 + x4*x4)))/(2.*x24)


#     integral, error = dblquad(  DGammaDuDt,
#                                 (x2+x4)**2,
#                                 (1-x3)**2,
#                                 x23min,
#                                 x23max,
#                                 args=(mh,x2,x3,x4),\
#                                 epsabs=1.49e-08, epsrel=1.49e-08)

#     return integral/S

# def new_nuh_nui_nuj_nuk(params, initial_neutrino, final_neutrinoi, final_neutrinoj, final_neutrinok):

#     M   = initial_neutrino.mass
#     r_i = final_neutrinoi.mass / M
#     r_j = final_neutrinoj.mass / M
#     r_k = final_neutrinok.mass / M

#     i = int(final_neutrinoi.name[-1]) - 1
#     j = int(final_neutrinoj.name[-1]) - 1
#     k = int(final_neutrinok.name[-1]) - 1

#     # Z
#     C_ijk = 1/6*(\
#       F_3nu_decay(r_i,r_j,r_k) * np.abs(params.C_weak[h,i])**2 * np.abs(params.C_weak[j,k])**2
#     + F_3nu_decay(r_j,r_k,r_i) * np.abs(params.C_weak[h,j])**2 * np.abs(params.C_weak[i,k])**2
#     + F_3nu_decay(r_k,r_i,r_j) * np.abs(params.C_weak[h,k])**2 * np.abs(params.C_weak[j,i])**2
#     + G_3nu_decay(r_i,r_j,r_k) * (params.C_weak[h,i] * params.C_weak[i,k] * params.C_weak[k,j] * params.C_weak[j,h])
#     + G_3nu_decay(r_k,r_i,r_j) * (params.C_weak[h,i] * params.C_weak[i,j] * params.C_weak[j,k] * params.C_weak[k,h])
#     + G_3nu_decay(r_j,r_k,r_i) * (params.C_weak[h,j] * params.C_weak[j,i] * params.C_weak[i,k] * params.C_weak[k,h])
#     )

#     # Z'
#     D_ijk = 1/6*(\
#       F_3nu_decay(r_i,r_j,r_k) * np.abs(params.C_dark[h,i])**2 * np.abs(params.C_dark[j,k])**2
#     + F_3nu_decay(r_j,r_k,r_i) * np.abs(params.C_dark[h,j])**2 * np.abs(params.C_dark[i,k])**2
#     + F_3nu_decay(r_k,r_i,r_j) * np.abs(params.C_dark[h,k])**2 * np.abs(params.C_dark[j,i])**2
#     + G_3nu_decay(r_i,r_j,r_k) * (params.C_dark[h,i] * params.C_dark[i,k] * params.C_dark[k,j] * params.C_dark[j,h])
#     + G_3nu_decay(r_k,r_i,r_j) * (params.C_dark[h,i] * params.C_dark[i,j] * params.C_dark[j,k] * params.C_dark[k,h])
#     + G_3nu_decay(r_j,r_k,r_i) * (params.C_dark[h,j] * params.C_dark[j,i] * params.C_dark[i,k] * params.C_dark[k,h])
#     )

#     # Z-Z' interference
#     I_ijk=1/6*(\
#       F_3nu_decay(r_i,r_j,r_k) * (C_dark[h,i]*C_dark[j,k] * C_weak[k,j]*C_weak[i,h]).real
#     + F_3nu_decay(r_j,r_k,r_i) * (C_dark[h,j]*C_dark[i,k] * C_weak[k,i]*C_weak[j,h]).real
#     + F_3nu_decay(r_k,r_i,r_j) * (C_dark[h,k]*C_dark[j,i] * C_weak[i,j]*C_weak[k,h]).real
#     + G_3nu_decay(r_i,r_j,r_k) * (C_dark[h,i]*C_dark[i,k] * C_weak[k,j]*C_weak[j,h]).real
#     + G_3nu_decay(r_k,r_i,r_j) * (C_dark[h,i]*C_dark[i,j] * C_weak[j,k]*C_weak[k,h]).real
#     + G_3nu_decay(r_j,r_k,r_i) * (C_dark[h,j]*C_dark[j,i] * C_weak[i,k]*C_weak[k,h]).real
#     )

#     # symmetry factor
#     a = [i,j,k]
#     if len(np.unique(a)) == 3:
#         S = 1
#     elif len(np.unique(a)) == 2:
#         S = 1/2
#     elif len(np.unique(a)) == 1:
#         S = 1/6

#     return 1/192/np.pi**3*M**5 * S * (const.Gf**2*C_ijk + params.GX**2*D_ijk + const.Gf*params.GX*I_ijk)


# def nu4_to_nualpha_l_l(params, final_lepton):
#     if (final_lepton==const.tau):
#         m_ell = const.m_tau
#     elif(final_lepton==const.muon):
#         m_ell = const.m_mu
#     elif(final_lepton==const.electron):
#         m_ell = const.m_e
#     else:
#         print("WARNING! Unable to set charged lepton mass. Assuming massless.")
#         m_ell = 0

#     if (final_lepton==const.tau):
#         CC_mixing = params.Utau4
#     elif(final_lepton==const.muon):
#         CC_mixing = params.Umu4
#     elif(final_lepton==const.electron):
#         CC_mixing = params.Ue4
#     else:
#         print("WARNING! Unable to set CC mixing parameter for decay. Assuming 0.")
#         CC_mixing = 0

#     mi = params.m4
#     m0 = 0.0
#     def func(u,t):
#         gv = (const.g/const.cw)**2/2.0 *( params.cmu4*params.ceV/const.Mz**2 - params.dmu4*params.deV/(t-params.mzprime**2) ) \
#                         - const.g**2/4.0*CC_mixing/const.Mw**2
#         ga = (const.g/const.cw)**2/2.0 *(-params.cmu4*params.ceA/const.Mz**2 + params.dmu4*params.deA/(t-params.mzprime**2) ) \
#                         + const.g**2/4.0*CC_mixing/const.Mw**2
#         # print "gv, ga: ", gv, ga
#         return 4.0*((gv + ga)**2 *(mi**2 + m_ell**2 - u)*(u - m0**2 -m_ell**2)
#                         + (gv - ga)**2*(mi**2 - m0**2 - m_ell**2)*(mi**2 + m_ell**2 - mi**2)
#                             + (gv**2 - ga**2)*m_ell**2/2.0*(mi**2 + m0**2 - t))

#     uminus = lambda t: (mi**2 - m0**2)**2/4.0/t - t/4.0*((const.kallen_sqrt(1, mi**2/t, m0**2/t)) + (const.kallen_sqrt(1,m_ell**2/t, m_ell**2/t)))**2
#     uplus = lambda t: (mi**2 - m0**2)**2/4.0/t - t/4.0*((const.kallen_sqrt(1, mi**2/t, m0**2/t)) - (const.kallen_sqrt(1,m_ell**2/t, m_ell**2/t)))**2

#     integral, error = dblquad(  func,
#                                 (mi-m0)**2,
#                                 4*m_ell**2,
#                                 uplus,
#                                 uminus,
#                                 args=(), epsabs=1.49e-08, epsrel=1.49e-08)

#     return integral*1.0/(2.0*np.pi)**3 / 32.0 / mi**3
