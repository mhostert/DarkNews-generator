import numpy as np

# import numpy.ma as ma

from DarkNews import Cfourvec as Cfv
from DarkNews.const import rng_interval

import logging

logger = logging.getLogger("logger." + __name__)


########################################################
# Kinematical limits


# 2 --> 2 scattering
def upscattering_Q2max(Enu, mHNL, M):
    s = 2 * Enu * M + M**2
    return (
        1
        / 2
        * (s) ** (-1)
        * (
            (M) ** (4)
            + (
                -1 * (mHNL) ** (2) * s
                + (
                    (s) ** (2)
                    + (
                        -1 * (M) ** (2) * ((mHNL) ** (2) + 2 * s)
                        + (-1 * (M) ** (2) + s) * (((M) ** (4) + ((((mHNL) ** (2) + -1 * s)) ** (2) + -2 * (M) ** (2) * ((mHNL) ** (2) + s)))) ** (1 / 2)
                    )
                )
            )
        )
    )


# def upscattering_Q2min(Enu, mHNL, M):
#     s = 2 * Enu * M + M**2
#     r = mHNL / np.sqrt(s)
#     m = M / np.sqrt(s)

#     # large cancellations at play -- expanding for small r
#     q2min = ma.masked_array(
#         data=1
#         / 2
#         * (
#             1
#             + (
#                 (m) ** (4)
#                 + (
#                     -1 * (r) ** (2)
#                     + (
#                         -1 * ((((-1 + (m) ** (2))) ** (2) + (-2 * (1 + (m) ** (2)) * (r) ** (2) + (r) ** (4)))) ** (1 / 2)
#                         + (m) ** (2) * (-2 + (-1 * (r) ** (2) + ((((-1 + (m) ** (2))) ** (2) + (-2 * (1 + (m) ** (2)) * (r) ** (2) + (r) ** (4)))) ** (1 / 2)))
#                     )
#                 )
#             )
#         )
#         * s,
#         mask=(r < 1e-3),
#         fill_value=(m) ** (2) * ((-1 + (m) ** (2))) ** (-2) * (r) ** (4) * s,
#     )
#     return q2min.filled()


# q2min = np.piecewise(r, [r>=1e-3, r<1e-3], \
# 						[lambda r: 1/2 * ( 1 + ( ( m )**( 4 ) + ( -1 * ( r )**( 2 ) + ( -1 * ( ( ( ( -1 + ( m )**( 2 ) ) )**( 2 ) + ( -2 * ( 1 + ( m )**( 2 ) ) * ( r )**( 2 ) + ( r )**( 4 ) ) ) )**( 1/2 ) + ( m )**( 2 ) * ( -2 + ( -1 * ( r )**( 2 ) + ( ( ( ( -1 + ( m )**( 2 ) ) )**( 2 ) + ( -2 * ( 1 + ( m )**( 2 ) ) * ( r )**( 2 ) + ( r )**( 4 ) ) ) )**( 1/2 ) ) ) ) ) ) ) * s,\
# 						 lambda r: ( m )**( 2 ) * ( ( -1 + ( m )**( 2 ) ) )**( -2 ) * ( r )**( 4 ) * s])
# return q2min


def upscattering_Q2min_s(Enu, mHNL, M):
    s = 2 * Enu * M + M**2
    r = mHNL / np.sqrt(s)
    m = M / np.sqrt(s)

    # For small r values, expand using series
    if r < 1e-3:
        return (m) ** (2) * ((-1 + (m) ** (2))) ** (-2) * (r) ** (4) * s

    # Otherwise, compute the full expression
    q2min = (
        1
        / 2
        * (
            1
            + (
                (m) ** (4)
                + (
                    -1 * (r) ** (2)
                    + (
                        -1 * ((((-1 + (m) ** (2))) ** (2) + (-2 * (1 + (m) ** (2)) * (r) ** (2) + (r) ** (4)))) ** (1 / 2)
                        + (m) ** (2) * (-2 + (-1 * (r) ** (2) + ((((-1 + (m) ** (2))) ** (2) + (-2 * (1 + (m) ** (2)) * (r) ** (2) + (r) ** (4)))) ** (1 / 2)))
                    )
                )
            )
        )
        * s
    )

    return q2min


# Vectorize the function to handle arrays
upscattering_Q2min = np.vectorize(upscattering_Q2min_s)


# 1 --> 3 decays (decay mandelstam)
def three_body_umax(m1, m2, m3, m4, t):
    return 1 / 4 * (((m1) ** (2) + ((m2) ** (2) + (-1 * (m3) ** (2) + -1 * (m4) ** (2))))) ** (2) * (t) ** (-1) + -1 * (
        (
            ((-1 * (m2) ** (2) + 1 / 4 * (t) ** (-1) * (((m2) ** (2) + (-1 * (m3) ** (2) + t))) ** (2))) ** (1 / 2)
            + -1 * ((-1 * (m4) ** (2) + 1 / 4 * (t) ** (-1) * ((-1 * (m1) ** (2) + ((m4) ** (2) + t))) ** (2))) ** (1 / 2)
        )
    ) ** (2)


def three_body_umin(m1, m2, m3, m4, t):
    return 1 / 4 * (((m1) ** (2) + ((m2) ** (2) + (-1 * (m3) ** (2) + -1 * (m4) ** (2))))) ** (2) * (t) ** (-1) + -1 * (
        (
            ((-1 * (m2) ** (2) + 1 / 4 * (t) ** (-1) * (((m2) ** (2) + (-1 * (m3) ** (2) + t))) ** (2))) ** (1 / 2)
            + ((-1 * (m4) ** (2) + 1 / 4 * (t) ** (-1) * ((-1 * (m1) ** (2) + ((m4) ** (2) + t))) ** (2))) ** (1 / 2)
        )
    ) ** (2)


def three_body_tmax(m1, m2, m3, m4):
    return (m1 - m4) ** 2


def three_body_tmin(m1, m2, m3, m4):
    return (m2 + m3) ** 2


# ######################
# 2 -> 2 scattering
# Ni(k1) target(k2) -->  Nj(k3) target(k4)
def two_to_two_scatter(samples, m1=1.0, m2=0.0, m3=1.0, m4=0.0, rng=np.random.random):

    if "Eprojectile" in samples.keys():
        Eprojectile = samples["Eprojectile"]
        s = m2**2 + 2 * Eprojectile * m2 + m1**2
        sample_size = np.shape(Eprojectile)[0]
    else:
        logger.error("Error! Could not determine the projectile energy")

    E1CM = (s + m1**2 - m2**2) / 2.0 / np.sqrt(s)
    E2CM = (s - m1**2 + m2**2) / 2.0 / np.sqrt(s)
    E3CM = (s + m3**2 - m4**2) / 2.0 / np.sqrt(s)
    E4CM = (s - m3**2 + m4**2) / 2.0 / np.sqrt(s)

    p1CM = np.sqrt(E1CM**2 - m1**2)
    p2CM = np.sqrt(E2CM**2 - m2**2)
    p3CM = np.sqrt(E3CM**2 - m3**2)
    # p4CM = np.sqrt(E4CM**2 - m4**2)

    # if massless proj and elastic in one, watch out for cancellations
    if m1 == 0 and m2 == m4:
        Q2min = upscattering_Q2min(Enu=(s - m2**2) / 2 / m2, mHNL=m3, M=m2)
        Q2max = upscattering_Q2max(Enu=(s - m2**2) / 2 / m2, mHNL=m3, M=m2)
    else:
        # general case
        Q2min = -(m1**2 + m3**2 - 2 * (E1CM * E3CM - p1CM * p3CM))
        Q2max = -(m1**2 + m3**2 - 2 * (E1CM * E3CM + p1CM * p3CM))

    if "unit_Q2" in samples.keys():
        Q2l = (np.log(Q2max) - np.log(Q2min)) * samples["unit_Q2"] + np.log(Q2min)
        Q2 = np.exp(Q2l)
    elif "Q2" in samples.keys():
        Q2 = samples["Q2"]
    else:
        logger.debug("DEBUG: Could not find Q2 samples, using uniform distribution instead.")
        Q2 = rng_interval(sample_size, Q2min, Q2max, rng=rng)

    # KINEMATICS TO LAB FRAME
    costN = (-Q2 - m1**2 - m3**2 + 2 * E1CM * E3CM) / (2 * p1CM * p3CM)
    beta = -p2CM / E2CM  # MINUS SIGN -- from CM to LAB
    # gamma = 1.0 / np.sqrt(1.0 - beta * beta)

    if "unit_phi3" in samples.keys():
        phi3 = 2 * np.pi * samples["unit_phi3"]
    elif "phi3" in samples.keys():
        phi3 = samples["phi3"]
    else:
        phi3 = rng_interval(sample_size, 0.0, 2 * np.pi, rng=rng)

    P1CM = Cfv.build_fourvec(E1CM, p1CM, np.full_like(costN, 1.0), np.full_like(phi3, 0))
    P2CM = Cfv.build_fourvec(E2CM, -p1CM, np.full_like(costN, 1.0), np.full_like(phi3, 0))
    P3CM = Cfv.build_fourvec(E3CM, p3CM, costN, phi3)
    P4CM = Cfv.build_fourvec(E4CM, -p3CM, costN, phi3)

    # incoming neutrino
    P1LAB = Cfv.L(P1CM, beta)
    # incoming Hadron
    P2LAB = Cfv.L(P2CM, beta)
    # outgoing neutrino
    P3LAB = Cfv.L(P3CM, beta)
    # outgoing Hadron
    P4LAB = Cfv.L(P4CM, beta)

    return P1LAB, P2LAB, P3LAB, P4LAB


# Two body decay
# p1 (k1) --> p2(k2) p3(k3)
def two_body_decay(samples, boost=False, m1=1, m2=0, m3=0, rng=np.random.random):

    if not samples:
        logger.debug("DEBUG: No samples were passed to two_body_decay. Assuming uniform phase space.")
        sample_size = np.shape(list(boost.values())[0])[0]
    else:
        # get sample size of the first item
        sample_size = np.shape(list(samples.values())[0])[0]

    # cosine of the angle between k2 and z axis
    if "unit_cost" in samples.keys():
        cost = 2 * samples["unit_cost"] - 1
    elif "cost" in samples.keys():
        cost = samples["cost"]
    else:
        logger.debug("DEBUG: Could not find cost samples, using uniform distribution instead.")
        cost = rng_interval(sample_size, -1, 1, rng=rng)

    E1CM_decay = np.full_like(cost, m1)
    E2CM_decay = np.full_like(cost, (m1**2 + m2**2 - m3**2) / 2.0 / m1)
    E3CM_decay = np.full_like(cost, (m1**2 - m2**2 + m3**2) / 2.0 / m1)

    p2CM_decay = np.full_like(cost, np.sqrt(E2CM_decay**2 - m2**2))
    # p3CM_decay = np.full_like(cost, np.sqrt(E3CM_decay**2 - m3**2))

    # azimuthal angle of k2
    if "unit_phiz" in samples.keys():
        phiz = 2 * samples["unit_phiz"] - 1
    elif "phiz" in samples.keys():
        phiz = samples["phiz"]
    else:
        logger.debug("DEBUG: Could not find phiz samples, using uniform distribution instead.")
        phiz = rng_interval(sample_size, 0.0, 2 * np.pi, rng=rng)

    P1CM_decay = Cfv.build_fourvec(E1CM_decay, p2CM_decay * 0.0, cost / cost, phiz * 0)
    P2CM_decay = Cfv.build_fourvec(E2CM_decay, p2CM_decay, cost, phiz)
    P3CM_decay = Cfv.build_fourvec(E3CM_decay, -p2CM_decay, cost, phiz)

    # four-momenta in the LAB frame
    if boost:
        EP_LAB = boost["EP_LAB"]
        p1_LAB = np.sqrt(EP_LAB**2 - m1**2)
        costP_LAB = boost["costP_LAB"]
        phiP_LAB = boost["phiP_LAB"]

        # Transform P2_CM into the LAB frame (determined by the PN vector)

        P1LAB_decay = Cfv.build_fourvec(EP_LAB, p1_LAB, costP_LAB, phiP_LAB)

        P2LAB_decay = Cfv.Tinv(P2CM_decay, -p1_LAB / EP_LAB, costP_LAB, phiP_LAB)
        P3LAB_decay = Cfv.Tinv(P3CM_decay, -p1_LAB / EP_LAB, costP_LAB, phiP_LAB)

        return P1LAB_decay, P2LAB_decay, P3LAB_decay

    else:
        return P1CM_decay, P2CM_decay, P3CM_decay


# Three body decay
# p1 (k1) --> p2(k2) p3(k3) p4(k4)
def three_body_decay(samples, boost=False, m1=1, m2=0, m3=0, m4=0, rng=np.random.random):

    if not samples:
        logger.error("Error! No samples were passed to three_body_decay.")
    else:
        # get sample size of the first item
        sample_size = np.shape(list(samples.values())[0])[0]

    # Mandelstam t = m23^2
    tminus = (m2 + m3) ** 2
    tplus = (m1 - m4) ** 2
    if "unit_t" in samples.keys():
        t = (tplus - tminus) * samples["unit_t"] + tminus
    elif "t" in samples.keys():
        t = samples["t"]
    else:
        logger.debug("DEBUG: Could not find t samples, using uniform distribution instead.")
        t = rng_interval(sample_size, tminus, tplus, rng=rng)

    # Mandelstam u = m_24^2
    # from MATHEMATICA
    uplus = 1 / 4 * (((m1) ** (2) + ((m2) ** (2) + (-1 * (m3) ** (2) + -1 * (m4) ** (2))))) ** (2) * (t) ** (-1) + -1 * (
        (
            ((-1 * (m2) ** (2) + 1 / 4 * (t) ** (-1) * (((m2) ** (2) + (-1 * (m3) ** (2) + t))) ** (2))) ** (1 / 2)
            + -1 * ((-1 * (m4) ** (2) + 1 / 4 * (t) ** (-1) * ((-1 * (m1) ** (2) + ((m4) ** (2) + t))) ** (2))) ** (1 / 2)
        )
    ) ** (2)
    # from MATHEMATICA
    uminus = 1 / 4 * (((m1) ** (2) + ((m2) ** (2) + (-1 * (m3) ** (2) + -1 * (m4) ** (2))))) ** (2) * (t) ** (-1) + -1 * (
        (
            ((-1 * (m2) ** (2) + 1 / 4 * (t) ** (-1) * (((m2) ** (2) + (-1 * (m3) ** (2) + t))) ** (2))) ** (1 / 2)
            + ((-1 * (m4) ** (2) + 1 / 4 * (t) ** (-1) * ((-1 * (m1) ** (2) + ((m4) ** (2) + t))) ** (2))) ** (1 / 2)
        )
    ) ** (2)
    if "unit_u" in samples.keys():
        u = (uplus - uminus) * samples["unit_u"] + uminus
    elif "u" in samples.keys():
        u = samples["u"]
    else:
        logger.debug("DEBUG: Could not find u samples, using uniform distribution instead.")
        u = rng_interval(sample_size, uminus, uplus, rng=rng)

    # Mandelstam v = m_34^2
    # v = m1**2 + m2**2 + m3**2 + m4**2 - u - t

    # E2CM_decay = (m1**2 + m2**2 - v) / 2.0 / m1
    E3CM_decay = (m1**2 + m3**2 - u) / 2.0 / m1
    E4CM_decay = (m1**2 + m4**2 - t) / 2.0 / m1

    # p2CM_decay = np.sqrt(E2CM_decay * E2CM_decay - m2**2)
    p3CM_decay = np.sqrt(E3CM_decay * E3CM_decay - m3**2)
    p4CM_decay = np.sqrt(E4CM_decay * E4CM_decay - m4**2)

    # Polar angle of P_3
    if "unit_c3" in samples.keys():
        c_theta3 = 2 * samples["unit_c3"] - 1
    elif "c3" in samples.keys():
        c_theta3 = samples["c3"]
    else:
        logger.debug("DEBUG: Could not find c3 samples, using uniform distribution instead.")
        c_theta3 = rng_interval(sample_size, -1, 1, rng=rng)

    phi3 = rng_interval(sample_size, 0.0, 2 * np.pi, rng=rng)

    # Azimuthal angle of P_4 wrt to P_3 (phi_34)
    if "unit_phi34" in samples.keys():
        phi34 = 2 * np.pi * samples["unit_phi34"]
    elif "phi34" in samples.keys():
        phi34 = samples["phi34"]
    else:
        logger.debug("DEBUG: Could not find phi34 samples, using uniform distribution instead.")
        phi34 = rng_interval(sample_size, 0, 2 * np.pi, rng=rng)

    # polar angle of P_4 wrt to P_3 is a known function of u and v
    c_theta34 = (t + u - m2**2 - m1**2 + 2 * E3CM_decay * E4CM_decay) / (2 * p3CM_decay * p4CM_decay)

    # p1
    P1CM_decay = Cfv.build_fourvec(m1 * np.ones(sample_size), np.zeros(sample_size), np.ones(sample_size), np.zeros(sample_size))
    # p3
    P3CM_decay = Cfv.build_fourvec(E3CM_decay, p3CM_decay, c_theta3, phi3)
    # p4 -- first in p3 along Z, then rotated to CM frame
    P4CM_decay = Cfv.rotationz(Cfv.rotationy_cos(Cfv.build_fourvec(E4CM_decay, p4CM_decay, c_theta34, phi34), c_theta3, sign=-1), phi3)
    # p2
    P2CM_decay = P1CM_decay - P4CM_decay - P3CM_decay

    # four-momenta in the LAB frame
    if boost:
        EN_LAB = boost["EP_LAB"]
        costN_LAB = boost["costP_LAB"]
        phiN_LAB = boost["phiP_LAB"]

        # Transform from CM into the LAB frame
        # Decaying neutrino
        P1LAB_decay = Cfv.Tinv(P1CM_decay, -np.sqrt(EN_LAB**2 - m1**2) / EN_LAB, costN_LAB, phiN_LAB)
        # Outgoing neutrino
        P2LAB_decay = Cfv.Tinv(P2CM_decay, -np.sqrt(EN_LAB**2 - m1**2) / EN_LAB, costN_LAB, phiN_LAB)
        # Outgoing lepton minus (3)
        P3LAB_decay = Cfv.Tinv(P3CM_decay, -np.sqrt(EN_LAB**2 - m1**2) / EN_LAB, costN_LAB, phiN_LAB)
        # Outgoing lepton plus (4)
        P4LAB_decay = Cfv.Tinv(P4CM_decay, -np.sqrt(EN_LAB**2 - m1**2) / EN_LAB, costN_LAB, phiN_LAB)

        return P1LAB_decay, P2LAB_decay, P3LAB_decay, P4LAB_decay

    # four-momenta in the parent particle rest frame
    else:
        return P1CM_decay, P2CM_decay, P3CM_decay, P4CM_decay
