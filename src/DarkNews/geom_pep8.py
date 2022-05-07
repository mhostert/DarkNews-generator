import numpy as np

from . import const
from .const import *

from DarkNews import logger


def upscattering_dxsec_dQ2(x_phase_space, process, diagrams=['total']):
    '''
        Return the differential cross section for upscattering in attobarns

            process: UpscatteringProcess object with all model parameters and scope of upscattering process

            diagrams:   all -- returns a dictionary with all the separate contributions to the xsecs
                        separating diagrams with Z', Z, S, etc.

                        NC_SQR
                        
                        KinMix_SQR
                        KinMix_NC_inter
                        
                        MassMix_SQR
                        MassMix_NC_inter
                        KinMix_MassMix_inter
                        
                        TMM_SQR
                        
                        Scalar_SQR
                        Scalar_NC_inter
                        Scalar_KinMix_inter
                        Scalar_MassMix_inter
                    
                        total

    '''

    # kinematics
    s, t, u = x_phase_space
    Q2 = -t

    # hadronic target
    target = process.target

    # masses
    M = process.target.mass
    Z = target.Z
    mHNL = process.m_ups
    mzprime = process.mzprime
    MSCALAR = process.mhprime

    # vertices
    Chad = process.Chad
    Cprimehad = process.Cprimehad
    Vhad = process.Vhad
    Shad = process.Shad
    Cij = process.Cij
    Cji = process.Cji
    Vij = process.Vij
    Vji = process.Vji
    Sij = process.Sij
    Sji = process.Sji

    mu_tr = process.mu_tr

    MAJ = process.MAJ
    h = process.h_upscattered

    # Form factors
    if target.is_nucleus:
        FFf1 = target.F1_EM(Q2)
        FFf2 = 0.0  ### FIX ME
        FFga = 0.0
        FFgp = 0.0

        FFNCf1 = target.F1_NC(Q2)
        FFNCf2 = 0.0
        FFNCga = 0.0
        FFNCgp = 0.0

        FFscalar = FFf1

    elif (target.is_nucleon):
        FFf1 = target.F1_EM(Q2)
        FFf2 = target.F2_EM(Q2)
        FFga = 0.0
        FFgp = 0.0

        FFNCf1 = target.F1_NC(Q2)
        FFNCf2 = target.F2_NC(Q2)
        FFNCga = target.F3_NC(Q2)
        FFNCgp = 0.0

        FFscalar = FFf1

    else:
        logger.error('upscattering on a lepton not implemented.')

    ## Spin summed (but not averaged) matrix elements from MATHEMATICA
    # |M|^2 = | M_SM + M_kinmix + M_massmix|^2
    if process.TheoryModel.HNLtype == 'majorana':
        # SM NC SQR
        def Lmunu_Hmunu_NC_SQR():
            return (Chad * Chad * Cij * Cji *
                    (16 * FFNCga * FFNCgp * (mHNL * mHNL) *
                     (-(mHNL * mHNL) + t) +
                     (4 * (FFNCgp * FFNCgp) * (mHNL * mHNL) * t *
                      (-(mHNL * mHNL) + t)) / (M * M) - 8 * FFNCf1 * FFNCf2 *
                     (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                      (t * t)) + 8 * (FFNCf1 * FFNCf1) *
                     (2 * (M * M * M * M) - 4 * (M * M) * s + 2 *
                      (s * s) + 2 * s * t + t * t - mHNL * mHNL *
                      (2 * s + t)) + 8 * (FFNCga * FFNCga) *
                     (2 * (M * M * M * M) + 2 * (s * s) + 4 * (M * M) *
                      (mHNL * mHNL - s - t) + 2 * s * t + t * t - mHNL * mHNL *
                      (2 * s + t)) -
                     (FFNCf2 * FFNCf2 *
                      (4 * (M * M * M * M) * t + 4 * (M * M) *
                       (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 * t *
                        (s + t)) + t * (mHNL * mHNL * mHNL * mHNL + 4 * s *
                                        (s + t) - mHNL * mHNL *
                                        (4 * s + t)))) / (M * M) +
                     (16 * FFNCf1 * FFNCga * h *
                      (2 * (M * M * M * M) * t + t *
                       (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                        (-s + t) + s * (2 * s + t)) + M * M *
                       (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                        (mHNL * mHNL) * t - t * (4 * s + t)))) / (s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s))) +
                     (16 * FFNCf2 * FFNCga * h *
                      (2 * (M * M * M * M) * t + t *
                       (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                        (-s + t) + s * (2 * s + t)) + M * M *
                       (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                        (mHNL * mHNL) * t - t * (4 * s + t)))) / (s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) /
                            (s * s))))) / (2. * ((MZBOSON * MZBOSON - t) *
                                                 (MZBOSON * MZBOSON - t)))

        # kinetic mixing term SQR
        def Lmunu_Hmunu_KinMix_SQR():
            return ((-8 * FFf1 * FFf2 * (M * M) *
                     (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                      (t * t)) + 8 * (FFf1 * FFf1) * (M * M) *
                     (2 * (M * M * M * M) - 4 * (M * M) * s + 2 *
                      (s * s) + 2 * s * t + t * t - mHNL * mHNL *
                      (2 * s + t)) - FFf2 * FFf2 *
                     (4 * (M * M * M * M) * t + 4 * (M * M) *
                      (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 * t *
                       (s + t)) + t * (mHNL * mHNL * mHNL * mHNL + 4 * s *
                                       (s + t) - mHNL * mHNL * (4 * s + t)))) *
                    (Vhad * Vhad) * Vij * Vji) / (2. * (M * M) *
                                                  ((mzprime * mzprime - t) *
                                                   (mzprime * mzprime - t)))

        # kinetic mixing + SM NC interference
        def Lmunu_Hmunu_KinMix_NC_inter():
            return ((Chad *
                     (4 * FFf1 * (M * M) *
                      (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                                 (mHNL * mHNL - s) - 2 * (M * M) *
                                 (mHNL * mHNL + s)) / (s * s)) *
                       (-(FFNCf2 *
                          (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                           (t * t))) + 2 * FFNCf1 *
                        (2 * (M * M * M * M) - 4 * (M * M) * s + 2 *
                         (s * s) + 2 * s * t + t * t - mHNL * mHNL *
                         (2 * s + t))) + 2 * FFNCga * h *
                       (2 * (M * M * M * M) * t + t *
                        (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                         (-s + t) + s * (2 * s + t)) + M * M *
                        (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                         (mHNL * mHNL) * t - t * (4 * s + t)))) + FFf2 *
                      (8 * FFNCga * h * (M * M) *
                       (2 * (M * M * M * M) * t + t *
                        (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                         (-s + t) + s * (2 * s + t)) + M * M *
                        (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                         (mHNL * mHNL) * t - t * (4 * s + t))) - s * Sqrt(
                             (M * M * M * M + (mHNL * mHNL - s) *
                              (mHNL * mHNL - s) - 2 * (M * M) *
                              (mHNL * mHNL + s)) / (s * s)) *
                       (4 * FFNCf1 * (M * M) *
                        (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                         (t * t)) + FFNCf2 *
                        (4 * (M * M * M * M) * t + 4 * (M * M) *
                         (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 * t *
                          (s + t)) + t * (mHNL * mHNL * mHNL * mHNL + 4 * s *
                                          (s + t) - mHNL * mHNL *
                                          (4 * s + t)))))) * Vhad *
                     (Cji * Vij + Cij * Vji)) /
                    (M * M * s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                                       (mHNL * mHNL - s) - 2 * (M * M) *
                                       (mHNL * mHNL + s)) / (s * s)) *
                     (MZBOSON * MZBOSON - t) * (-(mzprime * mzprime) + t)))

        # mass mixing + SM NC interference
        def Lmunu_Hmunu_MassMix_NC_inter():
            return -(
                (Sqrt(Chad * Cprimehad) *
                 (4 * FFf1 * (M * M) *
                  (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) *
                   (-(FFNCf2 *
                      (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                       (t * t))) + 2 * FFNCf1 *
                    (2 * (M * M * M * M) - 4 * (M * M) * s + 2 *
                     (s * s) + 2 * s * t + t * t - mHNL * mHNL *
                     (2 * s + t))) + 2 * FFNCga * h *
                   (2 * (M * M * M * M) * t + t *
                    (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                     (-s + t) + s * (2 * s + t)) + M * M *
                    (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                     (mHNL * mHNL) * t - t * (4 * s + t)))) + FFf2 *
                  (8 * FFNCga * h * (M * M) *
                   (2 * (M * M * M * M) * t + t *
                    (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                     (-s + t) + s * (2 * s + t)) + M * M *
                    (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                     (mHNL * mHNL) * t - t * (4 * s + t))) - s * Sqrt(
                         (M * M * M * M + (mHNL * mHNL - s) *
                          (mHNL * mHNL - s) - 2 * (M * M) *
                          (mHNL * mHNL + s)) / (s * s)) *
                   (4 * FFNCf1 * (M * M) *
                    (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                     (t * t)) + FFNCf2 *
                    (4 * (M * M * M * M) * t + 4 * (M * M) *
                     (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 * t *
                      (s + t)) + t * (mHNL * mHNL * mHNL * mHNL + 4 * s *
                                      (s + t) - mHNL * mHNL *
                                      (4 * s + t)))))) * Vhad *
                 (Cji * Vij + Cij * Vji)) / (M * M * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * (MZBOSON * MZBOSON - t) *
                                             (-(mzprime * mzprime) + t)))

        # kinetic mixing + mass mixing interference
        def Lmunu_Hmunu_KinMix_MassMix_inter():
            return (Cprimehad *
                    (4 * FFf1 * (M * M) *
                     (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                                (mHNL * mHNL - s) - 2 * (M * M) *
                                (mHNL * mHNL + s)) / (s * s)) *
                      (-(FFNCf2 *
                         (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                          (t * t))) + 2 * FFNCf1 *
                       (2 * (M * M * M * M) - 4 * (M * M) * s + 2 *
                        (s * s) + 2 * s * t + t * t - mHNL * mHNL *
                        (2 * s + t))) + 2 * FFNCga * h *
                      (2 * (M * M * M * M) * t + t *
                       (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                        (-s + t) + s * (2 * s + t)) + M * M *
                       (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                        (mHNL * mHNL) * t - t * (4 * s + t)))) + FFf2 *
                     (8 * FFNCga * h * (M * M) *
                      (2 * (M * M * M * M) * t + t *
                       (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                        (-s + t) + s * (2 * s + t)) + M * M *
                       (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                        (mHNL * mHNL) * t - t * (4 * s + t))) - s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) *
                      (4 * FFNCf1 * (M * M) *
                       (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                        (t * t)) + FFNCf2 *
                       (4 * (M * M * M * M) * t + 4 * (M * M) *
                        (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 * t *
                         (s + t)) + t * (mHNL * mHNL * mHNL * mHNL + 4 * s *
                                         (s + t) - mHNL * mHNL *
                                         (4 * s + t)))))) * Vhad * Vij *
                    Vji) / (M * M * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * ((mzprime * mzprime - t) *
                                    (mzprime * mzprime - t)))

        # mass mixing term SQR
        def Lmunu_Hmunu_MassMix_SQR():
            return (Cprimehad * Cprimehad *
                    (16 * FFNCga * FFNCgp * (mHNL * mHNL) *
                     (-(mHNL * mHNL) + t) +
                     (4 * (FFNCgp * FFNCgp) * (mHNL * mHNL) * t *
                      (-(mHNL * mHNL) + t)) / (M * M) - 8 * FFNCf1 * FFNCf2 *
                     (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                      (t * t)) + 8 * (FFNCf1 * FFNCf1) *
                     (2 * (M * M * M * M) - 4 * (M * M) * s + 2 *
                      (s * s) + 2 * s * t + t * t - mHNL * mHNL *
                      (2 * s + t)) + 8 * (FFNCga * FFNCga) *
                     (2 * (M * M * M * M) + 2 * (s * s) + 4 * (M * M) *
                      (mHNL * mHNL - s - t) + 2 * s * t + t * t - mHNL * mHNL *
                      (2 * s + t)) -
                     (FFNCf2 * FFNCf2 *
                      (4 * (M * M * M * M) * t + 4 * (M * M) *
                       (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 * t *
                        (s + t)) + t * (mHNL * mHNL * mHNL * mHNL + 4 * s *
                                        (s + t) - mHNL * mHNL *
                                        (4 * s + t)))) / (M * M) +
                     (16 * FFNCf1 * FFNCga * h *
                      (2 * (M * M * M * M) * t + t *
                       (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                        (-s + t) + s * (2 * s + t)) + M * M *
                       (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                        (mHNL * mHNL) * t - t * (4 * s + t)))) / (s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s))) +
                     (16 * FFNCf2 * FFNCga * h *
                      (2 * (M * M * M * M) * t + t *
                       (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                        (-s + t) + s * (2 * s + t)) + M * M *
                       (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                        (mHNL * mHNL) * t - t * (4 * s + t)))) /
                     (s * Sqrt(
                         (M * M * M * M + (mHNL * mHNL - s) *
                          (mHNL * mHNL - s) - 2 * (M * M) *
                          (mHNL * mHNL + s)) /
                         (s * s)))) * Vij * Vji) / (2. *
                                                    ((mzprime * mzprime - t) *
                                                     (mzprime * mzprime - t)))

        # transition magnetic moment
        def Lmunu_Hmunu_TMM_SQR():
            return (mu_tr)**(2) * (eQED)**(2) * (M)**(-2) * (((M)**(4) + (
                (((mHNL)**(2) + -1 * s))**(2) + -2 * (M)**(2) *
                ((mHNL)**(2) + s))))**(-1 / 2) * (t)**(-2) * (
                    (FFf2)**(2) * t *
                    ((((M)**(4) + ((((mHNL)**(2) + -1 * s))**(2) + -2 *
                                   (M)**(2) * ((mHNL)**(2) + s))))**(1 / 2) *
                     (-4 * (M)**(2) * (mHNL)**(4) +
                      (4 * ((M)**(2) + -1 * s) * ((M)**(2) +
                                                  ((mHNL)**(2) + -1 * s)) * t +
                       (-1 * ((mHNL)**(2) + -4 * s) * (t)**(2) +
                        (t)**(3)))) + h *
                     (4 * (M)**(2) * (mHNL)**(4) * ((M)**(2) +
                                                    ((mHNL)**(2) + -1 * s)) +
                      (-4 * ((M)**(2) + -1 * s) *
                       (((M + -1 * mHNL))**(2) + -1 * s) *
                       (((M + mHNL))**(2) + -1 * s) * t +
                       (-1 * ((M)**(2) + ((mHNL)**(2) + -1 * s)) *
                        ((mHNL)**(2) + 4 * s) * (t)**(2) +
                        (-1 * (M)**(2) + ((mHNL)**(2) + s)) * (t)**(3))))) +
                    (8 * FFf1 * FFf2 * (M)**(2) * t *
                     ((((M)**(4) + ((((mHNL)**(2) + -1 * s))**(2) + -2 *
                                    (M)**(2) * ((mHNL)**(2) + s))))**(1 / 2) *
                      (-2 * (mHNL)**(4) + ((mHNL)**(2) * t + (t)**(2))) + h *
                      (2 * (mHNL)**(2) + -1 * t) *
                      ((mHNL)**(4) +
                       (-1 * s * t +
                        ((M)**(2) * ((mHNL)**(2) + t) + -1 * (mHNL)**(2) *
                         (s + t))))) + 8 * (FFf1)**(2) * (M)**(2) *
                     (h * (2 * (M)**(2) * (mHNL)**(4) *
                           ((M)**(2) + ((mHNL)**(2) + -1 * s)) +
                           (((M)**(2) + ((mHNL)**(2) + -1 * s)) *
                            (2 * (M)**(4) + ((mHNL)**(4) +
                                             (-2 * (mHNL)**(2) * s +
                                              (2 * (s)**(2) + -4 * (M)**(2) *
                                               ((mHNL)**(2) + s))))) * t +
                            (-2 * (M)**(4) +
                             (-1 * (mHNL)**(4) +
                              ((mHNL)**(2) * s +
                               (-2 * (s)**(2) + (M)**(2) *
                                (3 * (mHNL)**(2) + 4 * s))))) * (t)**(2))) +
                      (((M)**(4) + ((((mHNL)**(2) + -1 * s))**(2) + -2 *
                                    (M)**(2) * ((mHNL)**(2) + s))))**(1 / 2) *
                      (-2 * (M)**(4) * t +
                       (t * (-1 * (mHNL)**(4) +
                             (-2 * s * (s + t) + (mHNL)**(2) *
                              (2 * s + t))) + 2 * (M)**(2) *
                        (-1 * (mHNL)**(4) + t * (2 * s + t))))))) * (Z)**(2)

        # scalar term SQR ########FIX-ME THESE ARE ALL STILL DIRAC vvvvvvv ########
        def Lmunu_Hmunu_Scalar_SQR():
            return (FFscalar * FFscalar * (Shad * Shad) * Sij * Sji *
                    (4 * (M * M) - t) *
                    (s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * (mHNL * mHNL - t) + h *
                     (mHNL * mHNL * mHNL * mHNL - s * t + M * M *
                      (mHNL * mHNL + t) - mHNL * mHNL * (s + t)))) / (s * Sqrt(
                          (M * M * M * M + (mHNL * mHNL - s) *
                           (mHNL * mHNL - s) - 2 * (M * M) *
                           (mHNL * mHNL + s)) /
                          (s * s)) * ((MSCALAR * MSCALAR - t) *
                                      (MSCALAR * MSCALAR - t)))

        # scalar + SM NC term
        def Lmunu_Hmunu_Scalar_NC_inter():
            return (Chad * Cij * FFscalar * mHNL * Shad * Sij *
                    (4 * FFNCf1 * (M * M) + FFNCf2 * t) *
                    (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                               (mHNL * mHNL - s) - 2 * (M * M) *
                               (mHNL * mHNL + s)) / (s * s)) *
                     (2 * (M * M) + mHNL * mHNL - 2 * s - t) + h *
                     (2 * (M * M * M * M) + mHNL * mHNL * mHNL * mHNL + M * M *
                      (-(mHNL * mHNL) - 4 * s + t) - mHNL * mHNL *
                      (3 * s + t) + s *
                      (2 * s + 3 * t)))) / (M * s * Sqrt(
                          (M * M * M * M + (mHNL * mHNL - s) *
                           (mHNL * mHNL - s) - 2 * (M * M) *
                           (mHNL * mHNL + s)) / (s * s)) *
                                            (MSCALAR * MSCALAR - t) *
                                            (-(MZBOSON * MZBOSON) + t))

        # scalar + kinetic mixing interference
        def Lmunu_Hmunu_Scalar_KinMix_inter():
            return (FFscalar * mHNL * Shad * Sij * (4 * FFf1 *
                                                    (M * M) + FFf2 * t) *
                    (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                               (mHNL * mHNL - s) - 2 * (M * M) *
                               (mHNL * mHNL + s)) / (s * s)) *
                     (2 * (M * M) + mHNL * mHNL - 2 * s - t) + h *
                     (2 * (M * M * M * M) + mHNL * mHNL * mHNL * mHNL + M * M *
                      (-(mHNL * mHNL) - 4 * s + t) - mHNL * mHNL *
                      (3 * s + t) + s * (2 * s + 3 * t))) * Vhad *
                    Vij) / (M * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * (MSCALAR * MSCALAR - t) *
                            (-(mzprime * mzprime) + t))

        # scalar + mass mixing interference
        def Lmunu_Hmunu_Scalar_MassMix_inter():
            return (Cprimehad * Cij * FFscalar * mHNL * Shad * Sij *
                    (4 * FFNCf1 * (M * M) + FFNCf2 * t) *
                    (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                               (mHNL * mHNL - s) - 2 * (M * M) *
                               (mHNL * mHNL + s)) / (s * s)) *
                     (2 * (M * M) + mHNL * mHNL - 2 * s - t) + h *
                     (2 * (M * M * M * M) + mHNL * mHNL * mHNL * mHNL + M * M *
                      (-(mHNL * mHNL) - 4 * s + t) - mHNL * mHNL *
                      (3 * s + t) + s *
                      (2 * s + 3 * t)))) / (M * s * Sqrt(
                          (M * M * M * M + (mHNL * mHNL - s) *
                           (mHNL * mHNL - s) - 2 * (M * M) *
                           (mHNL * mHNL + s)) / (s * s)) *
                                            (MSCALAR * MSCALAR - t) *
                                            (-(MZBOSON * MZBOSON) + t))

    elif process.TheoryModel.HNLtype == 'dirac':
        # SM NC SQR
        def Lmunu_Hmunu_NC_SQR():
            return (
                Chad * Chad * Cij * Cji *
                (-4 * (FFNCf2 * FFNCf2) * h * (M * M * M * M) *
                 (mHNL * mHNL * mHNL * mHNL) - 4 * (FFNCf2 * FFNCf2) * h *
                 (M * M) * (mHNL * mHNL * mHNL * mHNL * mHNL * mHNL) + 4 *
                 (FFNCf2 * FFNCf2) * h * (M * M) *
                 (mHNL * mHNL * mHNL * mHNL) * s - 4 * (FFNCf2 * FFNCf2) *
                 (M * M) *
                 (mHNL * mHNL * mHNL * mHNL) * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) - 4 * (FFNCf2 * FFNCf2) * h *
                 (M * M * M * M * M * M) * t + 8 * (FFNCf2 * FFNCf2) * h *
                 (M * M * M * M) * (mHNL * mHNL) * t + 11 *
                 (FFNCf2 * FFNCf2) * h * (M * M) *
                 (mHNL * mHNL * mHNL * mHNL) * t - 4 * (FFNCgp * FFNCgp) * h *
                 (M * M) *
                 (mHNL * mHNL * mHNL * mHNL) * t - FFNCf2 * FFNCf2 * h *
                 (mHNL * mHNL * mHNL * mHNL * mHNL * mHNL) * t - 4 *
                 (FFNCgp * FFNCgp) * h *
                 (mHNL * mHNL * mHNL * mHNL * mHNL * mHNL) * t + 12 *
                 (FFNCf2 * FFNCf2) * h * (M * M * M * M) * s * t + 5 *
                 (FFNCf2 * FFNCf2) * h *
                 (mHNL * mHNL * mHNL * mHNL) * s * t + 4 *
                 (FFNCgp * FFNCgp) * h *
                 (mHNL * mHNL * mHNL * mHNL) * s * t - 12 *
                 (FFNCf2 * FFNCf2) * h * (M * M) * (s * s) * t - 8 *
                 (FFNCf2 * FFNCf2) * h * (mHNL * mHNL) * (s * s) * t + 4 *
                 (FFNCf2 * FFNCf2) * h * (s * s * s) * t - 4 *
                 (FFNCf2 * FFNCf2) * (M * M * M * M) * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * t - 4 * (FFNCf2 * FFNCf2) * (M * M) *
                 (mHNL * mHNL) * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * t - FFNCf2 * FFNCf2 *
                 (mHNL * mHNL * mHNL * mHNL) * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * t - 4 * (FFNCgp * FFNCgp) *
                 (mHNL * mHNL * mHNL * mHNL) * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * t + 8 * (FFNCf2 * FFNCf2) * (M * M) *
                 (s * s) * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                                 (mHNL * mHNL - s) - 2 * (M * M) *
                                 (mHNL * mHNL + s)) / (s * s)) * t + 4 *
                 (FFNCf2 * FFNCf2) * (mHNL * mHNL) * (s * s) * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * t - 4 * (FFNCf2 * FFNCf2) *
                 (s * s * s) * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                                     (mHNL * mHNL - s) - 2 * (M * M) *
                                     (mHNL * mHNL + s)) / (s * s)) * t + 8 *
                 (FFNCf2 * FFNCf2) * h * (M * M * M * M) * (t * t) - 9 *
                 (FFNCf2 * FFNCf2) * h * (M * M) * (mHNL * mHNL) *
                 (t * t) - 4 * (FFNCgp * FFNCgp) * h * (M * M) *
                 (mHNL * mHNL) * (t * t) + FFNCf2 * FFNCf2 * h *
                 (mHNL * mHNL * mHNL * mHNL) * (t * t) + 4 *
                 (FFNCgp * FFNCgp) * h * (mHNL * mHNL * mHNL * mHNL) *
                 (t * t) - 12 * (FFNCf2 * FFNCf2) * h * (M * M) * s *
                 (t * t) - 3 * (FFNCf2 * FFNCf2) * h * (mHNL * mHNL) * s *
                 (t * t) + 4 * (FFNCgp * FFNCgp) * h * (mHNL * mHNL) * s *
                 (t * t) + 4 * (FFNCf2 * FFNCf2) * h * (s * s) * (t * t) + 8 *
                 (FFNCf2 * FFNCf2) * (M * M) * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * (t * t) + FFNCf2 * FFNCf2 *
                 (mHNL * mHNL) * s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                                           (mHNL * mHNL - s) - 2 * (M * M) *
                                           (mHNL * mHNL + s)) / (s * s)) *
                 (t * t) + 4 * (FFNCgp * FFNCgp) * (mHNL * mHNL) * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * (t * t) - 4 * (FFNCf2 * FFNCf2) *
                 (s * s) * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * (t * t) + 8 * (FFNCf1 * FFNCf1) * (M * M) *
                 (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                            (mHNL * mHNL - s) - 2 * (M * M) *
                            (mHNL * mHNL + s)) / (s * s)) *
                  (2 * (M * M * M * M) - 4 * (M * M) * s + 2 *
                   (s * s) + 2 * s * t + t * t - mHNL * mHNL *
                   (2 * s + t)) + h *
                  (2 * (M * M * M * M * M * M) - 2 *
                   ((mHNL * mHNL - s) *
                    (mHNL * mHNL - s)) * s - 2 * (M * M * M * M) *
                   (mHNL * mHNL + 3 * s) +
                   (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * s - 2 *
                    (s * s)) * t - (mHNL * mHNL + s) * (t * t) + M * M *
                   (6 * (s * s) + 2 * s * t + t * t + mHNL * mHNL *
                    (-2 * s + t)))) - 8 * FFNCf1 * (M * M) *
                 (2 * FFNCga * s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                                         (mHNL * mHNL - s) - 2 * (M * M) *
                                         (mHNL * mHNL + s)) / (s * s)) * t *
                  (-2 * (M * M) - mHNL * mHNL + 2 * s + t) + FFNCf2 * s * Sqrt(
                      (M * M * M * M + (mHNL * mHNL - s) *
                       (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                      (s * s)) *
                  (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                   (t * t)) + FFNCf2 * h * (mHNL * mHNL - 2 * t) *
                  (mHNL * mHNL * mHNL * mHNL - s * t + M * M *
                   (mHNL * mHNL + t) - mHNL * mHNL *
                   (s + t)) + 2 * FFNCga * h *
                  (-2 * (M * M * M * M) * t + t *
                   (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * (s - t) - s *
                    (2 * s + t)) + M * M *
                   (-4 * (mHNL * mHNL * mHNL * mHNL) + 3 *
                    (mHNL * mHNL) * t + t *
                    (4 * s + t)))) + 8 * (FFNCga * FFNCga) * (M * M) *
                 (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                            (mHNL * mHNL - s) - 2 * (M * M) *
                            (mHNL * mHNL + s)) / (s * s)) *
                  (2 * (M * M * M * M) + 2 * (s * s) + 4 * (M * M) *
                   (mHNL * mHNL - s - t) + 2 * s * t + t * t - mHNL * mHNL *
                   (2 * s + t)) + h *
                  (2 * (M * M * M * M * M * M) - 2 * ((mHNL * mHNL - s) *
                                                      (mHNL * mHNL - s)) * s +
                   (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * s - 2 *
                    (s * s)) * t - (mHNL * mHNL + s) * (t * t) - 2 *
                   (M * M * M * M) * (3 * (mHNL * mHNL + s) + 2 * t) + M * M *
                   (-4 * (mHNL * mHNL * mHNL * mHNL) + 6 *
                    (s * s) + 6 * s * t + t * t + mHNL * mHNL *
                    (2 * s + 5 * t)))) + 16 * FFNCga * (M * M) *
                 (FFNCgp * (mHNL * mHNL) *
                  (s * Sqrt(
                      (M * M * M * M + (mHNL * mHNL - s) *
                       (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                      (s * s)) * (-(mHNL * mHNL) + t) + h *
                   (-(mHNL * mHNL * mHNL * mHNL) + s * t - M * M *
                    (mHNL * mHNL + t) + mHNL * mHNL * (s + t))) + FFNCf2 *
                  (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) *
                   (2 * (M * M) + mHNL * mHNL - 2 * s - t) * t + h *
                   (2 * (M * M * M * M) * t + t *
                    (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                     (-s + t) + s * (2 * s + t)) + M * M *
                    (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                     (mHNL * mHNL) * t - t *
                     (4 * s + t))))))) / (4. * (M * M) * s * Sqrt(
                         (M * M * M * M + (mHNL * mHNL - s) *
                          (mHNL * mHNL - s) - 2 * (M * M) *
                          (mHNL * mHNL + s)) / (s * s)) *
                                          ((MZBOSON * MZBOSON - t) *
                                           (MZBOSON * MZBOSON - t)))

        # kinetic mixing term SQR
        def Lmunu_Hmunu_KinMix_SQR():
            return ((-(FFf2 * FFf2 *
                       (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                                  (mHNL * mHNL - s) - 2 * (M * M) *
                                  (mHNL * mHNL + s)) / (s * s)) *
                        (4 * (M * M) * (mHNL * mHNL * mHNL * mHNL) +
                         (2 * (M * M) + mHNL * mHNL - 2 * s) *
                         (2 * (M * M) + mHNL * mHNL - 2 * s) * t -
                         (8 * (M * M) + mHNL * mHNL - 4 * s) * (t * t)) + h *
                        (4 * (M * M) * (mHNL * mHNL * mHNL * mHNL) *
                         (M * M + mHNL * mHNL - s) +
                         (2 * (M * M) - 4 * M * mHNL + mHNL * mHNL - 2 * s) *
                         (2 * (M * M) + 4 * M * mHNL + mHNL * mHNL - 2 * s) *
                         (M * M + mHNL * mHNL - s) * t -
                         (8 * (M * M * M * M) + mHNL * mHNL * mHNL * mHNL - 3 *
                          (mHNL * mHNL) * s + 4 * (s * s) - 3 * (M * M) *
                          (3 * (mHNL * mHNL) + 4 * s)) *
                         (t * t)))) - 8 * FFf1 * FFf2 * (M * M) *
                     (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                                (mHNL * mHNL - s) - 2 * (M * M) *
                                (mHNL * mHNL + s)) / (s * s)) *
                      (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                       (t * t)) + h * (mHNL * mHNL - 2 * t) *
                      (mHNL * mHNL * mHNL * mHNL - s * t + M * M *
                       (mHNL * mHNL + t) - mHNL * mHNL *
                       (s + t))) + 8 * (FFf1 * FFf1) * (M * M) *
                     (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                                (mHNL * mHNL - s) - 2 * (M * M) *
                                (mHNL * mHNL + s)) / (s * s)) *
                      (2 * (M * M * M * M) - 4 * (M * M) * s + 2 *
                       (s * s) + 2 * s * t + t * t - mHNL * mHNL *
                       (2 * s + t)) + h *
                      (2 * (M * M * M * M * M * M) - 2 *
                       ((mHNL * mHNL - s) *
                        (mHNL * mHNL - s)) * s - 2 * (M * M * M * M) *
                       (mHNL * mHNL + 3 * s) +
                       (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * s - 2 *
                        (s * s)) * t - (mHNL * mHNL + s) * (t * t) + M * M *
                       (6 * (s * s) + 2 * s * t + t * t + mHNL * mHNL *
                        (-2 * s + t))))) *
                    (Vhad * Vhad) * Vij * Vji) / (4. * (M * M) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * ((mzprime * mzprime - t) *
                                    (mzprime * mzprime - t)))
            # return ((-8*FFf1*FFf2*(M*M)*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*(t*t)) + 8*(FFf1*FFf1)*(M*M)*(2*(M*M*M*M) - 4*(M*M)*s + 2*(s*s) + 2*s*t + t*t - mHNL*mHNL*(2*s + t)) - FFf2*FFf2*(4*(M*M*M*M)*t + 4*(M*M)*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*t*(s + t)) + t*(mHNL*mHNL*mHNL*mHNL + 4*s*(s + t) - mHNL*mHNL*(4*s + t))))*(Vhad*Vhad)*Vij*Vji)/(2.*(M*M)*((mzprime*mzprime - t)*(mzprime*mzprime - t)))

        # kinetic mixing + SM NC interference
        def Lmunu_Hmunu_KinMix_NC_inter():
            return -0.5 * (
                Chad * Cij *
                (-(FFf2 *
                   (4 * FFNCf2 * (M * M) *
                    (mHNL * mHNL * mHNL * mHNL) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) + 4 * FFNCf2 * (M * M * M * M) * s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * t - 16 * FFNCga *
                    (M * M * M * M) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * t + 4 * FFNCf2 * (M * M) *
                    (mHNL * mHNL) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * t - 8 * FFNCga * (M * M) *
                    (mHNL * mHNL) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * t + FFNCf2 *
                    (mHNL * mHNL * mHNL * mHNL) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * t - 8 * FFNCf2 * (M * M) * (s * s) * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * t + 16 * FFNCga *
                    (M * M) * (s * s) * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * t - 4 * FFNCf2 * (mHNL * mHNL) *
                    (s * s) * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * t + 4 * FFNCf2 * (s * s * s) * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * t - 8 * FFNCf2 *
                    (M * M) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * (t * t) + 8 * FFNCga * (M * M) * s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * (t * t) - FFNCf2 *
                    (mHNL * mHNL) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * (t * t) + 4 * FFNCf2 * (s * s) * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) /
                            (s * s)) * (t * t) + FFNCf2 * h *
                    (4 * (M * M) * (mHNL * mHNL * mHNL * mHNL) *
                     (M * M + mHNL * mHNL - s) +
                     (2 * (M * M) - 4 * M * mHNL + mHNL * mHNL - 2 * s) *
                     (2 * (M * M) + 4 * M * mHNL + mHNL * mHNL - 2 * s) *
                     (M * M + mHNL * mHNL - s) * t -
                     (8 * (M * M * M * M) + mHNL * mHNL * mHNL * mHNL - 3 *
                      (mHNL * mHNL) * s + 4 * (s * s) - 3 * (M * M) *
                      (3 * (mHNL * mHNL) + 4 * s)) *
                     (t * t)) + 4 * FFNCf1 * (M * M) *
                    (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                               (mHNL * mHNL - s) - 2 * (M * M) *
                               (mHNL * mHNL + s)) / (s * s)) *
                     (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                      (t * t)) + h * (mHNL * mHNL - 2 * t) *
                     (mHNL * mHNL * mHNL * mHNL - s * t + M * M *
                      (mHNL * mHNL + t) - mHNL * mHNL *
                      (s + t))) - 8 * FFNCga * h * (M * M) *
                    (2 * (M * M * M * M) * t + t *
                     (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                      (-s + t) + s * (2 * s + t)) + M * M *
                     (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                      (mHNL * mHNL) * t - t * (4 * s + t))))) + 4 * FFf1 *
                 (M * M) *
                 (-(FFNCf2 *
                    (mHNL * mHNL * mHNL * mHNL) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s))) + 4 * FFNCga * (M * M) * s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * t - FFNCf2 *
                  (mHNL * mHNL) * s * Sqrt(
                      (M * M * M * M + (mHNL * mHNL - s) *
                       (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                      (s * s)) * t + 2 * FFNCga * (mHNL * mHNL) * s * Sqrt(
                          (M * M * M * M + (mHNL * mHNL - s) *
                           (mHNL * mHNL - s) - 2 * (M * M) *
                           (mHNL * mHNL + s)) / (s * s)) * t - 4 * FFNCga *
                  (s * s) * Sqrt(
                      (M * M * M * M + (mHNL * mHNL - s) *
                       (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                      (s * s)) * t + 2 * FFNCf2 * s * Sqrt(
                          (M * M * M * M + (mHNL * mHNL - s) *
                           (mHNL * mHNL - s) - 2 * (M * M) *
                           (mHNL * mHNL + s)) / (s * s)) *
                  (t * t) - 2 * FFNCga * s * Sqrt(
                      (M * M * M * M + (mHNL * mHNL - s) *
                       (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                      (s * s)) * (t * t) - FFNCf2 * h * (mHNL * mHNL - 2 * t) *
                  (mHNL * mHNL * mHNL * mHNL - s * t + M * M *
                   (mHNL * mHNL + t) - mHNL * mHNL *
                   (s + t)) + 2 * FFNCga * h *
                  (2 * (M * M * M * M) * t + t *
                   (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL * (-s + t) + s *
                    (2 * s + t)) + M * M *
                   (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                    (mHNL * mHNL) * t - t * (4 * s + t))) + 2 * FFNCf1 *
                  (s * Sqrt(
                      (M * M * M * M + (mHNL * mHNL - s) *
                       (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                      (s * s)) * (2 * (M * M * M * M) - 4 * (M * M) * s + 2 *
                                  (s * s) + 2 * s * t + t * t - mHNL * mHNL *
                                  (2 * s + t)) + h *
                   (2 * (M * M * M * M * M * M) - 2 *
                    ((mHNL * mHNL - s) * (mHNL * mHNL - s)) * s - 2 *
                    (M * M * M * M) * (mHNL * mHNL + 3 * s) +
                    (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * s - 2 *
                     (s * s)) * t - (mHNL * mHNL + s) * (t * t) + M * M *
                    (6 * (s * s) + 2 * s * t + t * t + mHNL * mHNL *
                     (-2 * s + t)))))) * Vhad *
                Vij) / (M * M * s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                                          (mHNL * mHNL - s) - 2 * (M * M) *
                                          (mHNL * mHNL + s)) / (s * s)) *
                        (MZBOSON * MZBOSON - t) * (-(mzprime * mzprime) + t))

        # mass mixing + SM NC interference
        def Lmunu_Hmunu_MassMix_NC_inter():
            return -0.5 * (
                Sqrt(Chad * Cprimehad) *
                (-(FFf2 *
                   (4 * FFNCf2 * (M * M) *
                    (mHNL * mHNL * mHNL * mHNL) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) + 4 * FFNCf2 * (M * M * M * M) * s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * t - 16 * FFNCga *
                    (M * M * M * M) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * t + 4 * FFNCf2 * (M * M) *
                    (mHNL * mHNL) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * t - 8 * FFNCga * (M * M) *
                    (mHNL * mHNL) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * t + FFNCf2 *
                    (mHNL * mHNL * mHNL * mHNL) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * t - 8 * FFNCf2 * (M * M) * (s * s) * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * t + 16 * FFNCga *
                    (M * M) * (s * s) * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * t - 4 * FFNCf2 * (mHNL * mHNL) *
                    (s * s) * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * t + 4 * FFNCf2 * (s * s * s) * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * t - 8 * FFNCf2 *
                    (M * M) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * (t * t) + 8 * FFNCga * (M * M) * s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * (t * t) - FFNCf2 *
                    (mHNL * mHNL) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * (t * t) + 4 * FFNCf2 * (s * s) * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) /
                            (s * s)) * (t * t) + FFNCf2 * h *
                    (4 * (M * M) * (mHNL * mHNL * mHNL * mHNL) *
                     (M * M + mHNL * mHNL - s) +
                     (2 * (M * M) - 4 * M * mHNL + mHNL * mHNL - 2 * s) *
                     (2 * (M * M) + 4 * M * mHNL + mHNL * mHNL - 2 * s) *
                     (M * M + mHNL * mHNL - s) * t -
                     (8 * (M * M * M * M) + mHNL * mHNL * mHNL * mHNL - 3 *
                      (mHNL * mHNL) * s + 4 * (s * s) - 3 * (M * M) *
                      (3 * (mHNL * mHNL) + 4 * s)) *
                     (t * t)) + 4 * FFNCf1 * (M * M) *
                    (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                               (mHNL * mHNL - s) - 2 * (M * M) *
                               (mHNL * mHNL + s)) / (s * s)) *
                     (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                      (t * t)) + h * (mHNL * mHNL - 2 * t) *
                     (mHNL * mHNL * mHNL * mHNL - s * t + M * M *
                      (mHNL * mHNL + t) - mHNL * mHNL *
                      (s + t))) - 8 * FFNCga * h * (M * M) *
                    (2 * (M * M * M * M) * t + t *
                     (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                      (-s + t) + s * (2 * s + t)) + M * M *
                     (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                      (mHNL * mHNL) * t - t * (4 * s + t))))) + 4 * FFf1 *
                 (M * M) *
                 (-(FFNCf2 *
                    (mHNL * mHNL * mHNL * mHNL) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s))) + 4 * FFNCga * (M * M) * s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * t - FFNCf2 *
                  (mHNL * mHNL) * s * Sqrt(
                      (M * M * M * M + (mHNL * mHNL - s) *
                       (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                      (s * s)) * t + 2 * FFNCga * (mHNL * mHNL) * s * Sqrt(
                          (M * M * M * M + (mHNL * mHNL - s) *
                           (mHNL * mHNL - s) - 2 * (M * M) *
                           (mHNL * mHNL + s)) / (s * s)) * t - 4 * FFNCga *
                  (s * s) * Sqrt(
                      (M * M * M * M + (mHNL * mHNL - s) *
                       (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                      (s * s)) * t + 2 * FFNCf2 * s * Sqrt(
                          (M * M * M * M + (mHNL * mHNL - s) *
                           (mHNL * mHNL - s) - 2 * (M * M) *
                           (mHNL * mHNL + s)) / (s * s)) *
                  (t * t) - 2 * FFNCga * s * Sqrt(
                      (M * M * M * M + (mHNL * mHNL - s) *
                       (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                      (s * s)) * (t * t) - FFNCf2 * h * (mHNL * mHNL - 2 * t) *
                  (mHNL * mHNL * mHNL * mHNL - s * t + M * M *
                   (mHNL * mHNL + t) - mHNL * mHNL *
                   (s + t)) + 2 * FFNCga * h *
                  (2 * (M * M * M * M) * t + t *
                   (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL * (-s + t) + s *
                    (2 * s + t)) + M * M *
                   (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                    (mHNL * mHNL) * t - t * (4 * s + t))) + 2 * FFNCf1 *
                  (s * Sqrt(
                      (M * M * M * M + (mHNL * mHNL - s) *
                       (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                      (s * s)) * (2 * (M * M * M * M) - 4 * (M * M) * s + 2 *
                                  (s * s) + 2 * s * t + t * t - mHNL * mHNL *
                                  (2 * s + t)) + h *
                   (2 * (M * M * M * M * M * M) - 2 *
                    ((mHNL * mHNL - s) * (mHNL * mHNL - s)) * s - 2 *
                    (M * M * M * M) * (mHNL * mHNL + 3 * s) +
                    (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * s - 2 *
                     (s * s)) * t - (mHNL * mHNL + s) * (t * t) + M * M *
                    (6 * (s * s) + 2 * s * t + t * t + mHNL * mHNL *
                     (-2 * s + t)))))) * Vhad *
                (Cji * Vij + Cij * Vji)) / (M * M * s * Sqrt(
                    (M * M * M * M + (mHNL * mHNL - s) *
                     (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                    (s * s)) * (MZBOSON * MZBOSON - t) *
                                            (-(mzprime * mzprime) + t))

        # kinetic mixing + mass mixing interference
        def Lmunu_Hmunu_KinMix_MassMix_inter():
            return (Cprimehad *
                    (-(FFf2 *
                       (4 * FFNCf2 * (M * M) *
                        (mHNL * mHNL * mHNL * mHNL) * s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) + 4 * FFNCf2 *
                        (M * M * M * M) * s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * t - 16 * FFNCga *
                        (M * M * M * M) * s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * t + 4 * FFNCf2 *
                        (M * M) * (mHNL * mHNL) * s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * t - 8 * FFNCga *
                        (M * M) * (mHNL * mHNL) * s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * t + FFNCf2 *
                        (mHNL * mHNL * mHNL * mHNL) * s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * t - 8 * FFNCf2 *
                        (M * M) * (s * s) * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * t + 16 * FFNCga *
                        (M * M) * (s * s) * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * t - 4 * FFNCf2 *
                        (mHNL * mHNL) * (s * s) * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) /
                            (s * s)) * t + 4 * FFNCf2 * (s * s * s) * Sqrt(
                                (M * M * M * M + (mHNL * mHNL - s) *
                                 (mHNL * mHNL - s) - 2 * (M * M) *
                                 (mHNL * mHNL + s)) /
                                (s * s)) * t - 8 * FFNCf2 * (M * M) * s * Sqrt(
                                    (M * M * M * M + (mHNL * mHNL - s) *
                                     (mHNL * mHNL - s) - 2 * (M * M) *
                                     (mHNL * mHNL + s)) / (s * s)) *
                        (t * t) + 8 * FFNCga * (M * M) * s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) * (t * t) - FFNCf2 *
                        (mHNL * mHNL) * s * Sqrt(
                            (M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) /
                            (s * s)) * (t * t) + 4 * FFNCf2 * (s * s) * Sqrt(
                                (M * M * M * M + (mHNL * mHNL - s) *
                                 (mHNL * mHNL - s) - 2 * (M * M) *
                                 (mHNL * mHNL + s)) /
                                (s * s)) * (t * t) + FFNCf2 * h *
                        (4 * (M * M) * (mHNL * mHNL * mHNL * mHNL) *
                         (M * M + mHNL * mHNL - s) +
                         (2 * (M * M) - 4 * M * mHNL + mHNL * mHNL - 2 * s) *
                         (2 * (M * M) + 4 * M * mHNL + mHNL * mHNL - 2 * s) *
                         (M * M + mHNL * mHNL - s) * t -
                         (8 * (M * M * M * M) + mHNL * mHNL * mHNL * mHNL - 3 *
                          (mHNL * mHNL) * s + 4 * (s * s) - 3 * (M * M) *
                          (3 * (mHNL * mHNL) + 4 * s)) *
                         (t * t)) + 4 * FFNCf1 * (M * M) *
                        (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                                   (mHNL * mHNL - s) - 2 * (M * M) *
                                   (mHNL * mHNL + s)) / (s * s)) *
                         (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                          (t * t)) + h * (mHNL * mHNL - 2 * t) *
                         (mHNL * mHNL * mHNL * mHNL - s * t + M * M *
                          (mHNL * mHNL + t) - mHNL * mHNL *
                          (s + t))) - 8 * FFNCga * h * (M * M) *
                        (2 * (M * M * M * M) * t + t *
                         (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                          (-s + t) + s * (2 * s + t)) + M * M *
                         (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                          (mHNL * mHNL) * t - t * (4 * s + t))))) + 4 * FFf1 *
                     (M * M) *
                     (-(FFNCf2 * (mHNL * mHNL * mHNL * mHNL) * s * Sqrt(
                         (M * M * M * M + (mHNL * mHNL - s) *
                          (mHNL * mHNL - s) - 2 * (M * M) *
                          (mHNL * mHNL + s)) / (s * s))) + 4 * FFNCga *
                      (M * M) * s * Sqrt(
                          (M * M * M * M + (mHNL * mHNL - s) *
                           (mHNL * mHNL - s) - 2 * (M * M) *
                           (mHNL * mHNL + s)) /
                          (s * s)) * t - FFNCf2 * (mHNL * mHNL) * s * Sqrt(
                              (M * M * M * M + (mHNL * mHNL - s) *
                               (mHNL * mHNL - s) - 2 * (M * M) *
                               (mHNL * mHNL + s)) / (s * s)) * t + 2 * FFNCga *
                      (mHNL * mHNL) * s * Sqrt(
                          (M * M * M * M + (mHNL * mHNL - s) *
                           (mHNL * mHNL - s) - 2 * (M * M) *
                           (mHNL * mHNL + s)) /
                          (s * s)) * t - 4 * FFNCga * (s * s) * Sqrt(
                              (M * M * M * M + (mHNL * mHNL - s) *
                               (mHNL * mHNL - s) - 2 * (M * M) *
                               (mHNL * mHNL + s)) /
                              (s * s)) * t + 2 * FFNCf2 * s * Sqrt(
                                  (M * M * M * M + (mHNL * mHNL - s) *
                                   (mHNL * mHNL - s) - 2 * (M * M) *
                                   (mHNL * mHNL + s)) /
                                  (s * s)) * (t * t) - 2 * FFNCga * s * Sqrt(
                                      (M * M * M * M + (mHNL * mHNL - s) *
                                       (mHNL * mHNL - s) - 2 * (M * M) *
                                       (mHNL * mHNL + s)) / (s * s)) *
                      (t * t) - FFNCf2 * h * (mHNL * mHNL - 2 * t) *
                      (mHNL * mHNL * mHNL * mHNL - s * t + M * M *
                       (mHNL * mHNL + t) - mHNL * mHNL *
                       (s + t)) + 2 * FFNCga * h *
                      (2 * (M * M * M * M) * t + t *
                       (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                        (-s + t) + s * (2 * s + t)) + M * M *
                       (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                        (mHNL * mHNL) * t - t * (4 * s + t))) + 2 * FFNCf1 *
                      (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                                 (mHNL * mHNL - s) - 2 * (M * M) *
                                 (mHNL * mHNL + s)) / (s * s)) *
                       (2 * (M * M * M * M) - 4 * (M * M) * s + 2 *
                        (s * s) + 2 * s * t + t * t - mHNL * mHNL *
                        (2 * s + t)) + h *
                       (2 * (M * M * M * M * M * M) - 2 *
                        ((mHNL * mHNL - s) *
                         (mHNL * mHNL - s)) * s - 2 * (M * M * M * M) *
                        (mHNL * mHNL + 3 * s) +
                        (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * s - 2 *
                         (s * s)) * t - (mHNL * mHNL + s) * (t * t) + M * M *
                        (6 * (s * s) + 2 * s * t + t * t + mHNL * mHNL *
                         (-2 * s + t)))))) * Vhad * Vij *
                    Vji) / (2. * (M * M) * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * ((mzprime * mzprime - t) *
                                    (mzprime * mzprime - t)))

        # mass mixing term SQR
        def Lmunu_Hmunu_MassMix_SQR():
            return (
                Cprimehad * Cprimehad *
                (-4 * (FFNCf2 * FFNCf2) * h * (M * M * M * M) *
                 (mHNL * mHNL * mHNL * mHNL) - 4 * (FFNCf2 * FFNCf2) * h *
                 (M * M) * (mHNL * mHNL * mHNL * mHNL * mHNL * mHNL) + 4 *
                 (FFNCf2 * FFNCf2) * h * (M * M) *
                 (mHNL * mHNL * mHNL * mHNL) * s - 4 * (FFNCf2 * FFNCf2) *
                 (M * M) * (mHNL * mHNL * mHNL * mHNL) * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) - 4 * (FFNCf2 * FFNCf2) * h *
                 (M * M * M * M * M * M) * t + 8 * (FFNCf2 * FFNCf2) * h *
                 (M * M * M * M) * (mHNL * mHNL) * t + 11 *
                 (FFNCf2 * FFNCf2) * h * (M * M) *
                 (mHNL * mHNL * mHNL * mHNL) * t - 4 * (FFNCgp * FFNCgp) * h *
                 (M * M) *
                 (mHNL * mHNL * mHNL * mHNL) * t - FFNCf2 * FFNCf2 * h *
                 (mHNL * mHNL * mHNL * mHNL * mHNL * mHNL) * t - 4 *
                 (FFNCgp * FFNCgp) * h *
                 (mHNL * mHNL * mHNL * mHNL * mHNL * mHNL) * t + 12 *
                 (FFNCf2 * FFNCf2) * h * (M * M * M * M) * s * t + 5 *
                 (FFNCf2 * FFNCf2) * h *
                 (mHNL * mHNL * mHNL * mHNL) * s * t + 4 *
                 (FFNCgp * FFNCgp) * h *
                 (mHNL * mHNL * mHNL * mHNL) * s * t - 12 *
                 (FFNCf2 * FFNCf2) * h * (M * M) * (s * s) * t - 8 *
                 (FFNCf2 * FFNCf2) * h * (mHNL * mHNL) * (s * s) * t + 4 *
                 (FFNCf2 * FFNCf2) * h * (s * s * s) * t - 4 *
                 (FFNCf2 * FFNCf2) * (M * M * M * M) * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * t - 4 * (FFNCf2 * FFNCf2) * (M * M) *
                 (mHNL * mHNL) * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * t - FFNCf2 * FFNCf2 *
                 (mHNL * mHNL * mHNL * mHNL) * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * t - 4 * (FFNCgp * FFNCgp) *
                 (mHNL * mHNL * mHNL * mHNL) * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * t + 8 * (FFNCf2 * FFNCf2) * (M * M) *
                 (s * s) * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                                 (mHNL * mHNL - s) - 2 * (M * M) *
                                 (mHNL * mHNL + s)) / (s * s)) * t + 4 *
                 (FFNCf2 * FFNCf2) * (mHNL * mHNL) * (s * s) * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * t - 4 * (FFNCf2 * FFNCf2) *
                 (s * s * s) * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                                     (mHNL * mHNL - s) - 2 * (M * M) *
                                     (mHNL * mHNL + s)) / (s * s)) * t + 8 *
                 (FFNCf2 * FFNCf2) * h * (M * M * M * M) * (t * t) - 9 *
                 (FFNCf2 * FFNCf2) * h * (M * M) * (mHNL * mHNL) *
                 (t * t) - 4 * (FFNCgp * FFNCgp) * h * (M * M) *
                 (mHNL * mHNL) * (t * t) + FFNCf2 * FFNCf2 * h *
                 (mHNL * mHNL * mHNL * mHNL) * (t * t) + 4 *
                 (FFNCgp * FFNCgp) * h * (mHNL * mHNL * mHNL * mHNL) *
                 (t * t) - 12 * (FFNCf2 * FFNCf2) * h * (M * M) * s *
                 (t * t) - 3 * (FFNCf2 * FFNCf2) * h * (mHNL * mHNL) * s *
                 (t * t) + 4 * (FFNCgp * FFNCgp) * h * (mHNL * mHNL) * s *
                 (t * t) + 4 * (FFNCf2 * FFNCf2) * h * (s * s) * (t * t) + 8 *
                 (FFNCf2 * FFNCf2) * (M * M) * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * (t * t) + FFNCf2 * FFNCf2 *
                 (mHNL * mHNL) * s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                                           (mHNL * mHNL - s) - 2 * (M * M) *
                                           (mHNL * mHNL + s)) / (s * s)) *
                 (t * t) + 4 * (FFNCgp * FFNCgp) * (mHNL * mHNL) * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * (t * t) - 4 * (FFNCf2 * FFNCf2) *
                 (s * s) * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * (t * t) + 8 * (FFNCf1 * FFNCf1) * (M * M) *
                 (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                            (mHNL * mHNL - s) - 2 * (M * M) *
                            (mHNL * mHNL + s)) / (s * s)) *
                  (2 * (M * M * M * M) - 4 * (M * M) * s + 2 *
                   (s * s) + 2 * s * t + t * t - mHNL * mHNL *
                   (2 * s + t)) + h *
                  (2 * (M * M * M * M * M * M) - 2 *
                   ((mHNL * mHNL - s) *
                    (mHNL * mHNL - s)) * s - 2 * (M * M * M * M) *
                   (mHNL * mHNL + 3 * s) +
                   (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * s - 2 *
                    (s * s)) * t - (mHNL * mHNL + s) * (t * t) + M * M *
                   (6 * (s * s) + 2 * s * t + t * t + mHNL * mHNL *
                    (-2 * s + t)))) - 8 * FFNCf1 * (M * M) *
                 (2 * FFNCga * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * t *
                  (-2 * (M * M) - mHNL * mHNL + 2 * s + t) + FFNCf2 * s * Sqrt(
                      (M * M * M * M + (mHNL * mHNL - s) *
                       (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                      (s * s)) *
                  (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                   (t * t)) + FFNCf2 * h * (mHNL * mHNL - 2 * t) *
                  (mHNL * mHNL * mHNL * mHNL - s * t + M * M *
                   (mHNL * mHNL + t) - mHNL * mHNL *
                   (s + t)) + 2 * FFNCga * h *
                  (-2 * (M * M * M * M) * t + t *
                   (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * (s - t) - s *
                    (2 * s + t)) + M * M *
                   (-4 * (mHNL * mHNL * mHNL * mHNL) + 3 *
                    (mHNL * mHNL) * t + t *
                    (4 * s + t)))) + 8 * (FFNCga * FFNCga) * (M * M) *
                 (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                            (mHNL * mHNL - s) - 2 * (M * M) *
                            (mHNL * mHNL + s)) / (s * s)) *
                  (2 * (M * M * M * M) + 2 * (s * s) + 4 * (M * M) *
                   (mHNL * mHNL - s - t) + 2 * s * t + t * t - mHNL * mHNL *
                   (2 * s + t)) + h *
                  (2 * (M * M * M * M * M * M) - 2 * ((mHNL * mHNL - s) *
                                                      (mHNL * mHNL - s)) * s +
                   (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * s - 2 *
                    (s * s)) * t - (mHNL * mHNL + s) * (t * t) - 2 *
                   (M * M * M * M) * (3 * (mHNL * mHNL + s) + 2 * t) + M * M *
                   (-4 * (mHNL * mHNL * mHNL * mHNL) + 6 *
                    (s * s) + 6 * s * t + t * t + mHNL * mHNL *
                    (2 * s + 5 * t)))) + 16 * FFNCga * (M * M) *
                 (FFNCgp * (mHNL * mHNL) *
                  (s * Sqrt(
                      (M * M * M * M + (mHNL * mHNL - s) *
                       (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                      (s * s)) * (-(mHNL * mHNL) + t) + h *
                   (-(mHNL * mHNL * mHNL * mHNL) + s * t - M * M *
                    (mHNL * mHNL + t) + mHNL * mHNL * (s + t))) + FFNCf2 *
                  (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) *
                   (2 * (M * M) + mHNL * mHNL - 2 * s - t) * t + h *
                   (2 * (M * M * M * M) * t + t *
                    (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                     (-s + t) + s * (2 * s + t)) + M * M *
                    (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                     (mHNL * mHNL) * t - t *
                     (4 * s + t)))))) * Vij * Vji) / (4. * (M * M) * s * Sqrt(
                         (M * M * M * M + (mHNL * mHNL - s) *
                          (mHNL * mHNL - s) - 2 * (M * M) *
                          (mHNL * mHNL + s)) /
                         (s * s)) * ((mzprime * mzprime - t) *
                                     (mzprime * mzprime - t)))

        # transition magnetic moment
        def Lmunu_Hmunu_TMM_SQR():
            return (mu_tr)**(2) * (eQED)**(2) * (M)**(-2) * (((M)**(4) + (
                (((mHNL)**(2) + -1 * s))**(2) + -2 * (M)**(2) *
                ((mHNL)**(2) + s))))**(-1 / 2) * (t)**(-2) * (
                    (FFf2)**(2) * t *
                    ((((M)**(4) + ((((mHNL)**(2) + -1 * s))**(2) + -2 *
                                   (M)**(2) * ((mHNL)**(2) + s))))**(1 / 2) *
                     (-4 * (M)**(2) * (mHNL)**(4) +
                      (4 * ((M)**(2) + -1 * s) * ((M)**(2) +
                                                  ((mHNL)**(2) + -1 * s)) * t +
                       (-1 * ((mHNL)**(2) + -4 * s) * (t)**(2) +
                        (t)**(3)))) + h *
                     (4 * (M)**(2) * (mHNL)**(4) * ((M)**(2) +
                                                    ((mHNL)**(2) + -1 * s)) +
                      (-4 * ((M)**(2) + -1 * s) *
                       (((M + -1 * mHNL))**(2) + -1 * s) *
                       (((M + mHNL))**(2) + -1 * s) * t +
                       (-1 * ((M)**(2) + ((mHNL)**(2) + -1 * s)) *
                        ((mHNL)**(2) + 4 * s) * (t)**(2) +
                        (-1 * (M)**(2) + ((mHNL)**(2) + s)) * (t)**(3))))) +
                    (8 * FFf1 * FFf2 * (M)**(2) * t *
                     ((((M)**(4) + ((((mHNL)**(2) + -1 * s))**(2) + -2 *
                                    (M)**(2) * ((mHNL)**(2) + s))))**(1 / 2) *
                      (-2 * (mHNL)**(4) + ((mHNL)**(2) * t + (t)**(2))) + h *
                      (2 * (mHNL)**(2) + -1 * t) *
                      ((mHNL)**(4) +
                       (-1 * s * t +
                        ((M)**(2) * ((mHNL)**(2) + t) + -1 * (mHNL)**(2) *
                         (s + t))))) + 8 * (FFf1)**(2) * (M)**(2) *
                     (h * (2 * (M)**(2) * (mHNL)**(4) *
                           ((M)**(2) + ((mHNL)**(2) + -1 * s)) +
                           (((M)**(2) + ((mHNL)**(2) + -1 * s)) *
                            (2 * (M)**(4) + ((mHNL)**(4) +
                                             (-2 * (mHNL)**(2) * s +
                                              (2 * (s)**(2) + -4 * (M)**(2) *
                                               ((mHNL)**(2) + s))))) * t +
                            (-2 * (M)**(4) +
                             (-1 * (mHNL)**(4) +
                              ((mHNL)**(2) * s +
                               (-2 * (s)**(2) + (M)**(2) *
                                (3 * (mHNL)**(2) + 4 * s))))) * (t)**(2))) +
                      (((M)**(4) + ((((mHNL)**(2) + -1 * s))**(2) + -2 *
                                    (M)**(2) * ((mHNL)**(2) + s))))**(1 / 2) *
                      (-2 * (M)**(4) * t +
                       (t * (-1 * (mHNL)**(4) +
                             (-2 * s * (s + t) + (mHNL)**(2) *
                              (2 * s + t))) + 2 * (M)**(2) *
                        (-1 * (mHNL)**(4) + t * (2 * s + t))))))) * (Z)**(2)

        # scalar term SQR
        def Lmunu_Hmunu_Scalar_SQR():
            return (FFscalar * FFscalar * (Shad * Shad) * Sij * Sji *
                    (4 * (M * M) - t) *
                    (s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * (mHNL * mHNL - t) + h *
                     (mHNL * mHNL * mHNL * mHNL - s * t + M * M *
                      (mHNL * mHNL + t) - mHNL * mHNL * (s + t)))) / (s * Sqrt(
                          (M * M * M * M + (mHNL * mHNL - s) *
                           (mHNL * mHNL - s) - 2 * (M * M) *
                           (mHNL * mHNL + s)) /
                          (s * s)) * ((MSCALAR * MSCALAR - t) *
                                      (MSCALAR * MSCALAR - t)))

        # scalar + SM NC term
        def Lmunu_Hmunu_Scalar_NC_inter():
            return (Chad * Cij * FFscalar * mHNL * Shad * Sij *
                    (4 * FFNCf1 * (M * M) + FFNCf2 * t) *
                    (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                               (mHNL * mHNL - s) - 2 * (M * M) *
                               (mHNL * mHNL + s)) / (s * s)) *
                     (2 * (M * M) + mHNL * mHNL - 2 * s - t) + h *
                     (2 * (M * M * M * M) + mHNL * mHNL * mHNL * mHNL + M * M *
                      (-(mHNL * mHNL) - 4 * s + t) - mHNL * mHNL *
                      (3 * s + t) + s *
                      (2 * s + 3 * t)))) / (M * s * Sqrt(
                          (M * M * M * M + (mHNL * mHNL - s) *
                           (mHNL * mHNL - s) - 2 * (M * M) *
                           (mHNL * mHNL + s)) / (s * s)) *
                                            (MSCALAR * MSCALAR - t) *
                                            (-(MZBOSON * MZBOSON) + t))

        # scalar + kinetic mixing interference
        def Lmunu_Hmunu_Scalar_KinMix_inter():
            return (FFscalar * mHNL * Shad * Sij * (4 * FFf1 *
                                                    (M * M) + FFf2 * t) *
                    (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                               (mHNL * mHNL - s) - 2 * (M * M) *
                               (mHNL * mHNL + s)) / (s * s)) *
                     (2 * (M * M) + mHNL * mHNL - 2 * s - t) + h *
                     (2 * (M * M * M * M) + mHNL * mHNL * mHNL * mHNL + M * M *
                      (-(mHNL * mHNL) - 4 * s + t) - mHNL * mHNL *
                      (3 * s + t) + s * (2 * s + 3 * t))) * Vhad *
                    Vij) / (M * s * Sqrt(
                        (M * M * M * M + (mHNL * mHNL - s) *
                         (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                        (s * s)) * (MSCALAR * MSCALAR - t) *
                            (-(mzprime * mzprime) + t))

        # scalar + mass mixing interference
        def Lmunu_Hmunu_Scalar_MassMix_inter():
            return (Cprimehad * Cij * FFscalar * mHNL * Shad * Sij *
                    (4 * FFNCf1 * (M * M) + FFNCf2 * t) *
                    (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                               (mHNL * mHNL - s) - 2 * (M * M) *
                               (mHNL * mHNL + s)) / (s * s)) *
                     (2 * (M * M) + mHNL * mHNL - 2 * s - t) + h *
                     (2 * (M * M * M * M) + mHNL * mHNL * mHNL * mHNL + M * M *
                      (-(mHNL * mHNL) - 4 * s + t) - mHNL * mHNL *
                      (3 * s + t) + s *
                      (2 * s + 3 * t)))) / (M * s * Sqrt(
                          (M * M * M * M + (mHNL * mHNL - s) *
                           (mHNL * mHNL - s) - 2 * (M * M) *
                           (mHNL * mHNL + s)) / (s * s)) *
                                            (MSCALAR * MSCALAR - t) *
                                            (-(MZBOSON * MZBOSON) + t))
    else:
        logger.error(
            f"Error! Could not find HNL type '{process.TheoryModel.HNLtype}'.")
        raise ValueError

    Lmunu_Hmunu = {}
    Lmunu_Hmunu['NC_SQR'] = Lmunu_Hmunu_NC_SQR
    if process.TheoryModel.is_kinetically_mixed:
        Lmunu_Hmunu['KinMix_SQR'] = Lmunu_Hmunu_KinMix_SQR
        Lmunu_Hmunu['KinMix_NC_inter'] = Lmunu_Hmunu_KinMix_NC_inter
    if process.TheoryModel.is_mass_mixed:
        Lmunu_Hmunu['MassMix_SQR'] = Lmunu_Hmunu_MassMix_SQR
        Lmunu_Hmunu['MassMix_NC_inter'] = Lmunu_Hmunu_MassMix_NC_inter
        if process.TheoryModel.is_kinetically_mixed:
            Lmunu_Hmunu[
                'KinMix_MassMix_inter'] = Lmunu_Hmunu_KinMix_MassMix_inter
    if process.TheoryModel.is_TMM:
        Lmunu_Hmunu['TMM_SQR'] = Lmunu_Hmunu_TMM_SQR
    if process.TheoryModel.is_scalar_mixed:
        Lmunu_Hmunu['Scalar_SQR'] = Lmunu_Hmunu_Scalar_SQR
        Lmunu_Hmunu['Scalar_NC_inter'] = Lmunu_Hmunu_Scalar_NC_inter
        if process.TheoryModel.is_kinetically_mixed:
            Lmunu_Hmunu[
                'Scalar_KinMix_inter'] = Lmunu_Hmunu_Scalar_KinMix_inter
        if process.TheoryModel.is_mass_mixed:
            Lmunu_Hmunu[
                'Scalar_MassMix_inter'] = Lmunu_Hmunu_Scalar_MassMix_inter

    # phase space factors
    phase_space = const.kallen_sqrt(1.0, M**2 / s,
                                    mHNL**2 / s) / (32 * np.pi**2)
    phase_space *= 2 * np.pi  # integrated over phi

    # flux factor in cross section
    flux_factor = 1.0 / (s - M**2) / 2

    E1CM = (s - M**2) / 2.0 / np.sqrt(s)
    E3CM = (s + mHNL**2 - M**2) / 2.0 / np.sqrt(s)
    p1CM = E1CM  # always assuming massless projectile
    p3CM = np.sqrt(E3CM**2 - mHNL**2)

    # jacobian -- from angle to Q2
    physical_jacobian = 1.0 / 2.0 / p1CM / p3CM

    # hadronic spin average
    spin_average = 1 / 2

    # final prefactor: dsigma = prefactor*LmunuHmunu
    prefactor = flux_factor * phase_space * physical_jacobian * spin_average * invGeV2_to_attobarn

    # from amplitude to diff xsec:
    diff_xsec_terms = {}
    diff_xsec_terms['total'] = 0.0
    for diagram, lmnhmn in Lmunu_Hmunu.items():
        if (diagram in diagrams) or ('all' in diagrams) or ('total'
                                                            in diagrams):
            # all diff xsec terms
            diff_xsec_terms[diagram] = lmnhmn() * prefactor
            # summing all contributions (Z,Z',S,interferences,etc)
            diff_xsec_terms['total'] += diff_xsec_terms[diagram]

    # raise warning for any requested diagram not picked up here and setting to zero
    for missing_diagram in list(set(diagrams) - set(diff_xsec_terms.keys())):
        logger.warning(
            f'Warning: Diagram not found. Either not implemented or misspelled. Setting amplitude it to zero: {missing_diagram}'
        )
        diff_xsec_terms[missing_diagram] = prefactor * 0.0

    if 'all' in diagrams:
        # return all individual diagrams in a dictionary
        return diff_xsec_terms
    else:
        # return the sum of all diagrams requested
        return diff_xsec_terms['total']


def trident_dxsec_dQ2(x_phase_space, process):
    '''
        Return the differential cross section for trident scattering in attobarns

    '''

    # kinematics
    x1 = x_phase_space[0]
    x2 = x_phase_space[1]
    x3 = x_phase_space[2]
    x4 = x_phase_space[3]
    x5 = x_phase_space[4]
    x6 = x_phase_space[5]
    x7 = x_phase_space[6]
    x8 = x_phase_space[7]
    Enu = x_phase_space[8]

    # hadronic target
    target = process.target

    # masses
    M = process.target.mass
    Z = target.Z
    mzprime = process.mzprime

    # Z' propagator and vertices
    prop = 1.0 / (2.0 * m6 + mzprime**2)
    V2 = (process.Vijk + process.gprimeV**2 / 2.0 / sqrt(2.0) / const.Gf *
          process.CHARGE * prop)**2
    A2 = (process.Aijk + process.gprimeA**2 / 2.0 / sqrt(2.0) / const.Gf *
          process.CHARGE * prop)**2
    VA = (process.Aijk + process.gprimeA**2 / 2.0 / sqrt(2.0) / const.Gf *
          process.CHARGE * prop) * (process.Vijk +
                                    process.gprimeV**2 / 2.0 / sqrt(2.0) /
                                    const.Gf * process.CHARGE * prop)

    # Form factors
    if target.is_nucleus:
        FFf1 = target.F1_EM(Q2)
        FFf2 = 0.0  ### FIX ME
        FFga = 0.0
        FFgp = 0.0

        FFNCf1 = target.F1_NC(Q2)
        FFNCf2 = 0.0
        FFNCga = 0.0
        FFNCgp = 0.0

    elif (target.is_nucleon):
        FFf1 = target.F1_EM(Q2)
        FFf2 = target.F2_EM(Q2)
        FFga = 0.0
        FFgp = 0.0

        FFNCf1 = target.F1_NC(Q2)
        FFNCf2 = target.F2_NC(Q2)
        FFNCga = target.F3_NC(Q2)
        FFNCgp = 0.0

    else:
        logger.error('upscattering on a lepton not implemented.')

    ## Spin summed (but not averaged) matrix elements from MATHEMATICA
    # |M|^2 = | M_SM + M_kinmix + M_massmix|^2
    Lmunu_Hmunu = 0.0
    if process.TheoryModel.HNLtype == 'majorana':
        # SM NC SQR
        Lmunu_Hmunu += (
            Chad * Chad * Cij * Cji *
            (16 * FFNCga * FFNCgp * (mHNL * mHNL) * (-(mHNL * mHNL) + t) +
             (4 * (FFNCgp * FFNCgp) * (mHNL * mHNL) * t *
              (-(mHNL * mHNL) + t)) / (M * M) - 8 * FFNCf1 * FFNCf2 *
             (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 * (t * t)) + 8 *
             (FFNCf1 * FFNCf1) * (2 * (M * M * M * M) - 4 * (M * M) * s + 2 *
                                  (s * s) + 2 * s * t + t * t - mHNL * mHNL *
                                  (2 * s + t)) + 8 * (FFNCga * FFNCga) *
             (2 * (M * M * M * M) + 2 * (s * s) + 4 * (M * M) *
              (mHNL * mHNL - s - t) + 2 * s * t + t * t - mHNL * mHNL *
              (2 * s + t)) -
             (FFNCf2 * FFNCf2 *
              (4 * (M * M * M * M) * t + 4 * (M * M) *
               (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 * t *
                (s + t)) + t * (mHNL * mHNL * mHNL * mHNL + 4 * s *
                                (s + t) - mHNL * mHNL * (4 * s + t)))) /
             (M * M) + (16 * FFNCf1 * FFNCga * h *
                        (2 * (M * M * M * M) * t + t *
                         (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                          (-s + t) + s * (2 * s + t)) + M * M *
                         (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                          (mHNL * mHNL) * t - t * (4 * s + t)))) / (s * Sqrt(
                              (M * M * M * M + (mHNL * mHNL - s) *
                               (mHNL * mHNL - s) - 2 * (M * M) *
                               (mHNL * mHNL + s)) / (s * s))) +
             (16 * FFNCf2 * FFNCga * h *
              (2 * (M * M * M * M) * t + t *
               (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL * (-s + t) + s *
                (2 * s + t)) + M * M *
               (4 * (mHNL * mHNL * mHNL * mHNL) - 3 * (mHNL * mHNL) * t - t *
                (4 * s + t)))) / (s * Sqrt(
                    (M * M * M * M + (mHNL * mHNL - s) *
                     (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                    (s * s))))) / (2. * ((MZBOSON * MZBOSON - t) *
                                         (MZBOSON * MZBOSON - t)))

        if process.TheoryModel.is_kinetically_mixed:

            # kinetic mixing term SQR
            Lmunu_Hmunu += (
                (-8 * FFf1 * FFf2 * (M * M) *
                 (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                  (t * t)) + 8 * (FFf1 * FFf1) * (M * M) *
                 (2 * (M * M * M * M) - 4 * (M * M) * s + 2 *
                  (s * s) + 2 * s * t + t * t - mHNL * mHNL *
                  (2 * s + t)) - FFf2 * FFf2 *
                 (4 * (M * M * M * M) * t + 4 * (M * M) *
                  (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 * t *
                   (s + t)) + t * (mHNL * mHNL * mHNL * mHNL + 4 * s *
                                   (s + t) - mHNL * mHNL * (4 * s + t)))) *
                (Vhad * Vhad) * Vij * Vji) / (2. * (M * M) *
                                              ((mzprime * mzprime - t) *
                                               (mzprime * mzprime - t)))

            # # kinetic mixing + SM NC interference
            Lmunu_Hmunu += (
                (Chad *
                 (4 * FFf1 * (M * M) *
                  (s * Sqrt((M * M * M * M + (mHNL * mHNL - s) *
                             (mHNL * mHNL - s) - 2 * (M * M) *
                             (mHNL * mHNL + s)) / (s * s)) *
                   (-(FFNCf2 *
                      (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                       (t * t))) + 2 * FFNCf1 *
                    (2 * (M * M * M * M) - 4 * (M * M) * s + 2 *
                     (s * s) + 2 * s * t + t * t - mHNL * mHNL *
                     (2 * s + t))) + 2 * FFNCga * h *
                   (2 * (M * M * M * M) * t + t *
                    (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                     (-s + t) + s * (2 * s + t)) + M * M *
                    (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                     (mHNL * mHNL) * t - t * (4 * s + t)))) + FFf2 *
                  (8 * FFNCga * h * (M * M) *
                   (2 * (M * M * M * M) * t + t *
                    (-(mHNL * mHNL * mHNL * mHNL) + mHNL * mHNL *
                     (-s + t) + s * (2 * s + t)) + M * M *
                    (4 * (mHNL * mHNL * mHNL * mHNL) - 3 *
                     (mHNL * mHNL) * t - t * (4 * s + t))) - s * Sqrt(
                         (M * M * M * M + (mHNL * mHNL - s) *
                          (mHNL * mHNL - s) - 2 * (M * M) *
                          (mHNL * mHNL + s)) / (s * s)) *
                   (4 * FFNCf1 * (M * M) *
                    (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 *
                     (t * t)) + FFNCf2 *
                    (4 * (M * M * M * M) * t + 4 * (M * M) *
                     (mHNL * mHNL * mHNL * mHNL + mHNL * mHNL * t - 2 * t *
                      (s + t)) + t * (mHNL * mHNL * mHNL * mHNL + 4 * s *
                                      (s + t) - mHNL * mHNL *
                                      (4 * s + t)))))) * Vhad *
                 (Cji * Vij + Cij * Vji)) / (M * M * s * Sqrt(
                     (M * M * M * M + (mHNL * mHNL - s) *
                      (mHNL * mHNL - s) - 2 * (M * M) * (mHNL * mHNL + s)) /
                     (s * s)) * (MZBOSON * MZBOSON - t) *
                                             (-(mzprime * mzprime) + t)))

    # phase space factors
    phase_space = const.kallen_sqrt(1.0, M**2 / s,
                                    mHNL**2 / s) / (32 * np.pi**2)
    phase_space *= 2 * np.pi  # integrated over phi

    # flux factor in cross section
    flux_factor = 1.0 / (s - M**2) / 2

    E1CM = (s - M**2) / 2.0 / np.sqrt(s)
    E3CM = (s + mHNL**2 - M**2) / 2.0 / np.sqrt(s)
    p1CM = E1CM  # massless projectile
    p3CM = np.sqrt(E3CM**2 - mHNL**2)

    # jacobian -- from angle to Q2
    physical_jacobian = 1.0 / 2.0 / p1CM / p3CM

    # hadronic spin average
    spin_average = 1 / 2

    # differential cross section
    diff_xsec = flux_factor * Lmunu_Hmunu * phase_space * physical_jacobian * spin_average

    ####################
    dsigma = (
        8 * (alphaQED * alphaQED) * (FormFactor * FormFactor) * (Gf * Gf) *
        (Diag22 * ((x1 + 2 * x4) * (x1 + 2 * x4)) *
         (A2 *
          (-2 * (ml2 * ml2) * Mn * (x2 * x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) *
           (-(ml1 * ml1) + ml2 * ml2 - x1 + 2 * x2 - 2 * x3 - 2 * x5) - 2 * Mn *
           (x2 * x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                             (x2 * x2)) * x3 *
           (-(ml1 * ml1) + ml2 * ml2 - x1 + 2 * x2 - 2 * x3 - 2 * x5) + 2 *
           (ml2 * ml2) * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) *
           (-(ml1 * ml1) + ml2 * ml2 - x1 + 2 * x2 - 2 * x3 - 2 * x5) * x5 -
           x1 * x1 * (x2 * x2) * ((-2 * Enu * Mn + x2) *
                                  (-2 * Enu * Mn + x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) +
           3 * Mn * x1 * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) + 2 *
           (x1 * x1) * x2 * ((-2 * Enu * Mn + x2) *
                             (-2 * Enu * Mn + x2)) * (x5 * x5) *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) -
           4 * Mn * x1 * x2 * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                               (x2 * x2)) * (x5 * x5) *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) - x1 * x1 *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * (x5 * x5 * x5) *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) -
           2 * Mn * x1 * (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
                          (x2 * x2)) * (x5 * x5 * x5) *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) - 4 * ml1 *
           (ml2 * ml2 * ml2) * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (x2 - x5 - x6) + 4 * ml1 * ml2 * Mn * x1 * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (x2 - x5 - x6) + 4 * ml1 * ml2 * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * x3 * (x2 - x5 - x6) + 2 * ml1 * ml2 * (x1 * x1) * x2 *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x5 *
           (x2 - x5 - x6) - 4 * ml1 * ml2 * Mn * x1 * x2 *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * x5 * (x2 - x5 - x6) - 2 * ml1 * ml2 * (x1 * x1) *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * (x5 * x5) *
           (x2 - x5 - x6) - 4 * ml1 * ml2 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
            (x2 * x2)) * (x5 * x5) * (x2 - x5 - x6) - 4 * (ml2 * ml2) * Mn *
           (x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                        (x2 * x2)) * (-x1 + x2 - x3 - x4) * x6 - 4 * Mn *
           (x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                        (x2 * x2)) * x3 * (-x1 + x2 - x3 - x4) * x6 - 2 *
           (x1 * x1) * x2 * ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) *
           (x1 - x2 + x3 + x4) * x5 * x6 + 4 * Mn * x1 * x2 *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (x1 - x2 + x3 + x4) * x5 * x6 + 2 * (x1 * x1) *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) *
           (x1 - x2 + x3 + x4) * (x5 * x5) * x6 + 4 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
            (x2 * x2)) * (x1 - x2 + x3 + x4) * (x5 * x5) * x6 + x1 * x1 * x1 *
           ((-2 * Enu * Mn + x2) *
            (-2 * Enu * Mn + x2)) * x5 * (x2 - x5 - x6) * x6 + 2 * Mn *
           (x1 * x1) * (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
                        (x2 * x2)) * x5 * (x2 - x5 - x6) * x6 + 2 * (x1 * x1) *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x3 * x5 *
           (x2 - x5 - x6) * x6 + 4 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
            (x2 * x2)) * x3 * x5 * (x2 - x5 - x6) * x6 - 2 * (ml2 * ml2) * Mn *
           (x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                        (x2 * x2)) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) +
           Mn * x1 * (x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                                  (x2 * x2)) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) +
           x1 * x1 * x2 * ((-2 * Enu * Mn + x2) *
                           (-2 * Enu * Mn + x2)) * x5 * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) -
           2 * Mn * x1 * x2 * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                               (x2 * x2)) * x5 * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) -
           x1 * x1 * ((-2 * Enu * Mn + x2) *
                      (-2 * Enu * Mn + x2)) * (x5 * x5) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) -
           2 * Mn * x1 * (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
                          (x2 * x2)) * (x5 * x5) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)) + V2 *
          (-2 * (ml2 * ml2) * Mn * (x2 * x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) *
           (-(ml1 * ml1) + ml2 * ml2 - x1 + 2 * x2 - 2 * x3 - 2 * x5) - 2 * Mn *
           (x2 * x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                             (x2 * x2)) * x3 *
           (-(ml1 * ml1) + ml2 * ml2 - x1 + 2 * x2 - 2 * x3 - 2 * x5) + 2 *
           (ml2 * ml2) * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) *
           (-(ml1 * ml1) + ml2 * ml2 - x1 + 2 * x2 - 2 * x3 - 2 * x5) * x5 -
           x1 * x1 * (x2 * x2) * ((-2 * Enu * Mn + x2) *
                                  (-2 * Enu * Mn + x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) +
           3 * Mn * x1 * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) + 2 *
           (x1 * x1) * x2 * ((-2 * Enu * Mn + x2) *
                             (-2 * Enu * Mn + x2)) * (x5 * x5) *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) -
           4 * Mn * x1 * x2 * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                               (x2 * x2)) * (x5 * x5) *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) - x1 * x1 *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * (x5 * x5 * x5) *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) -
           2 * Mn * x1 * (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
                          (x2 * x2)) * (x5 * x5 * x5) *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) + 4 * ml1 *
           (ml2 * ml2 * ml2) * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (x2 - x5 - x6) - 4 * ml1 * ml2 * Mn * x1 * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (x2 - x5 - x6) - 4 * ml1 * ml2 * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * x3 * (x2 - x5 - x6) - 2 * ml1 * ml2 * (x1 * x1) * x2 *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x5 *
           (x2 - x5 - x6) + 4 * ml1 * ml2 * Mn * x1 * x2 *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * x5 * (x2 - x5 - x6) + 2 * ml1 * ml2 * (x1 * x1) *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * (x5 * x5) *
           (x2 - x5 - x6) + 4 * ml1 * ml2 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
            (x2 * x2)) * (x5 * x5) * (x2 - x5 - x6) - 4 * (ml2 * ml2) * Mn *
           (x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                        (x2 * x2)) * (-x1 + x2 - x3 - x4) * x6 - 4 * Mn *
           (x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                        (x2 * x2)) * x3 * (-x1 + x2 - x3 - x4) * x6 - 2 *
           (x1 * x1) * x2 * ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) *
           (x1 - x2 + x3 + x4) * x5 * x6 + 4 * Mn * x1 * x2 *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (x1 - x2 + x3 + x4) * x5 * x6 + 2 * (x1 * x1) *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) *
           (x1 - x2 + x3 + x4) * (x5 * x5) * x6 + 4 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
            (x2 * x2)) * (x1 - x2 + x3 + x4) * (x5 * x5) * x6 + x1 * x1 * x1 *
           ((-2 * Enu * Mn + x2) *
            (-2 * Enu * Mn + x2)) * x5 * (x2 - x5 - x6) * x6 + 2 * Mn *
           (x1 * x1) * (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
                        (x2 * x2)) * x5 * (x2 - x5 - x6) * x6 + 2 * (x1 * x1) *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x3 * x5 *
           (x2 - x5 - x6) * x6 + 4 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
            (x2 * x2)) * x3 * x5 * (x2 - x5 - x6) * x6 - 2 * (ml2 * ml2) * Mn *
           (x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                        (x2 * x2)) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) +
           Mn * x1 * (x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                                  (x2 * x2)) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) +
           x1 * x1 * x2 * ((-2 * Enu * Mn + x2) *
                           (-2 * Enu * Mn + x2)) * x5 * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) -
           2 * Mn * x1 * x2 * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                               (x2 * x2)) * x5 * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) -
           x1 * x1 * ((-2 * Enu * Mn + x2) *
                      (-2 * Enu * Mn + x2)) * (x5 * x5) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) -
           2 * Mn * x1 * (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
                          (x2 * x2)) * (x5 * x5) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)) +
          2 * VA *
          (-2 * (ml2 * ml2) * Mn * (x2 * x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) *
           (-(ml1 * ml1) + ml2 * ml2 - x1 + 2 * x2 - 2 * x3 - 2 * x5) - 2 * Mn *
           (x2 * x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                             (x2 * x2)) * x3 *
           (-(ml1 * ml1) + ml2 * ml2 - x1 + 2 * x2 - 2 * x3 - 2 * x5) + 2 *
           (ml2 * ml2) * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) *
           (-(ml1 * ml1) + ml2 * ml2 - x1 + 2 * x2 - 2 * x3 - 2 * x5) * x5 -
           x1 * x1 * (x2 * x2) * ((-2 * Enu * Mn + x2) *
                                  (-2 * Enu * Mn + x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) +
           3 * Mn * x1 * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) + 2 *
           (x1 * x1) * x2 * ((-2 * Enu * Mn + x2) *
                             (-2 * Enu * Mn + x2)) * (x5 * x5) *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) -
           4 * Mn * x1 * x2 * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                               (x2 * x2)) * (x5 * x5) *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) - x1 * x1 *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * (x5 * x5 * x5) *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) -
           2 * Mn * x1 * (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
                          (x2 * x2)) * (x5 * x5 * x5) *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) + 4 *
           (ml2 * ml2) * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (-x1 + x2 - x3 - x4) * x6 + 4 * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * x3 * (-x1 + x2 - x3 - x4) * x6 + 2 * (x1 * x1) * x2 *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) *
           (x1 - x2 + x3 + x4) * x5 * x6 - 4 * Mn * x1 * x2 *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (x1 - x2 + x3 + x4) * x5 * x6 - 2 * (x1 * x1) *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) *
           (x1 - x2 + x3 + x4) * (x5 * x5) * x6 - 4 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
            (x2 * x2)) * (x1 - x2 + x3 + x4) * (x5 * x5) * x6 - x1 * x1 * x1 *
           ((-2 * Enu * Mn + x2) *
            (-2 * Enu * Mn + x2)) * x5 * (x2 - x5 - x6) * x6 - 2 * Mn *
           (x1 * x1) * (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
                        (x2 * x2)) * x5 * (x2 - x5 - x6) * x6 - 2 * (x1 * x1) *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x3 * x5 *
           (x2 - x5 - x6) * x6 - 4 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
            (x2 * x2)) * x3 * x5 * (x2 - x5 - x6) * x6 + Mn * x1 * (x2 * x2) *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn * (x2 * x2)) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) + 2 *
           (ml2 * ml2) * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) -
           x1 * x1 * x2 * ((-2 * Enu * Mn + x2) *
                           (-2 * Enu * Mn + x2)) * x5 * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) +
           2 * Mn * x1 * x2 * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                               (x2 * x2)) * x5 * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) +
           x1 * x1 * ((-2 * Enu * Mn + x2) *
                      (-2 * Enu * Mn + x2)) * (x5 * x5) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) +
           2 * Mn * x1 * (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
                          (x2 * x2)) * (x5 * x5) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6))) +
         Diag11 * ((x1 + 2 * x3) * (x1 + 2 * x3)) *
         (A2 *
          (-4 * (ml1 * ml1) * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (-x1 + x2 - x3 - x4) * x5 - 4 * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) *
           (-x1 + x2 - x3 - x4) * x4 * x5 - 2 * (ml1 * ml1) * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) + Mn * x1 *
           (x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                        (x2 * x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) - 2 *
           (ml1 * ml1) * Mn * (x2 * x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) *
           (ml1 * ml1 - ml2 * ml2 - x1 + 2 * x2 - 2 * x4 - 2 * x6) - 2 * Mn *
           (x2 * x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                             (x2 * x2)) * x4 *
           (ml1 * ml1 - ml2 * ml2 - x1 + 2 * x2 - 2 * x4 - 2 * x6) - 4 *
           (ml1 * ml1 * ml1) * ml2 * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (x2 - x5 - x6) + 4 * ml1 * ml2 * Mn * x1 * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (x2 - x5 - x6) + 4 * ml1 * ml2 * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) * x4 *
           (x2 - x5 - x6) - 2 * (x1 * x1) * x2 * ((-2 * Enu * Mn + x2) *
                                                  (-2 * Enu * Mn + x2)) *
           (x1 - x2 + x3 + x4) * x5 * x6 + 4 * Mn * x1 * x2 *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (x1 - x2 + x3 + x4) * x5 * x6 + x1 * x1 * x2 *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) * x6 -
           2 * Mn * x1 * x2 * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                               (x2 * x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) * x6 + 2 *
           (ml1 * ml1) * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) *
           (ml1 * ml1 - ml2 * ml2 - x1 + 2 * x2 - 2 * x4 - 2 * x6) * x6 +
           2 * ml1 * ml2 * (x1 * x1) * x2 * ((-2 * Enu * Mn + x2) *
                                             (-2 * Enu * Mn + x2)) *
           (x2 - x5 - x6) * x6 - 4 * ml1 * ml2 * Mn * x1 * x2 *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (x2 - x5 - x6) * x6 + x1 * x1 * x1 *
           ((-2 * Enu * Mn + x2) *
            (-2 * Enu * Mn + x2)) * x5 * (x2 - x5 - x6) * x6 + 2 * Mn *
           (x1 * x1) * (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
                        (x2 * x2)) * x5 * (x2 - x5 - x6) * x6 + 2 * (x1 * x1) *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x4 * x5 *
           (x2 - x5 - x6) * x6 + 4 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn * (x2 * x2)) * x4 * x5 *
           (x2 - x5 - x6) * x6 + 2 * (x1 * x1) * ((-2 * Enu * Mn + x2) *
                                                  (-2 * Enu * Mn + x2)) *
           (x1 - x2 + x3 + x4) * x5 * (x6 * x6) + 4 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
            (x2 * x2)) * (x1 - x2 + x3 + x4) * x5 * (x6 * x6) - x1 * x1 * (
                (-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) *
           (x6 * x6) - 2 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn * (x2 * x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) *
           (x6 * x6) - 2 * ml1 * ml2 * (x1 * x1) * ((-2 * Enu * Mn + x2) *
                                                    (-2 * Enu * Mn + x2)) *
           (x2 - x5 - x6) * (x6 * x6) - 4 * ml1 * ml2 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
            (x2 * x2)) * (x2 - x5 - x6) * (x6 * x6) - x1 * x1 * (x2 * x2) * (
                (-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) +
           3 * Mn * x1 * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) + 2 *
           (x1 * x1) * x2 * ((-2 * Enu * Mn + x2) *
                             (-2 * Enu * Mn + x2)) * (x6 * x6) *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) -
           4 * Mn * x1 * x2 * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                               (x2 * x2)) * (x6 * x6) *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) -
           x1 * x1 * ((-2 * Enu * Mn + x2) *
                      (-2 * Enu * Mn + x2)) * (x6 * x6 * x6) *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) -
           2 * Mn * x1 * (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
                          (x2 * x2)) * (x6 * x6 * x6) *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)) + V2 *
          (-4 * (ml1 * ml1) * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (-x1 + x2 - x3 - x4) * x5 - 4 * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) *
           (-x1 + x2 - x3 - x4) * x4 * x5 - 2 * (ml1 * ml1) * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) + Mn * x1 *
           (x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                        (x2 * x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) - 2 *
           (ml1 * ml1) * Mn * (x2 * x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) *
           (ml1 * ml1 - ml2 * ml2 - x1 + 2 * x2 - 2 * x4 - 2 * x6) - 2 * Mn *
           (x2 * x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                             (x2 * x2)) * x4 *
           (ml1 * ml1 - ml2 * ml2 - x1 + 2 * x2 - 2 * x4 - 2 * x6) + 4 *
           (ml1 * ml1 * ml1) * ml2 * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (x2 - x5 - x6) - 4 * ml1 * ml2 * Mn * x1 * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (x2 - x5 - x6) - 4 * ml1 * ml2 * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) * x4 *
           (x2 - x5 - x6) - 2 * (x1 * x1) * x2 * ((-2 * Enu * Mn + x2) *
                                                  (-2 * Enu * Mn + x2)) *
           (x1 - x2 + x3 + x4) * x5 * x6 + 4 * Mn * x1 * x2 *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (x1 - x2 + x3 + x4) * x5 * x6 + x1 * x1 * x2 *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) * x6 -
           2 * Mn * x1 * x2 * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                               (x2 * x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) * x6 + 2 *
           (ml1 * ml1) * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) *
           (ml1 * ml1 - ml2 * ml2 - x1 + 2 * x2 - 2 * x4 - 2 * x6) * x6 -
           2 * ml1 * ml2 * (x1 * x1) * x2 * ((-2 * Enu * Mn + x2) *
                                             (-2 * Enu * Mn + x2)) *
           (x2 - x5 - x6) * x6 + 4 * ml1 * ml2 * Mn * x1 * x2 *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (x2 - x5 - x6) * x6 + x1 * x1 * x1 *
           ((-2 * Enu * Mn + x2) *
            (-2 * Enu * Mn + x2)) * x5 * (x2 - x5 - x6) * x6 + 2 * Mn *
           (x1 * x1) * (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
                        (x2 * x2)) * x5 * (x2 - x5 - x6) * x6 + 2 * (x1 * x1) *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x4 * x5 *
           (x2 - x5 - x6) * x6 + 4 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn * (x2 * x2)) * x4 * x5 *
           (x2 - x5 - x6) * x6 + 2 * (x1 * x1) * ((-2 * Enu * Mn + x2) *
                                                  (-2 * Enu * Mn + x2)) *
           (x1 - x2 + x3 + x4) * x5 * (x6 * x6) + 4 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
            (x2 * x2)) * (x1 - x2 + x3 + x4) * x5 * (x6 * x6) - x1 * x1 * (
                (-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) *
           (x6 * x6) - 2 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn * (x2 * x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) *
           (x6 * x6) + 2 * ml1 * ml2 * (x1 * x1) * ((-2 * Enu * Mn + x2) *
                                                    (-2 * Enu * Mn + x2)) *
           (x2 - x5 - x6) * (x6 * x6) + 4 * ml1 * ml2 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
            (x2 * x2)) * (x2 - x5 - x6) * (x6 * x6) - x1 * x1 * (x2 * x2) * (
                (-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) +
           3 * Mn * x1 * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) + 2 *
           (x1 * x1) * x2 * ((-2 * Enu * Mn + x2) *
                             (-2 * Enu * Mn + x2)) * (x6 * x6) *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) -
           4 * Mn * x1 * x2 * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                               (x2 * x2)) * (x6 * x6) *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) -
           x1 * x1 * ((-2 * Enu * Mn + x2) *
                      (-2 * Enu * Mn + x2)) * (x6 * x6 * x6) *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) -
           2 * Mn * x1 * (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
                          (x2 * x2)) * (x6 * x6 * x6) *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6)) +
          2 * VA *
          (-4 * (ml1 * ml1) * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (-x1 + x2 - x3 - x4) * x5 - 4 * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) *
           (-x1 + x2 - x3 - x4) * x4 * x5 - 2 * (ml1 * ml1) * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) + Mn * x1 *
           (x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                        (x2 * x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) + 2 *
           (ml1 * ml1) * Mn * (x2 * x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) *
           (ml1 * ml1 - ml2 * ml2 - x1 + 2 * x2 - 2 * x4 - 2 * x6) + 2 * Mn *
           (x2 * x2 * x2) * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                             (x2 * x2)) * x4 *
           (ml1 * ml1 - ml2 * ml2 - x1 + 2 * x2 - 2 * x4 - 2 * x6) - 2 *
           (x1 * x1) * x2 * ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) *
           (x1 - x2 + x3 + x4) * x5 * x6 + 4 * Mn * x1 * x2 *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
            (x2 * x2)) * (x1 - x2 + x3 + x4) * x5 * x6 + x1 * x1 * x2 *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) * x6 -
           2 * Mn * x1 * x2 * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                               (x2 * x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) * x6 - 2 *
           (ml1 * ml1) * Mn * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) *
           (ml1 * ml1 - ml2 * ml2 - x1 + 2 * x2 - 2 * x4 - 2 * x6) * x6 +
           x1 * x1 * x1 * ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x5 *
           (x2 - x5 - x6) * x6 + 2 * Mn * (x1 * x1) *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
            (x2 * x2)) * x5 * (x2 - x5 - x6) * x6 + 2 * (x1 * x1) *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x4 * x5 *
           (x2 - x5 - x6) * x6 + 4 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn * (x2 * x2)) * x4 * x5 *
           (x2 - x5 - x6) * x6 + 2 * (x1 * x1) * ((-2 * Enu * Mn + x2) *
                                                  (-2 * Enu * Mn + x2)) *
           (x1 - x2 + x3 + x4) * x5 * (x6 * x6) + 4 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
            (x2 * x2)) * (x1 - x2 + x3 + x4) * x5 * (x6 * x6) - x1 * x1 *
           ((-2 * Enu * Mn + x2) * (-2 * Enu * Mn + x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) *
           (x6 * x6) - 2 * Mn * x1 *
           (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn * (x2 * x2)) * x5 *
           (ml1 * ml1 - ml2 * ml2 + x1 - 2 * x2 + 2 * x3 + 2 * x5) *
           (x6 * x6) + x1 * x1 * (x2 * x2) * ((-2 * Enu * Mn + x2) *
                                              (-2 * Enu * Mn + x2)) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) -
           3 * Mn * x1 * (x2 * x2) *
           (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn * (x2 * x2)) * x6 *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) - 2 *
           (x1 * x1) * x2 * ((-2 * Enu * Mn + x2) *
                             (-2 * Enu * Mn + x2)) * (x6 * x6) *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) +
           4 * Mn * x1 * x2 * (-(Enu * Enu * Mn * x1) + Enu * x1 * x2 + Mn *
                               (x2 * x2)) * (x6 * x6) *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) +
           x1 * x1 * ((-2 * Enu * Mn + x2) *
                      (-2 * Enu * Mn + x2)) * (x6 * x6 * x6) *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6) +
           2 * Mn * x1 * (Enu * Enu * Mn * x1 - Enu * x1 * x2 - Mn *
                          (x2 * x2)) * (x6 * x6 * x6) *
           (-(ml1 * ml1) + ml2 * ml2 + x1 - 2 * x2 + 2 * x4 + 2 * x6))) +
         Diag12 * (x1 + 2 * x3) * (x1 + 2 * x4) *
         (2 * (Enu * Enu) * (Mn * Mn) * x1 *
          (-2 * V2 * x1 * (x2 * x2 * x2 * x2) + 4 * V2 *
           (x2 * x2 * x2 * x2 * x2) + V2 * x1 *
           (x2 * x2 * x2) * x3 + 2 * VA * x1 * (x2 * x2 * x2) * x3 - 6 * V2 *
           (x2 * x2 * x2 * x2) * x3 - 4 * VA *
           (x2 * x2 * x2 * x2) * x3 + 2 * V2 * (x2 * x2 * x2) *
           (x3 * x3) + 4 * VA * (x2 * x2 * x2) * (x3 * x3) + V2 * x1 *
           (x2 * x2 * x2) * x4 - 2 * VA * x1 * (x2 * x2 * x2) * x4 - 6 * V2 *
           (x2 * x2 * x2 * x2) * x4 + 4 * VA *
           (x2 * x2 * x2 * x2) * x4 + 4 * V2 *
           (x2 * x2 * x2) * x3 * x4 + 2 * V2 * (x2 * x2 * x2) *
           (x4 * x4) - 4 * VA * (x2 * x2 * x2) * (x4 * x4) + 4 * V2 *
           (x1 * x1) * (x2 * x2) * x5 + 8 * VA * (x1 * x1) *
           (x2 * x2) * x5 - 6 * V2 * x1 * (x2 * x2 * x2) * x5 - 12 * VA * x1 *
           (x2 * x2 * x2) * x5 - 8 * V2 * (x2 * x2 * x2 * x2) * x5 - 8 * VA *
           (x2 * x2 * x2 * x2) * x5 + 8 * V2 * x1 *
           (x2 * x2) * x3 * x5 + 16 * VA * x1 * (x2 * x2) * x3 * x5 + 6 * V2 *
           (x2 * x2 * x2) * x3 * x5 + 12 * VA *
           (x2 * x2 * x2) * x3 * x5 + 8 * V2 * x1 *
           (x2 * x2) * x4 * x5 + 16 * VA * x1 * (x2 * x2) * x4 * x5 + 6 * V2 *
           (x2 * x2 * x2) * x4 * x5 - 4 * VA *
           (x2 * x2 * x2) * x4 * x5 + 2 * V2 *
           (x2 * x2) * x3 * x4 * x5 + 4 * VA *
           (x2 * x2) * x3 * x4 * x5 - 2 * V2 * (x2 * x2) *
           (x4 * x4) * x5 + 4 * VA * (x2 * x2) * (x4 * x4) * x5 - 3 * V2 *
           (x1 * x1) * x2 * (x5 * x5) - 6 * VA * (x1 * x1) * x2 *
           (x5 * x5) + 12 * V2 * x1 * (x2 * x2) * (x5 * x5) + 24 * VA * x1 *
           (x2 * x2) * (x5 * x5) + 8 * V2 * (x2 * x2 * x2) *
           (x5 * x5) + 16 * VA * (x2 * x2 * x2) *
           (x5 * x5) - 6 * V2 * x1 * x2 * x3 *
           (x5 * x5) - 12 * VA * x1 * x2 * x3 * (x5 * x5) - 4 * V2 *
           (x2 * x2) * x3 * (x5 * x5) - 8 * VA * (x2 * x2) * x3 *
           (x5 * x5) - 12 * V2 * x1 * x2 * x4 *
           (x5 * x5) - 24 * VA * x1 * x2 * x4 * (x5 * x5) - 6 * V2 * x1 * x2 *
           (x5 * x5 * x5) - 12 * VA * x1 * x2 * (x5 * x5 * x5) - 4 * V2 *
           (x2 * x2) * (x5 * x5 * x5) - 8 * VA * (x2 * x2) *
           (x5 * x5 * x5) + 6 * V2 * x1 * x4 *
           (x5 * x5 * x5) + 12 * VA * x1 * x4 * (x5 * x5 * x5) + 4 * V2 *
           (x1 * x1) * (x2 * x2) * x6 - 8 * VA * (x1 * x1) *
           (x2 * x2) * x6 - 6 * V2 * x1 * (x2 * x2 * x2) * x6 + 12 * VA * x1 *
           (x2 * x2 * x2) * x6 - 8 * V2 * (x2 * x2 * x2 * x2) * x6 + 8 * VA *
           (x2 * x2 * x2 * x2) * x6 + 8 * V2 * x1 *
           (x2 * x2) * x3 * x6 - 16 * VA * x1 * (x2 * x2) * x3 * x6 + 6 * V2 *
           (x2 * x2 * x2) * x3 * x6 + 4 * VA *
           (x2 * x2 * x2) * x3 * x6 - 2 * V2 * (x2 * x2) *
           (x3 * x3) * x6 - 4 * VA * (x2 * x2) * (x3 * x3) * x6 + 8 * V2 * x1 *
           (x2 * x2) * x4 * x6 - 16 * VA * x1 * (x2 * x2) * x4 * x6 + 6 * V2 *
           (x2 * x2 * x2) * x4 * x6 - 12 * VA *
           (x2 * x2 * x2) * x4 * x6 + 2 * V2 *
           (x2 * x2) * x3 * x4 * x6 - 4 * VA *
           (x2 * x2) * x3 * x4 * x6 + 16 * V2 * x1 *
           (x2 * x2) * x5 * x6 + 8 * V2 *
           (x2 * x2 * x2) * x5 * x6 - 6 * V2 * x1 * x2 * x3 * x5 * x6 -
           12 * VA * x1 * x2 * x3 * x5 * x6 - 4 * V2 *
           (x2 * x2) * x3 * x5 * x6 - 8 * VA *
           (x2 * x2) * x3 * x5 * x6 - 6 * V2 * x1 * x2 * x4 * x5 * x6 +
           12 * VA * x1 * x2 * x4 * x5 * x6 - 4 * V2 *
           (x2 * x2) * x4 * x5 * x6 + 8 * VA *
           (x2 * x2) * x4 * x5 * x6 - 18 * V2 * x1 * x2 *
           (x5 * x5) * x6 - 36 * VA * x1 * x2 * (x5 * x5) * x6 - 4 * V2 *
           (x2 * x2) * (x5 * x5) * x6 - 8 * VA * (x2 * x2) *
           (x5 * x5) * x6 + 6 * V2 * x1 * x3 *
           (x5 * x5) * x6 + 12 * VA * x1 * x3 * (x5 * x5) * x6 + 12 * V2 * x1 *
           (x5 * x5 * x5) * x6 + 24 * VA * x1 * (x5 * x5 * x5) * x6 - 3 * V2 *
           (x1 * x1) * x2 * (x6 * x6) + 6 * VA * (x1 * x1) * x2 *
           (x6 * x6) + 12 * V2 * x1 * (x2 * x2) * (x6 * x6) - 24 * VA * x1 *
           (x2 * x2) * (x6 * x6) + 8 * V2 * (x2 * x2 * x2) *
           (x6 * x6) - 16 * VA * (x2 * x2 * x2) *
           (x6 * x6) - 12 * V2 * x1 * x2 * x3 *
           (x6 * x6) + 24 * VA * x1 * x2 * x3 *
           (x6 * x6) - 6 * V2 * x1 * x2 * x4 *
           (x6 * x6) + 12 * VA * x1 * x2 * x4 * (x6 * x6) - 4 * V2 *
           (x2 * x2) * x4 * (x6 * x6) + 8 * VA * (x2 * x2) * x4 *
           (x6 * x6) - 18 * V2 * x1 * x2 * x5 *
           (x6 * x6) + 36 * VA * x1 * x2 * x5 * (x6 * x6) - 4 * V2 *
           (x2 * x2) * x5 * (x6 * x6) + 8 * VA * (x2 * x2) * x5 *
           (x6 * x6) + 6 * V2 * x1 * x4 * x5 *
           (x6 * x6) - 12 * VA * x1 * x4 * x5 * (x6 * x6) - 6 * V2 * x1 * x2 *
           (x6 * x6 * x6) + 12 * VA * x1 * x2 * (x6 * x6 * x6) - 4 * V2 *
           (x2 * x2) * (x6 * x6 * x6) + 8 * VA * (x2 * x2) *
           (x6 * x6 * x6) + 6 * V2 * x1 * x3 *
           (x6 * x6 * x6) - 12 * VA * x1 * x3 *
           (x6 * x6 * x6) + 12 * V2 * x1 * x5 *
           (x6 * x6 * x6) - 24 * VA * x1 * x5 * (x6 * x6 * x6) + 2 *
           (ml1 * ml1 * ml1) * ml2 * V2 * (x2 * x2) *
           (-x2 + x5 + x6) + ml2 * ml2 * ml2 * ml2 * (x2 * x2) *
           (2 * VA * (x2 - x5 - x6) + V2 *
            (-x5 + x6)) + ml1 * ml1 * ml1 * ml1 * (x2 * x2) *
           (V2 * (x5 - x6) + 2 * VA * (-x2 + x5 + x6)) - 2 * ml1 * ml2 * V2 *
           (ml2 * ml2 * (x2 * x2) * (x2 - x5 - x6) + x1 *
            (x2 * x2 * x2 - 6 * (x2 * x2) * (x5 + x6) - 6 * x5 * x6 *
             (x5 + x6) + 3 * x2 *
             (x5 * x5 + 4 * x5 * x6 + x6 * x6)) + x2 * x2 *
            (4 * (x2 * x2) + (x5 + x6) * (x3 + x4 + 2 * (x5 + x6)) - x2 *
             (3 * x3 + 3 * x4 + 4 * (x5 + x6)))) - ml2 * ml2 *
           (V2 *
            (-2 * (x2 * x2 * x2 * x2) + x2 * x2 * x2 *
             (x3 + x4 + 4 * x5) + x2 * x2 *
             (-2 * x5 * (x3 - x4 + 2 * x5) + x1 *
              (3 * x5 - x6)) + 6 * x1 * x5 * (x5 - x6) * x6 + 3 * x1 * x2 *
             (-(x5 * x5) + x6 * x6)) + 2 * VA *
            (x2 * x2 * (-2 * (x2 * x2) - 2 * (x3 + 2 * x5) * (x5 + x6) + x2 *
                        (3 * x3 + x4 + 6 * x5 + 2 * x6)) + x1 *
             (x2 * x2 * x2 + 3 * (x2 * x2) * (x5 + x6) + 6 * x5 * x6 *
              (x5 + x6) - 3 * x2 *
              (x5 * x5 + 4 * x5 * x6 + x6 * x6)))) + ml1 * ml1 *
           (V2 *
            (2 * (x2 * x2 * x2 * x2) + 6 * x1 * x5 *
             (x5 - x6) * x6 - x2 * x2 * x2 * (x3 + x4 + 4 * x6) + 3 * x1 * x2 *
             (-(x5 * x5) + x6 * x6) + x2 * x2 *
             (x1 * (x5 - 3 * x6) + 2 * x6 * (-x3 + x4 + 2 * x6))) + 2 * VA *
            (x2 * x2 * (-2 * (x2 * x2) - 2 * (x5 + x6) * (x4 + 2 * x6) + x2 *
                        (x3 + 3 * x4 + 2 * x5 + 6 * x6)) + x1 *
             (x2 * x2 * x2 + 3 * (x2 * x2) * (x5 + x6) + 6 * x5 * x6 *
              (x5 + x6) - 3 * x2 *
              (x5 * x5 + 4 * x5 * x6 + x6 * x6))))) - 2 * Enu * Mn * x1 * x2 *
          (-2 * V2 * x1 * (x2 * x2 * x2 * x2) + 4 * V2 *
           (x2 * x2 * x2 * x2 * x2) + V2 * x1 *
           (x2 * x2 * x2) * x3 + 2 * VA * x1 * (x2 * x2 * x2) * x3 - 6 * V2 *
           (x2 * x2 * x2 * x2) * x3 - 4 * VA *
           (x2 * x2 * x2 * x2) * x3 + 2 * V2 * (x2 * x2 * x2) *
           (x3 * x3) + 4 * VA * (x2 * x2 * x2) * (x3 * x3) + V2 * x1 *
           (x2 * x2 * x2) * x4 - 2 * VA * x1 * (x2 * x2 * x2) * x4 - 6 * V2 *
           (x2 * x2 * x2 * x2) * x4 + 4 * VA *
           (x2 * x2 * x2 * x2) * x4 + 4 * V2 *
           (x2 * x2 * x2) * x3 * x4 + 2 * V2 * (x2 * x2 * x2) *
           (x4 * x4) - 4 * VA * (x2 * x2 * x2) * (x4 * x4) + 4 * V2 *
           (x1 * x1) * (x2 * x2) * x5 + 8 * VA * (x1 * x1) *
           (x2 * x2) * x5 - 6 * V2 * x1 * (x2 * x2 * x2) * x5 - 12 * VA * x1 *
           (x2 * x2 * x2) * x5 - 8 * V2 * (x2 * x2 * x2 * x2) * x5 - 8 * VA *
           (x2 * x2 * x2 * x2) * x5 + 8 * V2 * x1 *
           (x2 * x2) * x3 * x5 + 16 * VA * x1 * (x2 * x2) * x3 * x5 + 6 * V2 *
           (x2 * x2 * x2) * x3 * x5 + 12 * VA *
           (x2 * x2 * x2) * x3 * x5 + 8 * V2 * x1 *
           (x2 * x2) * x4 * x5 + 16 * VA * x1 * (x2 * x2) * x4 * x5 + 6 * V2 *
           (x2 * x2 * x2) * x4 * x5 - 4 * VA *
           (x2 * x2 * x2) * x4 * x5 + 2 * V2 *
           (x2 * x2) * x3 * x4 * x5 + 4 * VA *
           (x2 * x2) * x3 * x4 * x5 - 2 * V2 * (x2 * x2) *
           (x4 * x4) * x5 + 4 * VA * (x2 * x2) * (x4 * x4) * x5 - 3 * V2 *
           (x1 * x1) * x2 * (x5 * x5) - 6 * VA * (x1 * x1) * x2 *
           (x5 * x5) + 12 * V2 * x1 * (x2 * x2) * (x5 * x5) + 24 * VA * x1 *
           (x2 * x2) * (x5 * x5) + 8 * V2 * (x2 * x2 * x2) *
           (x5 * x5) + 16 * VA * (x2 * x2 * x2) *
           (x5 * x5) - 6 * V2 * x1 * x2 * x3 *
           (x5 * x5) - 12 * VA * x1 * x2 * x3 * (x5 * x5) - 4 * V2 *
           (x2 * x2) * x3 * (x5 * x5) - 8 * VA * (x2 * x2) * x3 *
           (x5 * x5) - 12 * V2 * x1 * x2 * x4 *
           (x5 * x5) - 24 * VA * x1 * x2 * x4 * (x5 * x5) - 6 * V2 * x1 * x2 *
           (x5 * x5 * x5) - 12 * VA * x1 * x2 * (x5 * x5 * x5) - 4 * V2 *
           (x2 * x2) * (x5 * x5 * x5) - 8 * VA * (x2 * x2) *
           (x5 * x5 * x5) + 6 * V2 * x1 * x4 *
           (x5 * x5 * x5) + 12 * VA * x1 * x4 * (x5 * x5 * x5) + 4 * V2 *
           (x1 * x1) * (x2 * x2) * x6 - 8 * VA * (x1 * x1) *
           (x2 * x2) * x6 - 6 * V2 * x1 * (x2 * x2 * x2) * x6 + 12 * VA * x1 *
           (x2 * x2 * x2) * x6 - 8 * V2 * (x2 * x2 * x2 * x2) * x6 + 8 * VA *
           (x2 * x2 * x2 * x2) * x6 + 8 * V2 * x1 *
           (x2 * x2) * x3 * x6 - 16 * VA * x1 * (x2 * x2) * x3 * x6 + 6 * V2 *
           (x2 * x2 * x2) * x3 * x6 + 4 * VA *
           (x2 * x2 * x2) * x3 * x6 - 2 * V2 * (x2 * x2) *
           (x3 * x3) * x6 - 4 * VA * (x2 * x2) * (x3 * x3) * x6 + 8 * V2 * x1 *
           (x2 * x2) * x4 * x6 - 16 * VA * x1 * (x2 * x2) * x4 * x6 + 6 * V2 *
           (x2 * x2 * x2) * x4 * x6 - 12 * VA *
           (x2 * x2 * x2) * x4 * x6 + 2 * V2 *
           (x2 * x2) * x3 * x4 * x6 - 4 * VA *
           (x2 * x2) * x3 * x4 * x6 + 16 * V2 * x1 *
           (x2 * x2) * x5 * x6 + 8 * V2 *
           (x2 * x2 * x2) * x5 * x6 - 6 * V2 * x1 * x2 * x3 * x5 * x6 -
           12 * VA * x1 * x2 * x3 * x5 * x6 - 4 * V2 *
           (x2 * x2) * x3 * x5 * x6 - 8 * VA *
           (x2 * x2) * x3 * x5 * x6 - 6 * V2 * x1 * x2 * x4 * x5 * x6 +
           12 * VA * x1 * x2 * x4 * x5 * x6 - 4 * V2 *
           (x2 * x2) * x4 * x5 * x6 + 8 * VA *
           (x2 * x2) * x4 * x5 * x6 - 18 * V2 * x1 * x2 *
           (x5 * x5) * x6 - 36 * VA * x1 * x2 * (x5 * x5) * x6 - 4 * V2 *
           (x2 * x2) * (x5 * x5) * x6 - 8 * VA * (x2 * x2) *
           (x5 * x5) * x6 + 6 * V2 * x1 * x3 *
           (x5 * x5) * x6 + 12 * VA * x1 * x3 * (x5 * x5) * x6 + 12 * V2 * x1 *
           (x5 * x5 * x5) * x6 + 24 * VA * x1 * (x5 * x5 * x5) * x6 - 3 * V2 *
           (x1 * x1) * x2 * (x6 * x6) + 6 * VA * (x1 * x1) * x2 *
           (x6 * x6) + 12 * V2 * x1 * (x2 * x2) * (x6 * x6) - 24 * VA * x1 *
           (x2 * x2) * (x6 * x6) + 8 * V2 * (x2 * x2 * x2) *
           (x6 * x6) - 16 * VA * (x2 * x2 * x2) *
           (x6 * x6) - 12 * V2 * x1 * x2 * x3 *
           (x6 * x6) + 24 * VA * x1 * x2 * x3 *
           (x6 * x6) - 6 * V2 * x1 * x2 * x4 *
           (x6 * x6) + 12 * VA * x1 * x2 * x4 * (x6 * x6) - 4 * V2 *
           (x2 * x2) * x4 * (x6 * x6) + 8 * VA * (x2 * x2) * x4 *
           (x6 * x6) - 18 * V2 * x1 * x2 * x5 *
           (x6 * x6) + 36 * VA * x1 * x2 * x5 * (x6 * x6) - 4 * V2 *
           (x2 * x2) * x5 * (x6 * x6) + 8 * VA * (x2 * x2) * x5 *
           (x6 * x6) + 6 * V2 * x1 * x4 * x5 *
           (x6 * x6) - 12 * VA * x1 * x4 * x5 * (x6 * x6) - 6 * V2 * x1 * x2 *
           (x6 * x6 * x6) + 12 * VA * x1 * x2 * (x6 * x6 * x6) - 4 * V2 *
           (x2 * x2) * (x6 * x6 * x6) + 8 * VA * (x2 * x2) *
           (x6 * x6 * x6) + 6 * V2 * x1 * x3 *
           (x6 * x6 * x6) - 12 * VA * x1 * x3 *
           (x6 * x6 * x6) + 12 * V2 * x1 * x5 *
           (x6 * x6 * x6) - 24 * VA * x1 * x5 * (x6 * x6 * x6) + 2 *
           (ml1 * ml1 * ml1) * ml2 * V2 * (x2 * x2) *
           (-x2 + x5 + x6) + ml2 * ml2 * ml2 * ml2 * (x2 * x2) *
           (2 * VA * (x2 - x5 - x6) + V2 *
            (-x5 + x6)) + ml1 * ml1 * ml1 * ml1 * (x2 * x2) *
           (V2 * (x5 - x6) + 2 * VA * (-x2 + x5 + x6)) - 2 * ml1 * ml2 * V2 *
           (ml2 * ml2 * (x2 * x2) * (x2 - x5 - x6) + x1 *
            (x2 * x2 * x2 - 6 * (x2 * x2) * (x5 + x6) - 6 * x5 * x6 *
             (x5 + x6) + 3 * x2 *
             (x5 * x5 + 4 * x5 * x6 + x6 * x6)) + x2 * x2 *
            (4 * (x2 * x2) + (x5 + x6) * (x3 + x4 + 2 * (x5 + x6)) - x2 *
             (3 * x3 + 3 * x4 + 4 * (x5 + x6)))) - ml2 * ml2 *
           (V2 *
            (-2 * (x2 * x2 * x2 * x2) + x2 * x2 * x2 *
             (x3 + x4 + 4 * x5) + x2 * x2 *
             (-2 * x5 * (x3 - x4 + 2 * x5) + x1 *
              (3 * x5 - x6)) + 6 * x1 * x5 * (x5 - x6) * x6 + 3 * x1 * x2 *
             (-(x5 * x5) + x6 * x6)) + 2 * VA *
            (x2 * x2 * (-2 * (x2 * x2) - 2 * (x3 + 2 * x5) * (x5 + x6) + x2 *
                        (3 * x3 + x4 + 6 * x5 + 2 * x6)) + x1 *
             (x2 * x2 * x2 + 3 * (x2 * x2) * (x5 + x6) + 6 * x5 * x6 *
              (x5 + x6) - 3 * x2 *
              (x5 * x5 + 4 * x5 * x6 + x6 * x6)))) + ml1 * ml1 *
           (V2 *
            (2 * (x2 * x2 * x2 * x2) + 6 * x1 * x5 *
             (x5 - x6) * x6 - x2 * x2 * x2 * (x3 + x4 + 4 * x6) + 3 * x1 * x2 *
             (-(x5 * x5) + x6 * x6) + x2 * x2 *
             (x1 * (x5 - 3 * x6) + 2 * x6 * (-x3 + x4 + 2 * x6))) + 2 * VA *
            (x2 * x2 * (-2 * (x2 * x2) - 2 * (x5 + x6) * (x4 + 2 * x6) + x2 *
                        (x3 + 3 * x4 + 2 * x5 + 6 * x6)) + x1 *
             (x2 * x2 * x2 + 3 * (x2 * x2) * (x5 + x6) + 6 * x5 * x6 *
              (x5 + x6) - 3 * x2 *
              (x5 * x5 + 4 * x5 * x6 + x6 * x6))))) + x2 * x2 *
          (4 * (Mn * Mn) * V2 * x1 * (x2 * x2 * x2 * x2) - 8 * (Mn * Mn) * V2 *
           (x2 * x2 * x2 * x2 * x2) - 2 * (Mn * Mn) * V2 * x1 *
           (x2 * x2 * x2) * x3 - 4 * (Mn * Mn) * VA * x1 *
           (x2 * x2 * x2) * x3 + 12 * (Mn * Mn) * V2 *
           (x2 * x2 * x2 * x2) * x3 + 8 * (Mn * Mn) * VA *
           (x2 * x2 * x2 * x2) * x3 - 4 * (Mn * Mn) * V2 * (x2 * x2 * x2) *
           (x3 * x3) - 8 * (Mn * Mn) * VA * (x2 * x2 * x2) * (x3 * x3) - 2 *
           (Mn * Mn) * V2 * x1 * (x2 * x2 * x2) * x4 + 4 *
           (Mn * Mn) * VA * x1 * (x2 * x2 * x2) * x4 + 12 * (Mn * Mn) * V2 *
           (x2 * x2 * x2 * x2) * x4 - 8 * (Mn * Mn) * VA *
           (x2 * x2 * x2 * x2) * x4 - 8 * (Mn * Mn) * V2 *
           (x2 * x2 * x2) * x3 * x4 - 4 * (Mn * Mn) * V2 * (x2 * x2 * x2) *
           (x4 * x4) + 8 * (Mn * Mn) * VA * (x2 * x2 * x2) * (x4 * x4) - 4 *
           (Mn * Mn) * V2 * (x1 * x1) * (x2 * x2) * x5 - 8 * (Mn * Mn) * VA *
           (x1 * x1) * (x2 * x2) * x5 + V2 * (x1 * x1 * x1) *
           (x2 * x2) * x5 + 2 * VA * (x1 * x1 * x1) * (x2 * x2) * x5 + 4 *
           (Mn * Mn) * V2 * x1 * (x2 * x2 * x2) * x5 + 8 *
           (Mn * Mn) * VA * x1 * (x2 * x2 * x2) * x5 - 2 * V2 * (x1 * x1) *
           (x2 * x2 * x2) * x5 - 4 * VA * (x1 * x1) *
           (x2 * x2 * x2) * x5 + 16 * (Mn * Mn) * V2 *
           (x2 * x2 * x2 * x2) * x5 + 16 * (Mn * Mn) * VA *
           (x2 * x2 * x2 * x2) * x5 - 8 * (Mn * Mn) * V2 * x1 *
           (x2 * x2) * x3 * x5 - 16 * (Mn * Mn) * VA * x1 *
           (x2 * x2) * x3 * x5 + 2 * V2 * (x1 * x1) *
           (x2 * x2) * x3 * x5 + 4 * VA * (x1 * x1) *
           (x2 * x2) * x3 * x5 - 12 * (Mn * Mn) * V2 *
           (x2 * x2 * x2) * x3 * x5 - 24 * (Mn * Mn) * VA *
           (x2 * x2 * x2) * x3 * x5 - 8 * (Mn * Mn) * V2 * x1 *
           (x2 * x2) * x4 * x5 - 16 * (Mn * Mn) * VA * x1 *
           (x2 * x2) * x4 * x5 + 2 * V2 * (x1 * x1) *
           (x2 * x2) * x4 * x5 + 4 * VA * (x1 * x1) *
           (x2 * x2) * x4 * x5 - 12 * (Mn * Mn) * V2 *
           (x2 * x2 * x2) * x4 * x5 + 8 * (Mn * Mn) * VA *
           (x2 * x2 * x2) * x4 * x5 - 4 * (Mn * Mn) * V2 *
           (x2 * x2) * x3 * x4 * x5 - 8 * (Mn * Mn) * VA *
           (x2 * x2) * x3 * x4 * x5 + 4 * (Mn * Mn) * V2 * (x2 * x2) *
           (x4 * x4) * x5 - 8 * (Mn * Mn) * VA * (x2 * x2) *
           (x4 * x4) * x5 + 2 * (Mn * Mn) * V2 * (x1 * x1) * x2 *
           (x5 * x5) + 4 * (Mn * Mn) * VA * (x1 * x1) * x2 * (x5 * x5) - V2 *
           (x1 * x1 * x1) * x2 * (x5 * x5) - 2 * VA * (x1 * x1 * x1) * x2 *
           (x5 * x5) - 8 * (Mn * Mn) * V2 * x1 * (x2 * x2) * (x5 * x5) - 16 *
           (Mn * Mn) * VA * x1 * (x2 * x2) * (x5 * x5) + 4 * V2 * (x1 * x1) *
           (x2 * x2) * (x5 * x5) + 8 * VA * (x1 * x1) * (x2 * x2) *
           (x5 * x5) - 16 * (Mn * Mn) * V2 * (x2 * x2 * x2) * (x5 * x5) - 32 *
           (Mn * Mn) * VA * (x2 * x2 * x2) * (x5 * x5) + 4 *
           (Mn * Mn) * V2 * x1 * x2 * x3 * (x5 * x5) + 8 *
           (Mn * Mn) * VA * x1 * x2 * x3 * (x5 * x5) - 2 * V2 *
           (x1 * x1) * x2 * x3 * (x5 * x5) - 4 * VA * (x1 * x1) * x2 * x3 *
           (x5 * x5) + 8 * (Mn * Mn) * V2 * (x2 * x2) * x3 * (x5 * x5) + 16 *
           (Mn * Mn) * VA * (x2 * x2) * x3 * (x5 * x5) + 8 *
           (Mn * Mn) * V2 * x1 * x2 * x4 * (x5 * x5) + 16 *
           (Mn * Mn) * VA * x1 * x2 * x4 * (x5 * x5) - 4 * V2 *
           (x1 * x1) * x2 * x4 * (x5 * x5) - 8 * VA * (x1 * x1) * x2 * x4 *
           (x5 * x5) + 4 * (Mn * Mn) * V2 * x1 * x2 * (x5 * x5 * x5) + 8 *
           (Mn * Mn) * VA * x1 * x2 * (x5 * x5 * x5) - 2 * V2 *
           (x1 * x1) * x2 * (x5 * x5 * x5) - 4 * VA * (x1 * x1) * x2 *
           (x5 * x5 * x5) + 8 * (Mn * Mn) * V2 * (x2 * x2) *
           (x5 * x5 * x5) + 16 * (Mn * Mn) * VA * (x2 * x2) *
           (x5 * x5 * x5) - 4 * (Mn * Mn) * V2 * x1 * x4 * (x5 * x5 * x5) - 8 *
           (Mn * Mn) * VA * x1 * x4 * (x5 * x5 * x5) + 2 * V2 *
           (x1 * x1) * x4 * (x5 * x5 * x5) + 4 * VA * (x1 * x1) * x4 *
           (x5 * x5 * x5) + 4 * (ml1 * ml1 * ml1) * ml2 * (Mn * Mn) * V2 *
           (x2 * x2) * (x2 - x5 - x6) - 4 * (Mn * Mn) * V2 * (x1 * x1) *
           (x2 * x2) * x6 + 8 * (Mn * Mn) * VA * (x1 * x1) *
           (x2 * x2) * x6 + V2 * (x1 * x1 * x1) * (x2 * x2) * x6 - 2 * VA *
           (x1 * x1 * x1) * (x2 * x2) * x6 + 4 * (Mn * Mn) * V2 * x1 *
           (x2 * x2 * x2) * x6 - 8 * (Mn * Mn) * VA * x1 *
           (x2 * x2 * x2) * x6 - 2 * V2 * (x1 * x1) *
           (x2 * x2 * x2) * x6 + 4 * VA * (x1 * x1) *
           (x2 * x2 * x2) * x6 + 16 * (Mn * Mn) * V2 *
           (x2 * x2 * x2 * x2) * x6 - 16 * (Mn * Mn) * VA *
           (x2 * x2 * x2 * x2) * x6 - 8 * (Mn * Mn) * V2 * x1 *
           (x2 * x2) * x3 * x6 + 16 * (Mn * Mn) * VA * x1 *
           (x2 * x2) * x3 * x6 + 2 * V2 * (x1 * x1) *
           (x2 * x2) * x3 * x6 - 4 * VA * (x1 * x1) *
           (x2 * x2) * x3 * x6 - 12 * (Mn * Mn) * V2 *
           (x2 * x2 * x2) * x3 * x6 - 8 * (Mn * Mn) * VA *
           (x2 * x2 * x2) * x3 * x6 + 4 * (Mn * Mn) * V2 * (x2 * x2) *
           (x3 * x3) * x6 + 8 * (Mn * Mn) * VA * (x2 * x2) *
           (x3 * x3) * x6 - 8 * (Mn * Mn) * V2 * x1 *
           (x2 * x2) * x4 * x6 + 16 * (Mn * Mn) * VA * x1 *
           (x2 * x2) * x4 * x6 + 2 * V2 * (x1 * x1) *
           (x2 * x2) * x4 * x6 - 4 * VA * (x1 * x1) *
           (x2 * x2) * x4 * x6 - 12 * (Mn * Mn) * V2 *
           (x2 * x2 * x2) * x4 * x6 + 24 * (Mn * Mn) * VA *
           (x2 * x2 * x2) * x4 * x6 - 4 * (Mn * Mn) * V2 *
           (x2 * x2) * x3 * x4 * x6 + 8 * (Mn * Mn) * VA *
           (x2 * x2) * x3 * x4 * x6 - 16 * (Mn * Mn) * V2 * x1 *
           (x2 * x2) * x5 * x6 + 4 * V2 * (x1 * x1) *
           (x2 * x2) * x5 * x6 - 16 * (Mn * Mn) * V2 *
           (x2 * x2 * x2) * x5 * x6 + 4 *
           (Mn * Mn) * V2 * x1 * x2 * x3 * x5 * x6 + 8 *
           (Mn * Mn) * VA * x1 * x2 * x3 * x5 * x6 - 2 * V2 *
           (x1 * x1) * x2 * x3 * x5 * x6 - 4 * VA *
           (x1 * x1) * x2 * x3 * x5 * x6 + 8 * (Mn * Mn) * V2 *
           (x2 * x2) * x3 * x5 * x6 + 16 * (Mn * Mn) * VA *
           (x2 * x2) * x3 * x5 * x6 + 4 *
           (Mn * Mn) * V2 * x1 * x2 * x4 * x5 * x6 - 8 *
           (Mn * Mn) * VA * x1 * x2 * x4 * x5 * x6 - 2 * V2 *
           (x1 * x1) * x2 * x4 * x5 * x6 + 4 * VA *
           (x1 * x1) * x2 * x4 * x5 * x6 + 8 * (Mn * Mn) * V2 *
           (x2 * x2) * x4 * x5 * x6 - 16 * (Mn * Mn) * VA *
           (x2 * x2) * x4 * x5 * x6 + 12 * (Mn * Mn) * V2 * x1 * x2 *
           (x5 * x5) * x6 + 24 * (Mn * Mn) * VA * x1 * x2 *
           (x5 * x5) * x6 - 6 * V2 * (x1 * x1) * x2 *
           (x5 * x5) * x6 - 12 * VA * (x1 * x1) * x2 * (x5 * x5) * x6 + 8 *
           (Mn * Mn) * V2 * (x2 * x2) * (x5 * x5) * x6 + 16 * (Mn * Mn) * VA *
           (x2 * x2) * (x5 * x5) * x6 - 4 * (Mn * Mn) * V2 * x1 * x3 *
           (x5 * x5) * x6 - 8 * (Mn * Mn) * VA * x1 * x3 *
           (x5 * x5) * x6 + 2 * V2 * (x1 * x1) * x3 * (x5 * x5) * x6 + 4 * VA *
           (x1 * x1) * x3 * (x5 * x5) * x6 - 8 * (Mn * Mn) * V2 * x1 *
           (x5 * x5 * x5) * x6 - 16 * (Mn * Mn) * VA * x1 *
           (x5 * x5 * x5) * x6 + 4 * V2 * (x1 * x1) *
           (x5 * x5 * x5) * x6 + 8 * VA * (x1 * x1) * (x5 * x5 * x5) * x6 + 2 *
           (Mn * Mn) * V2 * (x1 * x1) * x2 * (x6 * x6) - 4 * (Mn * Mn) * VA *
           (x1 * x1) * x2 * (x6 * x6) - V2 * (x1 * x1 * x1) * x2 *
           (x6 * x6) + 2 * VA * (x1 * x1 * x1) * x2 * (x6 * x6) - 8 *
           (Mn * Mn) * V2 * x1 * (x2 * x2) * (x6 * x6) + 16 *
           (Mn * Mn) * VA * x1 * (x2 * x2) * (x6 * x6) + 4 * V2 * (x1 * x1) *
           (x2 * x2) * (x6 * x6) - 8 * VA * (x1 * x1) * (x2 * x2) *
           (x6 * x6) - 16 * (Mn * Mn) * V2 * (x2 * x2 * x2) * (x6 * x6) + 32 *
           (Mn * Mn) * VA * (x2 * x2 * x2) * (x6 * x6) + 8 *
           (Mn * Mn) * V2 * x1 * x2 * x3 * (x6 * x6) - 16 *
           (Mn * Mn) * VA * x1 * x2 * x3 * (x6 * x6) - 4 * V2 *
           (x1 * x1) * x2 * x3 * (x6 * x6) + 8 * VA * (x1 * x1) * x2 * x3 *
           (x6 * x6) + 4 * (Mn * Mn) * V2 * x1 * x2 * x4 * (x6 * x6) - 8 *
           (Mn * Mn) * VA * x1 * x2 * x4 * (x6 * x6) - 2 * V2 *
           (x1 * x1) * x2 * x4 * (x6 * x6) + 4 * VA * (x1 * x1) * x2 * x4 *
           (x6 * x6) + 8 * (Mn * Mn) * V2 * (x2 * x2) * x4 * (x6 * x6) - 16 *
           (Mn * Mn) * VA * (x2 * x2) * x4 * (x6 * x6) + 12 *
           (Mn * Mn) * V2 * x1 * x2 * x5 * (x6 * x6) - 24 *
           (Mn * Mn) * VA * x1 * x2 * x5 * (x6 * x6) - 6 * V2 *
           (x1 * x1) * x2 * x5 * (x6 * x6) + 12 * VA * (x1 * x1) * x2 * x5 *
           (x6 * x6) + 8 * (Mn * Mn) * V2 * (x2 * x2) * x5 * (x6 * x6) - 16 *
           (Mn * Mn) * VA * (x2 * x2) * x5 * (x6 * x6) - 4 *
           (Mn * Mn) * V2 * x1 * x4 * x5 * (x6 * x6) + 8 *
           (Mn * Mn) * VA * x1 * x4 * x5 * (x6 * x6) + 2 * V2 *
           (x1 * x1) * x4 * x5 * (x6 * x6) - 4 * VA * (x1 * x1) * x4 * x5 *
           (x6 * x6) + 4 * (Mn * Mn) * V2 * x1 * x2 * (x6 * x6 * x6) - 8 *
           (Mn * Mn) * VA * x1 * x2 * (x6 * x6 * x6) - 2 * V2 *
           (x1 * x1) * x2 * (x6 * x6 * x6) + 4 * VA * (x1 * x1) * x2 *
           (x6 * x6 * x6) + 8 * (Mn * Mn) * V2 * (x2 * x2) *
           (x6 * x6 * x6) - 16 * (Mn * Mn) * VA * (x2 * x2) *
           (x6 * x6 * x6) - 4 * (Mn * Mn) * V2 * x1 * x3 * (x6 * x6 * x6) + 8 *
           (Mn * Mn) * VA * x1 * x3 * (x6 * x6 * x6) + 2 * V2 *
           (x1 * x1) * x3 * (x6 * x6 * x6) - 4 * VA * (x1 * x1) * x3 *
           (x6 * x6 * x6) - 8 * (Mn * Mn) * V2 * x1 * x5 *
           (x6 * x6 * x6) + 16 * (Mn * Mn) * VA * x1 * x5 *
           (x6 * x6 * x6) + 4 * V2 * (x1 * x1) * x5 * (x6 * x6 * x6) - 8 * VA *
           (x1 * x1) * x5 * (x6 * x6 * x6) + 2 * (ml1 * ml1 * ml1 * ml1) *
           (Mn * Mn) * (x2 * x2) * (2 * VA * (x2 - x5 - x6) + V2 *
                                    (-x5 + x6)) + 2 * (ml2 * ml2 * ml2 * ml2) *
           (Mn * Mn) * (x2 * x2) * (V2 * (x5 - x6) + 2 * VA *
                                    (-x2 + x5 + x6)) + 2 * ml1 * ml2 * V2 *
           (2 * (ml2 * ml2) * (Mn * Mn) * (x2 * x2) *
            (x2 - x5 - x6) - x1 * x1 *
            (x2 * x2 * x2 - 2 * (x2 * x2) * (x5 + x6) - 2 * x5 * x6 *
             (x5 + x6) + x2 *
             (x5 * x5 + 4 * x5 * x6 + x6 * x6)) + 2 * (Mn * Mn) *
            (-(x1 * (x2 * x2 * x2 + 2 * (x2 * x2) * (x5 + x6) + 2 * x5 * x6 *
                     (x5 + x6) - x2 *
                     (x5 * x5 + 4 * x5 * x6 + x6 * x6))) + x2 * x2 *
             (4 * (x2 * x2) + (x5 + x6) * (x3 + x4 + 2 * (x5 + x6)) - x2 *
              (3 * x3 + 3 * x4 + 4 * (x5 + x6))))) + ml2 * ml2 *
           (x1 * x1 * (-(V2 * (x5 - x6) * (x2 * x2 + 2 * x5 * x6 - x2 *
                                           (x5 + x6))) - 2 * VA *
                       (x2 * x2 * (x5 + x6) + 2 * x5 * x6 * (x5 + x6) - x2 *
                        (x5 * x5 + 4 * x5 * x6 + x6 * x6))) + 2 * (Mn * Mn) *
            (V2 *
             (-2 * (x2 * x2 * x2 * x2) + x2 * x2 * x2 *
              (x3 + x4 + 4 * x5) + 2 * x1 * x5 * (x5 - x6) * x6 + x1 * x2 *
              (-(x5 * x5) + x6 * x6) + x2 * x2 *
              (-2 * x5 * (x3 - x4 + 2 * x5) + x1 * (x5 + x6))) + 2 * VA *
             (x2 * x2 * (-2 * (x2 * x2) - 2 * (x3 + 2 * x5) * (x5 + x6) + x2 *
                         (3 * x3 + x4 + 6 * x5 + 2 * x6)) + x1 *
              (x2 * x2 * x2 + x2 * x2 * (x5 + x6) + 2 * x5 * x6 *
               (x5 + x6) - x2 *
               (x5 * x5 + 4 * x5 * x6 + x6 * x6))))) + ml1 * ml1 *
           (x1 * x1 * (V2 * (x5 - x6) * (x2 * x2 + 2 * x5 * x6 - x2 *
                                         (x5 + x6)) + 2 * VA *
                       (x2 * x2 * (x5 + x6) + 2 * x5 * x6 * (x5 + x6) - x2 *
                        (x5 * x5 + 4 * x5 * x6 + x6 * x6))) - 2 * (Mn * Mn) *
            (V2 *
             (2 * (x2 * x2 * x2 * x2) + 2 * x1 * x5 *
              (x5 - x6) * x6 - x2 * x2 * x2 * (x3 + x4 + 4 * x6) + x1 * x2 *
              (-(x5 * x5) + x6 * x6) - x2 * x2 *
              (2 * (x3 - x4 - 2 * x6) * x6 + x1 * (x5 + x6))) + 2 * VA *
             (x2 * x2 * (-2 * (x2 * x2) - 2 * (x5 + x6) * (x4 + 2 * x6) + x2 *
                         (x3 + 3 * x4 + 2 * x5 + 6 * x6)) + x1 *
              (x2 * x2 * x2 + x2 * x2 * (x5 + x6) + 2 * x5 * x6 *
               (x5 + x6) - x2 * (x5 * x5 + 4 * x5 * x6 + x6 * x6)))))) + A2 *
          (2 * (Enu * Enu) * (Mn * Mn) * x1 *
           (-2 * x1 * (x2 * x2 * x2 * x2) + 4 * (x2 * x2 * x2 * x2 * x2) + x1 *
            (x2 * x2 * x2) * x3 - 6 * (x2 * x2 * x2 * x2) * x3 + 2 *
            (x2 * x2 * x2) * (x3 * x3) + x1 * (x2 * x2 * x2) * x4 - 6 *
            (x2 * x2 * x2 * x2) * x4 + 4 * (x2 * x2 * x2) * x3 * x4 + 2 *
            (x2 * x2 * x2) * (x4 * x4) + 4 * (x1 * x1) *
            (x2 * x2) * x5 - 6 * x1 * (x2 * x2 * x2) * x5 - 8 *
            (x2 * x2 * x2 * x2) * x5 + 8 * x1 * (x2 * x2) * x3 * x5 + 6 *
            (x2 * x2 * x2) * x3 * x5 + 8 * x1 * (x2 * x2) * x4 * x5 + 6 *
            (x2 * x2 * x2) * x4 * x5 + 2 * (x2 * x2) * x3 * x4 * x5 - 2 *
            (x2 * x2) * (x4 * x4) * x5 - 3 * (x1 * x1) * x2 *
            (x5 * x5) + 12 * x1 * (x2 * x2) * (x5 * x5) + 8 * (x2 * x2 * x2) *
            (x5 * x5) - 6 * x1 * x2 * x3 * (x5 * x5) - 4 * (x2 * x2) * x3 *
            (x5 * x5) - 12 * x1 * x2 * x4 * (x5 * x5) - 6 * x1 * x2 *
            (x5 * x5 * x5) - 4 * (x2 * x2) * (x5 * x5 * x5) + 6 * x1 * x4 *
            (x5 * x5 * x5) + 2 * (ml1 * ml1 * ml1) * ml2 * (x2 * x2) *
            (x2 - x5 - x6) + ml1 * ml1 * ml1 * ml1 * (x2 * x2) *
            (x5 - x6) + 4 * (x1 * x1) * (x2 * x2) * x6 - 6 * x1 *
            (x2 * x2 * x2) * x6 - 8 * (x2 * x2 * x2 * x2) * x6 + 8 * x1 *
            (x2 * x2) * x3 * x6 + 6 * (x2 * x2 * x2) * x3 * x6 - 2 *
            (x2 * x2) * (x3 * x3) * x6 + 8 * x1 * (x2 * x2) * x4 * x6 + 6 *
            (x2 * x2 * x2) * x4 * x6 + 2 * (x2 * x2) * x3 * x4 * x6 + 16 * x1 *
            (x2 * x2) * x5 * x6 + 8 *
            (x2 * x2 * x2) * x5 * x6 - 6 * x1 * x2 * x3 * x5 * x6 - 4 *
            (x2 * x2) * x3 * x5 * x6 - 6 * x1 * x2 * x4 * x5 * x6 - 4 *
            (x2 * x2) * x4 * x5 * x6 - 18 * x1 * x2 * (x5 * x5) * x6 - 4 *
            (x2 * x2) * (x5 * x5) * x6 + 6 * x1 * x3 *
            (x5 * x5) * x6 + 12 * x1 * (x5 * x5 * x5) * x6 - 3 *
            (x1 * x1) * x2 * (x6 * x6) + 12 * x1 * (x2 * x2) * (x6 * x6) + 8 *
            (x2 * x2 * x2) * (x6 * x6) - 12 * x1 * x2 * x3 *
            (x6 * x6) - 6 * x1 * x2 * x4 * (x6 * x6) - 4 * (x2 * x2) * x4 *
            (x6 * x6) - 18 * x1 * x2 * x5 * (x6 * x6) - 4 * (x2 * x2) * x5 *
            (x6 * x6) + 6 * x1 * x4 * x5 * (x6 * x6) - 6 * x1 * x2 *
            (x6 * x6 * x6) - 4 * (x2 * x2) * (x6 * x6 * x6) + 6 * x1 * x3 *
            (x6 * x6 * x6) + 12 * x1 * x5 *
            (x6 * x6 * x6) + ml2 * ml2 * ml2 * ml2 * (x2 * x2) *
            (-x5 + x6) + ml2 * ml2 * (2 * (x2 * x2 * x2 * x2) - x2 * x2 * x2 *
                                      (x3 + x4 + 4 * x5) + 6 * x1 * x5 * x6 *
                                      (-x5 + x6) + 3 * x1 * x2 *
                                      (x5 * x5 - x6 * x6) + x2 * x2 *
                                      (2 * x5 * (x3 - x4 + 2 * x5) + x1 *
                                       (-3 * x5 + x6))) + ml1 * ml1 *
            (2 * (x2 * x2 * x2 * x2) + 6 * x1 * x5 *
             (x5 - x6) * x6 - x2 * x2 * x2 * (x3 + x4 + 4 * x6) + 3 * x1 * x2 *
             (-(x5 * x5) + x6 * x6) + x2 * x2 *
             (x1 * (x5 - 3 * x6) + 2 * x6 *
              (-x3 + x4 + 2 * x6))) + 2 * ml1 * ml2 *
            (ml2 * ml2 * (x2 * x2) * (x2 - x5 - x6) + x1 *
             (x2 * x2 * x2 - 6 * (x2 * x2) * (x5 + x6) - 6 * x5 * x6 *
              (x5 + x6) + 3 * x2 *
              (x5 * x5 + 4 * x5 * x6 + x6 * x6)) + x2 * x2 *
             (4 * (x2 * x2) + (x5 + x6) * (x3 + x4 + 2 * (x5 + x6)) - x2 *
              (3 * x3 + 3 * x4 + 4 * (x5 + x6))))) - 2 * Enu * Mn * x1 * x2 *
           (-2 * x1 * (x2 * x2 * x2 * x2) + 4 * (x2 * x2 * x2 * x2 * x2) + x1 *
            (x2 * x2 * x2) * x3 - 6 * (x2 * x2 * x2 * x2) * x3 + 2 *
            (x2 * x2 * x2) * (x3 * x3) + x1 * (x2 * x2 * x2) * x4 - 6 *
            (x2 * x2 * x2 * x2) * x4 + 4 * (x2 * x2 * x2) * x3 * x4 + 2 *
            (x2 * x2 * x2) * (x4 * x4) + 4 * (x1 * x1) *
            (x2 * x2) * x5 - 6 * x1 * (x2 * x2 * x2) * x5 - 8 *
            (x2 * x2 * x2 * x2) * x5 + 8 * x1 * (x2 * x2) * x3 * x5 + 6 *
            (x2 * x2 * x2) * x3 * x5 + 8 * x1 * (x2 * x2) * x4 * x5 + 6 *
            (x2 * x2 * x2) * x4 * x5 + 2 * (x2 * x2) * x3 * x4 * x5 - 2 *
            (x2 * x2) * (x4 * x4) * x5 - 3 * (x1 * x1) * x2 *
            (x5 * x5) + 12 * x1 * (x2 * x2) * (x5 * x5) + 8 * (x2 * x2 * x2) *
            (x5 * x5) - 6 * x1 * x2 * x3 * (x5 * x5) - 4 * (x2 * x2) * x3 *
            (x5 * x5) - 12 * x1 * x2 * x4 * (x5 * x5) - 6 * x1 * x2 *
            (x5 * x5 * x5) - 4 * (x2 * x2) * (x5 * x5 * x5) + 6 * x1 * x4 *
            (x5 * x5 * x5) + 2 * (ml1 * ml1 * ml1) * ml2 * (x2 * x2) *
            (x2 - x5 - x6) + ml1 * ml1 * ml1 * ml1 * (x2 * x2) *
            (x5 - x6) + 4 * (x1 * x1) * (x2 * x2) * x6 - 6 * x1 *
            (x2 * x2 * x2) * x6 - 8 * (x2 * x2 * x2 * x2) * x6 + 8 * x1 *
            (x2 * x2) * x3 * x6 + 6 * (x2 * x2 * x2) * x3 * x6 - 2 *
            (x2 * x2) * (x3 * x3) * x6 + 8 * x1 * (x2 * x2) * x4 * x6 + 6 *
            (x2 * x2 * x2) * x4 * x6 + 2 * (x2 * x2) * x3 * x4 * x6 + 16 * x1 *
            (x2 * x2) * x5 * x6 + 8 *
            (x2 * x2 * x2) * x5 * x6 - 6 * x1 * x2 * x3 * x5 * x6 - 4 *
            (x2 * x2) * x3 * x5 * x6 - 6 * x1 * x2 * x4 * x5 * x6 - 4 *
            (x2 * x2) * x4 * x5 * x6 - 18 * x1 * x2 * (x5 * x5) * x6 - 4 *
            (x2 * x2) * (x5 * x5) * x6 + 6 * x1 * x3 *
            (x5 * x5) * x6 + 12 * x1 * (x5 * x5 * x5) * x6 - 3 *
            (x1 * x1) * x2 * (x6 * x6) + 12 * x1 * (x2 * x2) * (x6 * x6) + 8 *
            (x2 * x2 * x2) * (x6 * x6) - 12 * x1 * x2 * x3 *
            (x6 * x6) - 6 * x1 * x2 * x4 * (x6 * x6) - 4 * (x2 * x2) * x4 *
            (x6 * x6) - 18 * x1 * x2 * x5 * (x6 * x6) - 4 * (x2 * x2) * x5 *
            (x6 * x6) + 6 * x1 * x4 * x5 * (x6 * x6) - 6 * x1 * x2 *
            (x6 * x6 * x6) - 4 * (x2 * x2) * (x6 * x6 * x6) + 6 * x1 * x3 *
            (x6 * x6 * x6) + 12 * x1 * x5 *
            (x6 * x6 * x6) + ml2 * ml2 * ml2 * ml2 * (x2 * x2) *
            (-x5 + x6) + ml2 * ml2 * (2 * (x2 * x2 * x2 * x2) - x2 * x2 * x2 *
                                      (x3 + x4 + 4 * x5) + 6 * x1 * x5 * x6 *
                                      (-x5 + x6) + 3 * x1 * x2 *
                                      (x5 * x5 - x6 * x6) + x2 * x2 *
                                      (2 * x5 * (x3 - x4 + 2 * x5) + x1 *
                                       (-3 * x5 + x6))) + ml1 * ml1 *
            (2 * (x2 * x2 * x2 * x2) + 6 * x1 * x5 *
             (x5 - x6) * x6 - x2 * x2 * x2 * (x3 + x4 + 4 * x6) + 3 * x1 * x2 *
             (-(x5 * x5) + x6 * x6) + x2 * x2 *
             (x1 * (x5 - 3 * x6) + 2 * x6 *
              (-x3 + x4 + 2 * x6))) + 2 * ml1 * ml2 *
            (ml2 * ml2 * (x2 * x2) * (x2 - x5 - x6) + x1 *
             (x2 * x2 * x2 - 6 * (x2 * x2) * (x5 + x6) - 6 * x5 * x6 *
              (x5 + x6) + 3 * x2 *
              (x5 * x5 + 4 * x5 * x6 + x6 * x6)) + x2 * x2 *
             (4 * (x2 * x2) + (x5 + x6) * (x3 + x4 + 2 * (x5 + x6)) - x2 *
              (3 * x3 + 3 * x4 + 4 * (x5 + x6))))) + x2 * x2 *
           (-4 * (ml2 * ml2) * (Mn * Mn) * (x2 * x2 * x2 * x2) + 4 *
            (Mn * Mn) * x1 * (x2 * x2 * x2 * x2) - 8 * (Mn * Mn) *
            (x2 * x2 * x2 * x2 * x2) + 2 * (ml2 * ml2) * (Mn * Mn) *
            (x2 * x2 * x2) * x3 - 2 * (Mn * Mn) * x1 *
            (x2 * x2 * x2) * x3 + 12 * (Mn * Mn) *
            (x2 * x2 * x2 * x2) * x3 - 4 * (Mn * Mn) * (x2 * x2 * x2) *
            (x3 * x3) + 2 * (ml2 * ml2) * (Mn * Mn) * (x2 * x2 * x2) * x4 - 2 *
            (Mn * Mn) * x1 * (x2 * x2 * x2) * x4 + 12 * (Mn * Mn) *
            (x2 * x2 * x2 * x2) * x4 - 8 * (Mn * Mn) *
            (x2 * x2 * x2) * x3 * x4 - 4 * (Mn * Mn) * (x2 * x2 * x2) *
            (x4 * x4) + 2 * (ml2 * ml2 * ml2 * ml2) * (Mn * Mn) *
            (x2 * x2) * x5 + 2 * (ml2 * ml2) * (Mn * Mn) * x1 *
            (x2 * x2) * x5 - ml2 * ml2 * (x1 * x1) * (x2 * x2) * x5 - 4 *
            (Mn * Mn) * (x1 * x1) * (x2 * x2) * x5 + x1 * x1 * x1 *
            (x2 * x2) * x5 + 8 * (ml2 * ml2) * (Mn * Mn) *
            (x2 * x2 * x2) * x5 + 4 * (Mn * Mn) * x1 *
            (x2 * x2 * x2) * x5 - 2 * (x1 * x1) * (x2 * x2 * x2) * x5 + 16 *
            (Mn * Mn) * (x2 * x2 * x2 * x2) * x5 - 4 * (ml2 * ml2) *
            (Mn * Mn) * (x2 * x2) * x3 * x5 - 8 * (Mn * Mn) * x1 *
            (x2 * x2) * x3 * x5 + 2 * (x1 * x1) * (x2 * x2) * x3 * x5 - 12 *
            (Mn * Mn) * (x2 * x2 * x2) * x3 * x5 + 4 * (ml2 * ml2) *
            (Mn * Mn) * (x2 * x2) * x4 * x5 - 8 * (Mn * Mn) * x1 *
            (x2 * x2) * x4 * x5 + 2 * (x1 * x1) * (x2 * x2) * x4 * x5 - 12 *
            (Mn * Mn) * (x2 * x2 * x2) * x4 * x5 - 4 * (Mn * Mn) *
            (x2 * x2) * x3 * x4 * x5 + 4 * (Mn * Mn) * (x2 * x2) *
            (x4 * x4) * x5 - 2 * (ml2 * ml2) * (Mn * Mn) * x1 * x2 *
            (x5 * x5) + ml2 * ml2 * (x1 * x1) * x2 * (x5 * x5) + 2 *
            (Mn * Mn) * (x1 * x1) * x2 * (x5 * x5) - x1 * x1 * x1 * x2 *
            (x5 * x5) - 8 * (ml2 * ml2) * (Mn * Mn) * (x2 * x2) *
            (x5 * x5) - 8 * (Mn * Mn) * x1 * (x2 * x2) * (x5 * x5) + 4 *
            (x1 * x1) * (x2 * x2) * (x5 * x5) - 16 * (Mn * Mn) *
            (x2 * x2 * x2) * (x5 * x5) + 4 * (Mn * Mn) * x1 * x2 * x3 *
            (x5 * x5) - 2 * (x1 * x1) * x2 * x3 * (x5 * x5) + 8 * (Mn * Mn) *
            (x2 * x2) * x3 * (x5 * x5) + 8 * (Mn * Mn) * x1 * x2 * x4 *
            (x5 * x5) - 4 * (x1 * x1) * x2 * x4 * (x5 * x5) + 4 *
            (Mn * Mn) * x1 * x2 * (x5 * x5 * x5) - 2 * (x1 * x1) * x2 *
            (x5 * x5 * x5) + 8 * (Mn * Mn) * (x2 * x2) * (x5 * x5 * x5) - 4 *
            (Mn * Mn) * x1 * x4 * (x5 * x5 * x5) + 2 * (x1 * x1) * x4 *
            (x5 * x5 * x5) - 2 * (ml2 * ml2 * ml2 * ml2) * (Mn * Mn) *
            (x2 * x2) * x6 + 2 * (ml2 * ml2) * (Mn * Mn) * x1 *
            (x2 * x2) * x6 + ml2 * ml2 * (x1 * x1) * (x2 * x2) * x6 - 4 *
            (Mn * Mn) * (x1 * x1) * (x2 * x2) * x6 + x1 * x1 * x1 *
            (x2 * x2) * x6 + 4 * (Mn * Mn) * x1 * (x2 * x2 * x2) * x6 - 2 *
            (x1 * x1) * (x2 * x2 * x2) * x6 + 16 * (Mn * Mn) *
            (x2 * x2 * x2 * x2) * x6 - 8 * (Mn * Mn) * x1 *
            (x2 * x2) * x3 * x6 + 2 * (x1 * x1) * (x2 * x2) * x3 * x6 - 12 *
            (Mn * Mn) * (x2 * x2 * x2) * x3 * x6 + 4 * (Mn * Mn) * (x2 * x2) *
            (x3 * x3) * x6 - 8 * (Mn * Mn) * x1 * (x2 * x2) * x4 * x6 + 2 *
            (x1 * x1) * (x2 * x2) * x4 * x6 - 12 * (Mn * Mn) *
            (x2 * x2 * x2) * x4 * x6 - 4 * (Mn * Mn) *
            (x2 * x2) * x3 * x4 * x6 - 16 * (Mn * Mn) * x1 *
            (x2 * x2) * x5 * x6 + 4 * (x1 * x1) * (x2 * x2) * x5 * x6 - 16 *
            (Mn * Mn) * (x2 * x2 * x2) * x5 * x6 + 4 *
            (Mn * Mn) * x1 * x2 * x3 * x5 * x6 - 2 *
            (x1 * x1) * x2 * x3 * x5 * x6 + 8 * (Mn * Mn) *
            (x2 * x2) * x3 * x5 * x6 + 4 *
            (Mn * Mn) * x1 * x2 * x4 * x5 * x6 - 2 *
            (x1 * x1) * x2 * x4 * x5 * x6 + 8 * (Mn * Mn) *
            (x2 * x2) * x4 * x5 * x6 + 4 * (ml2 * ml2) * (Mn * Mn) * x1 *
            (x5 * x5) * x6 - 2 * (ml2 * ml2) * (x1 * x1) *
            (x5 * x5) * x6 + 12 * (Mn * Mn) * x1 * x2 * (x5 * x5) * x6 - 6 *
            (x1 * x1) * x2 * (x5 * x5) * x6 + 8 * (Mn * Mn) * (x2 * x2) *
            (x5 * x5) * x6 - 4 * (Mn * Mn) * x1 * x3 * (x5 * x5) * x6 + 2 *
            (x1 * x1) * x3 * (x5 * x5) * x6 - 8 * (Mn * Mn) * x1 *
            (x5 * x5 * x5) * x6 + 4 * (x1 * x1) * (x5 * x5 * x5) * x6 + 2 *
            (ml2 * ml2) * (Mn * Mn) * x1 * x2 * (x6 * x6) - ml2 * ml2 *
            (x1 * x1) * x2 * (x6 * x6) + 2 * (Mn * Mn) * (x1 * x1) * x2 *
            (x6 * x6) - x1 * x1 * x1 * x2 * (x6 * x6) - 8 * (Mn * Mn) * x1 *
            (x2 * x2) * (x6 * x6) + 4 * (x1 * x1) * (x2 * x2) *
            (x6 * x6) - 16 * (Mn * Mn) * (x2 * x2 * x2) * (x6 * x6) + 8 *
            (Mn * Mn) * x1 * x2 * x3 * (x6 * x6) - 4 * (x1 * x1) * x2 * x3 *
            (x6 * x6) + 4 * (Mn * Mn) * x1 * x2 * x4 * (x6 * x6) - 2 *
            (x1 * x1) * x2 * x4 * (x6 * x6) + 8 * (Mn * Mn) * (x2 * x2) * x4 *
            (x6 * x6) - 4 * (ml2 * ml2) * (Mn * Mn) * x1 * x5 * (x6 * x6) + 2 *
            (ml2 * ml2) * (x1 * x1) * x5 * (x6 * x6) + 12 *
            (Mn * Mn) * x1 * x2 * x5 * (x6 * x6) - 6 * (x1 * x1) * x2 * x5 *
            (x6 * x6) + 8 * (Mn * Mn) * (x2 * x2) * x5 * (x6 * x6) - 4 *
            (Mn * Mn) * x1 * x4 * x5 * (x6 * x6) + 2 * (x1 * x1) * x4 * x5 *
            (x6 * x6) + 4 * (Mn * Mn) * x1 * x2 * (x6 * x6 * x6) - 2 *
            (x1 * x1) * x2 * (x6 * x6 * x6) + 8 * (Mn * Mn) * (x2 * x2) *
            (x6 * x6 * x6) - 4 * (Mn * Mn) * x1 * x3 * (x6 * x6 * x6) + 2 *
            (x1 * x1) * x3 * (x6 * x6 * x6) - 8 * (Mn * Mn) * x1 * x5 *
            (x6 * x6 * x6) + 4 * (x1 * x1) * x5 * (x6 * x6 * x6) + 2 *
            (ml1 * ml1 * ml1 * ml1) * (Mn * Mn) * (x2 * x2) * (-x5 + x6) + 4 *
            (ml1 * ml1 * ml1) * ml2 * (Mn * Mn) * (x2 * x2) *
            (-x2 + x5 + x6) + ml1 * ml1 *
            (x1 * x1 * (x5 - x6) * (x2 * x2 + 2 * x5 * x6 - x2 *
                                    (x5 + x6)) + 2 * (Mn * Mn) *
             (-2 * (x2 * x2 * x2 * x2) + 2 * x1 * x5 * x6 *
              (-x5 + x6) + x2 * x2 * x2 * (x3 + x4 + 4 * x6) + x1 * x2 *
              (x5 * x5 - x6 * x6) + x2 * x2 * (2 *
                                               (x3 - x4 - 2 * x6) * x6 + x1 *
                                               (x5 + x6)))) - 2 * ml1 * ml2 *
            (2 * (ml2 * ml2) * (Mn * Mn) * (x2 * x2) *
             (x2 - x5 - x6) - x1 * x1 *
             (x2 * x2 * x2 - 2 * (x2 * x2) * (x5 + x6) - 2 * x5 * x6 *
              (x5 + x6) + x2 *
              (x5 * x5 + 4 * x5 * x6 + x6 * x6)) + 2 * (Mn * Mn) *
             (-(x1 * (x2 * x2 * x2 + 2 * (x2 * x2) * (x5 + x6) + 2 * x5 * x6 *
                      (x5 + x6) - x2 *
                      (x5 * x5 + 4 * x5 * x6 + x6 * x6))) + x2 * x2 *
              (4 * (x2 * x2) + (x5 + x6) * (x3 + x4 + 2 * (x5 + x6)) - x2 *
               (3 * x3 + 3 * x4 + 4 * (x5 + x6))))))))) *
        (Z * Z)) / (Enu * Enu * (Mn * Mn) * (x1 * x1) * (x2 * x2 * x2 * x2) *
                    ((x1 + 2 * x3) * (x1 + 2 * x3)) * ((x1 + 2 * x4) *
                                                       (x1 + 2 * x4)))

    return diff_xsec * invGeV2_to_attobarn