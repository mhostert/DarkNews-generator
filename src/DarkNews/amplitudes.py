import numpy as np
import scipy
import vegas as vg
import random 

#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )
from . import Cfourvec as Cfv


from . import pdg 
from . import const
from .const import *
from . import fourvec
from . import phase_space
from . import decay_rates as dr

from DarkNews import logger


def upscattering_dxsec_dQ2(mandelstamm, process):

    # kinematics
    s,t,u = mandelstamm
    Q2 = -t

    # hadronic target
    target = process.target

    # masses
    M = process.target.mass
    MA= M
    mHNL = process.m_ups
    mzprime = process.mzprime

    # vertices
    Chad = process.Chad
    Cprimehad = process.Cprimehad
    Vhad = process.Vhad
    Cij = process.Cij
    Cji = process.Cji
    Vij = process.Vij
    Vji = process.Vji

    MAJ = process.MAJ
    h = process.h_upscattered


    # Form factors
    if target.is_nucleus:
        FFf1 = target.F1_EM(Q2)
        FFf2 = 0.0 ### FIX ME 
        FFga = 0.0
        FFgp = 0.0

        FFNCf1 = target.F1_NC(Q2)  ### FIX ME 
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
        logger.error('upscattering on a lepton not implemented yet.')
        
    ## Spin summed (but not averaged) matrix elements from MATHEMATICA
    # |M|^2 = | M_SM + M_kinmix + M_massmix|^2

    if process.TheoryModel.HNLtype == 'majorana':
        # SM NC SQR
        Lmunu_Hmunu = (Chad*Chad*Cij*Cji*(16*FFNCga*FFNCgp*(mHNL*mHNL)*(-(mHNL*mHNL) + t) + (4*(FFNCgp*FFNCgp)*(mHNL*mHNL)*t*(-(mHNL*mHNL) + t))/(M*M) - 8*FFNCf1*FFNCf2*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*(t*t)) + 8*(FFNCf1*FFNCf1)*(2*(M*M*M*M) - 4*(M*M)*s + 2*(s*s) + 2*s*t + t*t - mHNL*mHNL*(2*s + t)) + 8*(FFNCga*FFNCga)*(2*(M*M*M*M) + 2*(s*s) + 4*(M*M)*(mHNL*mHNL - s - t) + 2*s*t + t*t - mHNL*mHNL*(2*s + t)) - (FFNCf2*FFNCf2*(4*(M*M*M*M)*t + 4*(M*M)*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*t*(s + t)) + t*(mHNL*mHNL*mHNL*mHNL + 4*s*(s + t) - mHNL*mHNL*(4*s + t))))/(M*M) + (16*FFNCf1*FFNCga*h*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t))))/(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))) + (16*FFNCf2*FFNCga*h*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t))))/(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s)))))/(2.*((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)))

        if process.TheoryModel.is_kinetically_mixed: 

            # kinetic mixing term SQR
            Lmunu_Hmunu += ((-8*FFf1*FFf2*(M*M)*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*(t*t)) + 8*(FFf1*FFf1)*(M*M)*(2*(M*M*M*M) - 4*(M*M)*s + 2*(s*s) + 2*s*t + t*t - mHNL*mHNL*(2*s + t)) - FFf2*FFf2*(4*(M*M*M*M)*t + 4*(M*M)*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*t*(s + t)) + t*(mHNL*mHNL*mHNL*mHNL + 4*s*(s + t) - mHNL*mHNL*(4*s + t))))*(Vhad*Vhad)*Vij*Vji)/(2.*(M*M)*((mzprime*mzprime - t)*(mzprime*mzprime - t)))

            # # kinetic mixing + SM NC interference
            Lmunu_Hmunu += ((Chad*(4*FFf1*(M*M)*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(-(FFNCf2*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*(t*t))) + 2*FFNCf1*(2*(M*M*M*M) - 4*(M*M)*s + 2*(s*s) + 2*s*t + t*t - mHNL*mHNL*(2*s + t))) + 2*FFNCga*h*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t)))) + FFf2*(8*FFNCga*h*(M*M)*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t))) - s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(4*FFNCf1*(M*M)*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*(t*t)) + FFNCf2*(4*(M*M*M*M)*t + 4*(M*M)*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*t*(s + t)) + t*(mHNL*mHNL*mHNL*mHNL + 4*s*(s + t) - mHNL*mHNL*(4*s + t))))))*Vhad*(Cji*Vij + Cij*Vji))/(M*M*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)))

        if process.TheoryModel.is_mass_mixed: 
            
            # mass mixing + SM NC interference
            Lmunu_Hmunu += -((Sqrt(Chad*Cprimehad)*(4*FFf1*(M*M)*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(-(FFNCf2*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*(t*t))) + 2*FFNCf1*(2*(M*M*M*M) - 4*(M*M)*s + 2*(s*s) + 2*s*t + t*t - mHNL*mHNL*(2*s + t))) + 2*FFNCga*h*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t)))) + FFf2*(8*FFNCga*h*(M*M)*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t))) - s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(4*FFNCf1*(M*M)*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*(t*t)) + FFNCf2*(4*(M*M*M*M)*t + 4*(M*M)*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*t*(s + t)) + t*(mHNL*mHNL*mHNL*mHNL + 4*s*(s + t) - mHNL*mHNL*(4*s + t))))))*Vhad*(Cji*Vij + Cij*Vji))/(M*M*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t)))
            
            # kinetic mixing + mass mixing interference
            Lmunu_Hmunu += (Cprimehad*(4*FFf1*(M*M)*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(-(FFNCf2*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*(t*t))) + 2*FFNCf1*(2*(M*M*M*M) - 4*(M*M)*s + 2*(s*s) + 2*s*t + t*t - mHNL*mHNL*(2*s + t))) + 2*FFNCga*h*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t)))) + FFf2*(8*FFNCga*h*(M*M)*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t))) - s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(4*FFNCf1*(M*M)*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*(t*t)) + FFNCf2*(4*(M*M*M*M)*t + 4*(M*M)*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*t*(s + t)) + t*(mHNL*mHNL*mHNL*mHNL + 4*s*(s + t) - mHNL*mHNL*(4*s + t))))))*Vhad*Vij*Vji)/(M*M*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*((mzprime*mzprime - t)*(mzprime*mzprime - t)))

            # mass mixing term SQR
            Lmunu_Hmunu += (Cprimehad*Cprimehad*(16*FFNCga*FFNCgp*(mHNL*mHNL)*(-(mHNL*mHNL) + t) + (4*(FFNCgp*FFNCgp)*(mHNL*mHNL)*t*(-(mHNL*mHNL) + t))/(M*M) - 8*FFNCf1*FFNCf2*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*(t*t)) + 8*(FFNCf1*FFNCf1)*(2*(M*M*M*M) - 4*(M*M)*s + 2*(s*s) + 2*s*t + t*t - mHNL*mHNL*(2*s + t)) + 8*(FFNCga*FFNCga)*(2*(M*M*M*M) + 2*(s*s) + 4*(M*M)*(mHNL*mHNL - s - t) + 2*s*t + t*t - mHNL*mHNL*(2*s + t)) - (FFNCf2*FFNCf2*(4*(M*M*M*M)*t + 4*(M*M)*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*t*(s + t)) + t*(mHNL*mHNL*mHNL*mHNL + 4*s*(s + t) - mHNL*mHNL*(4*s + t))))/(M*M) + (16*FFNCf1*FFNCga*h*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t))))/(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))) + (16*FFNCf2*FFNCga*h*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t))))/(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))))*Vij*Vji)/(2.*((mzprime*mzprime - t)*(mzprime*mzprime - t)))


    elif process.TheoryModel.HNLtype == 'dirac':

        # SM NC SQR
        Lmunu_Hmunu = (Chad*Chad*Cij*Cji*(-4*(FFNCf2*FFNCf2)*h*(M*M*M*M)*(mHNL*mHNL*mHNL*mHNL) - 4*(FFNCf2*FFNCf2)*h*(M*M)*(mHNL*mHNL*mHNL*mHNL*mHNL*mHNL) + 4*(FFNCf2*FFNCf2)*h*(M*M)*(mHNL*mHNL*mHNL*mHNL)*s - 4*(FFNCf2*FFNCf2)*(M*M)*(mHNL*mHNL*mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s)) - 4*(FFNCf2*FFNCf2)*h*(M*M*M*M*M*M)*t + 8*(FFNCf2*FFNCf2)*h*(M*M*M*M)*(mHNL*mHNL)*t + 11*(FFNCf2*FFNCf2)*h*(M*M)*(mHNL*mHNL*mHNL*mHNL)*t - 4*(FFNCgp*FFNCgp)*h*(M*M)*(mHNL*mHNL*mHNL*mHNL)*t - FFNCf2*FFNCf2*h*(mHNL*mHNL*mHNL*mHNL*mHNL*mHNL)*t - 4*(FFNCgp*FFNCgp)*h*(mHNL*mHNL*mHNL*mHNL*mHNL*mHNL)*t + 12*(FFNCf2*FFNCf2)*h*(M*M*M*M)*s*t + 5*(FFNCf2*FFNCf2)*h*(mHNL*mHNL*mHNL*mHNL)*s*t + 4*(FFNCgp*FFNCgp)*h*(mHNL*mHNL*mHNL*mHNL)*s*t - 12*(FFNCf2*FFNCf2)*h*(M*M)*(s*s)*t - 8*(FFNCf2*FFNCf2)*h*(mHNL*mHNL)*(s*s)*t + 4*(FFNCf2*FFNCf2)*h*(s*s*s)*t - 4*(FFNCf2*FFNCf2)*(M*M*M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 4*(FFNCf2*FFNCf2)*(M*M)*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - FFNCf2*FFNCf2*(mHNL*mHNL*mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 4*(FFNCgp*FFNCgp)*(mHNL*mHNL*mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 8*(FFNCf2*FFNCf2)*(M*M)*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 4*(FFNCf2*FFNCf2)*(mHNL*mHNL)*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 4*(FFNCf2*FFNCf2)*(s*s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 8*(FFNCf2*FFNCf2)*h*(M*M*M*M)*(t*t) - 9*(FFNCf2*FFNCf2)*h*(M*M)*(mHNL*mHNL)*(t*t) - 4*(FFNCgp*FFNCgp)*h*(M*M)*(mHNL*mHNL)*(t*t) + FFNCf2*FFNCf2*h*(mHNL*mHNL*mHNL*mHNL)*(t*t) + 4*(FFNCgp*FFNCgp)*h*(mHNL*mHNL*mHNL*mHNL)*(t*t) - 12*(FFNCf2*FFNCf2)*h*(M*M)*s*(t*t) - 3*(FFNCf2*FFNCf2)*h*(mHNL*mHNL)*s*(t*t) + 4*(FFNCgp*FFNCgp)*h*(mHNL*mHNL)*s*(t*t) + 4*(FFNCf2*FFNCf2)*h*(s*s)*(t*t) + 8*(FFNCf2*FFNCf2)*(M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) + FFNCf2*FFNCf2*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) + 4*(FFNCgp*FFNCgp)*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) - 4*(FFNCf2*FFNCf2)*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) + 8*(FFNCf1*FFNCf1)*(M*M)*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(2*(M*M*M*M) - 4*(M*M)*s + 2*(s*s) + 2*s*t + t*t - mHNL*mHNL*(2*s + t)) + h*(2*(M*M*M*M*M*M) - 2*((mHNL*mHNL - s)*(mHNL*mHNL - s))*s - 2*(M*M*M*M)*(mHNL*mHNL + 3*s) + (mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*s - 2*(s*s))*t - (mHNL*mHNL + s)*(t*t) + M*M*(6*(s*s) + 2*s*t + t*t + mHNL*mHNL*(-2*s + t)))) - 8*FFNCf1*(M*M)*(2*FFNCga*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t*(-2*(M*M) - mHNL*mHNL + 2*s + t) + FFNCf2*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*(t*t)) + FFNCf2*h*(mHNL*mHNL - 2*t)*(mHNL*mHNL*mHNL*mHNL - s*t + M*M*(mHNL*mHNL + t) - mHNL*mHNL*(s + t)) + 2*FFNCga*h*(-2*(M*M*M*M)*t + t*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*(s - t) - s*(2*s + t)) + M*M*(-4*(mHNL*mHNL*mHNL*mHNL) + 3*(mHNL*mHNL)*t + t*(4*s + t)))) + 8*(FFNCga*FFNCga)*(M*M)*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(2*(M*M*M*M) + 2*(s*s) + 4*(M*M)*(mHNL*mHNL - s - t) + 2*s*t + t*t - mHNL*mHNL*(2*s + t)) + h*(2*(M*M*M*M*M*M) - 2*((mHNL*mHNL - s)*(mHNL*mHNL - s))*s + (mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*s - 2*(s*s))*t - (mHNL*mHNL + s)*(t*t) - 2*(M*M*M*M)*(3*(mHNL*mHNL + s) + 2*t) + M*M*(-4*(mHNL*mHNL*mHNL*mHNL) + 6*(s*s) + 6*s*t + t*t + mHNL*mHNL*(2*s + 5*t)))) + 16*FFNCga*(M*M)*(FFNCgp*(mHNL*mHNL)*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(-(mHNL*mHNL) + t) + h*(-(mHNL*mHNL*mHNL*mHNL) + s*t - M*M*(mHNL*mHNL + t) + mHNL*mHNL*(s + t))) + FFNCf2*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(2*(M*M) + mHNL*mHNL - 2*s - t)*t + h*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t)))))))/(4.*(M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*((MZBOSON*MZBOSON - t)*(MZBOSON*MZBOSON - t)))


        if process.TheoryModel.is_kinetically_mixed: 

            # kinetic mixing term SQR
            Lmunu_Hmunu += ((-(FFf2*FFf2*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(4*(M*M)*(mHNL*mHNL*mHNL*mHNL) + (2*(M*M) + mHNL*mHNL - 2*s)*(2*(M*M) + mHNL*mHNL - 2*s)*t - (8*(M*M) + mHNL*mHNL - 4*s)*(t*t)) + h*(4*(M*M)*(mHNL*mHNL*mHNL*mHNL)*(M*M + mHNL*mHNL - s) + (2*(M*M) - 4*M*mHNL + mHNL*mHNL - 2*s)*(2*(M*M) + 4*M*mHNL + mHNL*mHNL - 2*s)*(M*M + mHNL*mHNL - s)*t - (8*(M*M*M*M) + mHNL*mHNL*mHNL*mHNL - 3*(mHNL*mHNL)*s + 4*(s*s) - 3*(M*M)*(3*(mHNL*mHNL) + 4*s))*(t*t)))) - 8*FFf1*FFf2*(M*M)*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*(t*t)) + h*(mHNL*mHNL - 2*t)*(mHNL*mHNL*mHNL*mHNL - s*t + M*M*(mHNL*mHNL + t) - mHNL*mHNL*(s + t))) + 8*(FFf1*FFf1)*(M*M)*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(2*(M*M*M*M) - 4*(M*M)*s + 2*(s*s) + 2*s*t + t*t - mHNL*mHNL*(2*s + t)) + h*(2*(M*M*M*M*M*M) - 2*((mHNL*mHNL - s)*(mHNL*mHNL - s))*s - 2*(M*M*M*M)*(mHNL*mHNL + 3*s) + (mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*s - 2*(s*s))*t - (mHNL*mHNL + s)*(t*t) + M*M*(6*(s*s) + 2*s*t + t*t + mHNL*mHNL*(-2*s + t)))))*(Vhad*Vhad)*Vij*Vji)/(4.*(M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*((mzprime*mzprime - t)*(mzprime*mzprime - t)))

            # # kinetic mixing + SM NC interference
            Lmunu_Hmunu += 0.5*(Chad*(-(FFf2*(4*FFNCf2*(M*M)*(mHNL*mHNL*mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s)) + 4*FFNCf2*(M*M*M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 16*FFNCga*(M*M*M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 4*FFNCf2*(M*M)*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 8*FFNCga*(M*M)*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + FFNCf2*(mHNL*mHNL*mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 8*FFNCf2*(M*M)*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 16*FFNCga*(M*M)*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 4*FFNCf2*(mHNL*mHNL)*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 4*FFNCf2*(s*s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 8*FFNCf2*(M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) + 8*FFNCga*(M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) - FFNCf2*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) + 4*FFNCf2*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) + FFNCf2*h*(4*(M*M)*(mHNL*mHNL*mHNL*mHNL)*(M*M + mHNL*mHNL - s) + (2*(M*M) - 4*M*mHNL + mHNL*mHNL - 2*s)*(2*(M*M) + 4*M*mHNL + mHNL*mHNL - 2*s)*(M*M + mHNL*mHNL - s)*t - (8*(M*M*M*M) + mHNL*mHNL*mHNL*mHNL - 3*(mHNL*mHNL)*s + 4*(s*s) - 3*(M*M)*(3*(mHNL*mHNL) + 4*s))*(t*t)) + 4*FFNCf1*(M*M)*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*(t*t)) + h*(mHNL*mHNL - 2*t)*(mHNL*mHNL*mHNL*mHNL - s*t + M*M*(mHNL*mHNL + t) - mHNL*mHNL*(s + t))) - 8*FFNCga*h*(M*M)*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t))))) + 4*FFf1*(M*M)*(-(FFNCf2*(mHNL*mHNL*mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))) + 4*FFNCga*(M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - FFNCf2*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 2*FFNCga*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 4*FFNCga*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 2*FFNCf2*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) - 2*FFNCga*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) - FFNCf2*h*(mHNL*mHNL - 2*t)*(mHNL*mHNL*mHNL*mHNL - s*t + M*M*(mHNL*mHNL + t) - mHNL*mHNL*(s + t)) + 2*FFNCga*h*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t))) + 2*FFNCf1*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(2*(M*M*M*M) - 4*(M*M)*s + 2*(s*s) + 2*s*t + t*t - mHNL*mHNL*(2*s + t)) + h*(2*(M*M*M*M*M*M) - 2*((mHNL*mHNL - s)*(mHNL*mHNL - s))*s - 2*(M*M*M*M)*(mHNL*mHNL + 3*s) + (mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*s - 2*(s*s))*t - (mHNL*mHNL + s)*(t*t) + M*M*(6*(s*s) + 2*s*t + t*t + mHNL*mHNL*(-2*s + t))))))*Vhad*(Cji*Vij + Cij*Vji))/(M*M*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t))

        if process.TheoryModel.is_mass_mixed: 
            
            # mass mixing + SM NC interference
            Lmunu_Hmunu += -0.5*(Sqrt(Chad*Cprimehad)*(-(FFf2*(4*FFNCf2*(M*M)*(mHNL*mHNL*mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s)) + 4*FFNCf2*(M*M*M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 16*FFNCga*(M*M*M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 4*FFNCf2*(M*M)*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 8*FFNCga*(M*M)*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + FFNCf2*(mHNL*mHNL*mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 8*FFNCf2*(M*M)*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 16*FFNCga*(M*M)*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 4*FFNCf2*(mHNL*mHNL)*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 4*FFNCf2*(s*s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 8*FFNCf2*(M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) + 8*FFNCga*(M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) - FFNCf2*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) + 4*FFNCf2*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) + FFNCf2*h*(4*(M*M)*(mHNL*mHNL*mHNL*mHNL)*(M*M + mHNL*mHNL - s) + (2*(M*M) - 4*M*mHNL + mHNL*mHNL - 2*s)*(2*(M*M) + 4*M*mHNL + mHNL*mHNL - 2*s)*(M*M + mHNL*mHNL - s)*t - (8*(M*M*M*M) + mHNL*mHNL*mHNL*mHNL - 3*(mHNL*mHNL)*s + 4*(s*s) - 3*(M*M)*(3*(mHNL*mHNL) + 4*s))*(t*t)) + 4*FFNCf1*(M*M)*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*(t*t)) + h*(mHNL*mHNL - 2*t)*(mHNL*mHNL*mHNL*mHNL - s*t + M*M*(mHNL*mHNL + t) - mHNL*mHNL*(s + t))) - 8*FFNCga*h*(M*M)*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t))))) + 4*FFf1*(M*M)*(-(FFNCf2*(mHNL*mHNL*mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))) + 4*FFNCga*(M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - FFNCf2*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 2*FFNCga*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 4*FFNCga*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 2*FFNCf2*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) - 2*FFNCga*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) - FFNCf2*h*(mHNL*mHNL - 2*t)*(mHNL*mHNL*mHNL*mHNL - s*t + M*M*(mHNL*mHNL + t) - mHNL*mHNL*(s + t)) + 2*FFNCga*h*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t))) + 2*FFNCf1*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(2*(M*M*M*M) - 4*(M*M)*s + 2*(s*s) + 2*s*t + t*t - mHNL*mHNL*(2*s + t)) + h*(2*(M*M*M*M*M*M) - 2*((mHNL*mHNL - s)*(mHNL*mHNL - s))*s - 2*(M*M*M*M)*(mHNL*mHNL + 3*s) + (mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*s - 2*(s*s))*t - (mHNL*mHNL + s)*(t*t) + M*M*(6*(s*s) + 2*s*t + t*t + mHNL*mHNL*(-2*s + t))))))*Vhad*(Cji*Vij + Cij*Vji))/(M*M*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(MZBOSON*MZBOSON - t)*(-(mzprime*mzprime) + t))
            
            # kinetic mixing + mass mixing interference
            Lmunu_Hmunu += (Cprimehad*(-(FFf2*(4*FFNCf2*(M*M)*(mHNL*mHNL*mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s)) + 4*FFNCf2*(M*M*M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 16*FFNCga*(M*M*M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 4*FFNCf2*(M*M)*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 8*FFNCga*(M*M)*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + FFNCf2*(mHNL*mHNL*mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 8*FFNCf2*(M*M)*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 16*FFNCga*(M*M)*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 4*FFNCf2*(mHNL*mHNL)*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 4*FFNCf2*(s*s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 8*FFNCf2*(M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) + 8*FFNCga*(M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) - FFNCf2*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) + 4*FFNCf2*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) + FFNCf2*h*(4*(M*M)*(mHNL*mHNL*mHNL*mHNL)*(M*M + mHNL*mHNL - s) + (2*(M*M) - 4*M*mHNL + mHNL*mHNL - 2*s)*(2*(M*M) + 4*M*mHNL + mHNL*mHNL - 2*s)*(M*M + mHNL*mHNL - s)*t - (8*(M*M*M*M) + mHNL*mHNL*mHNL*mHNL - 3*(mHNL*mHNL)*s + 4*(s*s) - 3*(M*M)*(3*(mHNL*mHNL) + 4*s))*(t*t)) + 4*FFNCf1*(M*M)*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*(t*t)) + h*(mHNL*mHNL - 2*t)*(mHNL*mHNL*mHNL*mHNL - s*t + M*M*(mHNL*mHNL + t) - mHNL*mHNL*(s + t))) - 8*FFNCga*h*(M*M)*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t))))) + 4*FFf1*(M*M)*(-(FFNCf2*(mHNL*mHNL*mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))) + 4*FFNCga*(M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - FFNCf2*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 2*FFNCga*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 4*FFNCga*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 2*FFNCf2*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) - 2*FFNCga*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) - FFNCf2*h*(mHNL*mHNL - 2*t)*(mHNL*mHNL*mHNL*mHNL - s*t + M*M*(mHNL*mHNL + t) - mHNL*mHNL*(s + t)) + 2*FFNCga*h*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t))) + 2*FFNCf1*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(2*(M*M*M*M) - 4*(M*M)*s + 2*(s*s) + 2*s*t + t*t - mHNL*mHNL*(2*s + t)) + h*(2*(M*M*M*M*M*M) - 2*((mHNL*mHNL - s)*(mHNL*mHNL - s))*s - 2*(M*M*M*M)*(mHNL*mHNL + 3*s) + (mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*s - 2*(s*s))*t - (mHNL*mHNL + s)*(t*t) + M*M*(6*(s*s) + 2*s*t + t*t + mHNL*mHNL*(-2*s + t))))))*Vhad*Vij*Vji)/(2.*(M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*((mzprime*mzprime - t)*(mzprime*mzprime - t)))


            # mass mixing term SQR
            Lmunu_Hmunu += (Cprimehad*Cprimehad*(-4*(FFNCf2*FFNCf2)*h*(M*M*M*M)*(mHNL*mHNL*mHNL*mHNL) - 4*(FFNCf2*FFNCf2)*h*(M*M)*(mHNL*mHNL*mHNL*mHNL*mHNL*mHNL) + 4*(FFNCf2*FFNCf2)*h*(M*M)*(mHNL*mHNL*mHNL*mHNL)*s - 4*(FFNCf2*FFNCf2)*(M*M)*(mHNL*mHNL*mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s)) - 4*(FFNCf2*FFNCf2)*h*(M*M*M*M*M*M)*t + 8*(FFNCf2*FFNCf2)*h*(M*M*M*M)*(mHNL*mHNL)*t + 11*(FFNCf2*FFNCf2)*h*(M*M)*(mHNL*mHNL*mHNL*mHNL)*t - 4*(FFNCgp*FFNCgp)*h*(M*M)*(mHNL*mHNL*mHNL*mHNL)*t - FFNCf2*FFNCf2*h*(mHNL*mHNL*mHNL*mHNL*mHNL*mHNL)*t - 4*(FFNCgp*FFNCgp)*h*(mHNL*mHNL*mHNL*mHNL*mHNL*mHNL)*t + 12*(FFNCf2*FFNCf2)*h*(M*M*M*M)*s*t + 5*(FFNCf2*FFNCf2)*h*(mHNL*mHNL*mHNL*mHNL)*s*t + 4*(FFNCgp*FFNCgp)*h*(mHNL*mHNL*mHNL*mHNL)*s*t - 12*(FFNCf2*FFNCf2)*h*(M*M)*(s*s)*t - 8*(FFNCf2*FFNCf2)*h*(mHNL*mHNL)*(s*s)*t + 4*(FFNCf2*FFNCf2)*h*(s*s*s)*t - 4*(FFNCf2*FFNCf2)*(M*M*M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 4*(FFNCf2*FFNCf2)*(M*M)*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - FFNCf2*FFNCf2*(mHNL*mHNL*mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 4*(FFNCgp*FFNCgp)*(mHNL*mHNL*mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 8*(FFNCf2*FFNCf2)*(M*M)*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 4*(FFNCf2*FFNCf2)*(mHNL*mHNL)*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t - 4*(FFNCf2*FFNCf2)*(s*s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t + 8*(FFNCf2*FFNCf2)*h*(M*M*M*M)*(t*t) - 9*(FFNCf2*FFNCf2)*h*(M*M)*(mHNL*mHNL)*(t*t) - 4*(FFNCgp*FFNCgp)*h*(M*M)*(mHNL*mHNL)*(t*t) + FFNCf2*FFNCf2*h*(mHNL*mHNL*mHNL*mHNL)*(t*t) + 4*(FFNCgp*FFNCgp)*h*(mHNL*mHNL*mHNL*mHNL)*(t*t) - 12*(FFNCf2*FFNCf2)*h*(M*M)*s*(t*t) - 3*(FFNCf2*FFNCf2)*h*(mHNL*mHNL)*s*(t*t) + 4*(FFNCgp*FFNCgp)*h*(mHNL*mHNL)*s*(t*t) + 4*(FFNCf2*FFNCf2)*h*(s*s)*(t*t) + 8*(FFNCf2*FFNCf2)*(M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) + FFNCf2*FFNCf2*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) + 4*(FFNCgp*FFNCgp)*(mHNL*mHNL)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) - 4*(FFNCf2*FFNCf2)*(s*s)*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(t*t) + 8*(FFNCf1*FFNCf1)*(M*M)*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(2*(M*M*M*M) - 4*(M*M)*s + 2*(s*s) + 2*s*t + t*t - mHNL*mHNL*(2*s + t)) + h*(2*(M*M*M*M*M*M) - 2*((mHNL*mHNL - s)*(mHNL*mHNL - s))*s - 2*(M*M*M*M)*(mHNL*mHNL + 3*s) + (mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*s - 2*(s*s))*t - (mHNL*mHNL + s)*(t*t) + M*M*(6*(s*s) + 2*s*t + t*t + mHNL*mHNL*(-2*s + t)))) - 8*FFNCf1*(M*M)*(2*FFNCga*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*t*(-2*(M*M) - mHNL*mHNL + 2*s + t) + FFNCf2*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*t - 2*(t*t)) + FFNCf2*h*(mHNL*mHNL - 2*t)*(mHNL*mHNL*mHNL*mHNL - s*t + M*M*(mHNL*mHNL + t) - mHNL*mHNL*(s + t)) + 2*FFNCga*h*(-2*(M*M*M*M)*t + t*(mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*(s - t) - s*(2*s + t)) + M*M*(-4*(mHNL*mHNL*mHNL*mHNL) + 3*(mHNL*mHNL)*t + t*(4*s + t)))) + 8*(FFNCga*FFNCga)*(M*M)*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(2*(M*M*M*M) + 2*(s*s) + 4*(M*M)*(mHNL*mHNL - s - t) + 2*s*t + t*t - mHNL*mHNL*(2*s + t)) + h*(2*(M*M*M*M*M*M) - 2*((mHNL*mHNL - s)*(mHNL*mHNL - s))*s + (mHNL*mHNL*mHNL*mHNL + mHNL*mHNL*s - 2*(s*s))*t - (mHNL*mHNL + s)*(t*t) - 2*(M*M*M*M)*(3*(mHNL*mHNL + s) + 2*t) + M*M*(-4*(mHNL*mHNL*mHNL*mHNL) + 6*(s*s) + 6*s*t + t*t + mHNL*mHNL*(2*s + 5*t)))) + 16*FFNCga*(M*M)*(FFNCgp*(mHNL*mHNL)*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(-(mHNL*mHNL) + t) + h*(-(mHNL*mHNL*mHNL*mHNL) + s*t - M*M*(mHNL*mHNL + t) + mHNL*mHNL*(s + t))) + FFNCf2*(s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*(2*(M*M) + mHNL*mHNL - 2*s - t)*t + h*(2*(M*M*M*M)*t + t*(-(mHNL*mHNL*mHNL*mHNL) + mHNL*mHNL*(-s + t) + s*(2*s + t)) + M*M*(4*(mHNL*mHNL*mHNL*mHNL) - 3*(mHNL*mHNL)*t - t*(4*s + t))))))*Vij*Vji)/(4.*(M*M)*s*Sqrt((M*M*M*M + (mHNL*mHNL - s)*(mHNL*mHNL - s) - 2*(M*M)*(mHNL*mHNL + s))/(s*s))*((mzprime*mzprime - t)*(mzprime*mzprime - t)))

  

    else:
        print(f"Error! Could not find HNL type '{process.TheoryModel.HNLtype}'.")
        raise ValueError



    # phase space factors
    phase_space = const.kallen_sqrt(1.0, M**2/s, mHNL**2/s)/(32*np.pi**2)
    phase_space *= 2*np.pi # integrated over phi 

    # flux factor in cross section
    flux_factor = 1.0/(s - M**2)/2
    
    E1CM = (s - M**2)/2.0/np.sqrt(s)
    E3CM = (s + mHNL**2 - M**2)/2.0/np.sqrt(s)
    p1CM = E1CM # massless projectile
    p3CM = np.sqrt(E3CM**2 - mHNL**2)

    # jacobian -- from angle to Q2
    physical_jacobian = 1.0/2.0/p1CM/p3CM

    # hadronic spin average
    spin_average = 1/2

    # differential cross section
    diff_xsec = flux_factor * Lmunu_Hmunu * phase_space * physical_jacobian * spin_average

    return diff_xsec*invGeV2_to_attobarn