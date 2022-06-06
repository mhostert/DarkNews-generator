import numpy as np
import pandas as pd
from scipy.stats import expon

from DarkNews import const


# radius of MB
radius_MB = 610 #cm

# geometry of muBoone
	#cryostat vessel
r_muB = 191.61
l_muB  = 1086.49
	#detector
z_muB = 1040.
x_muB = 256.
y_muB = 232.
dif_z = l_muB - z_muB
    #outer spheres
r_s_muB = 305.250694958
theta_lim_muB = 38.8816337686 * np.pi / 180.0
    #how much volume for each - rates
sphere_cut_muB = 0.030441980173709752
cylinder_cut_muB = 1. - 2*sphere_cut_muB


def random_cylinder(num=100):

    # generate a random position for scattering position
    u0 = np.random.random(num)
    phi0 = np.random.random(num) * 2. * np.pi
    r0 = r_muB * u0**(1./2.)

    x0 = r0 * np.cos(phi0)
    y0 = r0 * np.sin(phi0)
    z0 = np.random.random(num) * l_muB

    return np.array([x0,y0,z0])

# Generate random uniformly distributed points in sphere
def random_sphere(num=100,seed=0,radius=1,theta=[0,np.pi],phi=[0,2*np.pi]):
    if seed:
        np.random.seed(seed)
    u0 = np.random.random(num)
    cos_theta = [np.cos(theta[0]),np.cos(theta[1])]
    cos_theta0 = np.random.random(num) * (cos_theta[0]-cos_theta[1]) + cos_theta[1]
    phi0 = phi[0] + np.random.random(num) * (phi[1]-phi[0])
    theta0 = np.arccos(cos_theta0)
    r0 = radius * u0**(1./3.)
    
    return np.array([r0,theta0,phi0])
    
# Generate random uniformly distributed points in cut sphere
def random_cut_sphere(num=100,seed=0,radius=1,up=True):
    
    output = random_sphere(num=int(3.25*num),radius=radius,theta=[0,theta_lim_muB])
    mask = output[0] > r_s_muB * np.cos(theta_lim_muB) / np.cos(output[1])
    output = (output.T[mask]).T
    while len(output[0]) < num:
        output_2 = random_sphere(num=10,radius=radius,theta=[0,theta_lim_muB])
        mask = output_2[0] > r_s_muB * np.cos(theta_lim_muB) / np.cos(output_2[1])
        output_2 = (output_2.T[mask]).T
        output = np.concatenate([output,output_2],axis=1)
    
    output = (output.T[:num]).T
    output = np.array([output[0] * np.sin(output[1]) * np.cos(output[2]), output[0] * np.sin(output[1]) * np.sin(output[2]), output[0] * np.cos(output[1])])

    if up:
        output[2] += l_muB - radius * np.cos(theta_lim_muB)
    else:
        output[2] = radius * np.cos(theta_lim_muB) - output[2]
    
    return output

    

def get_angle(p1,p2):
    x1,y1,z1 = p1
    x2,y2,z2 = p2
    return np.arccos((x1*x2+y1*y2+z1*z2)/(np.sqrt(x1*x1+y1*y1+z1*z1)*np.sqrt(x2*x2+y2*y2+z2*z2)))

def dot4(p1,p2):
    return p1[0]*p2[0] - p1[1]*p2[1] - p1[2]*p2[2] - p1[3]*p2[3]

def get_3direction(p0):
    p = p0.T[1:]
    norm = np.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
    p /= norm
    return p

def decay_position(pN, l_decay_proper_cm,random_gen = True):

    # decay the particle
    M4 = np.sqrt(dot4(pN.T,pN.T))
    gammabeta_inv = M4/(np.sqrt(pN[:,0]**2 -  M4*M4 ))
    ######################
    # Sample from decay propability
    if random_gen:
        d_decay = np.random.exponential(scale=l_decay_proper_cm/gammabeta_inv) # centimeters
    else:
        d_decay = l_decay_proper_cm/gammabeta_inv

    # direction of N
    t = d_decay/const.c_LIGHT/gammabeta_inv
    x,y,z = get_3direction(pN)*d_decay
    
    return t,x,y,z


def select_MB_decay(df,seed=0,coupling_factor=1.,l_decay_proper_cm =0):
    df = df.copy(deep=True)

    # get momenta and decay length for decay_N
    pN = df.P_decay_N_parent.values
    if not(l_decay_proper_cm):
        l_decay_proper_cm = const.get_decay_rate_in_cm(np.sum(df.w_decay_rate_0)) / coupling_factor**2
    else:
        l_decay_proper_cm /= coupling_factor**2

    # compute the position of decay
    x,y,z = decay_position(pN, l_decay_proper_cm)[1:]
    length_events = len(x)

    # generate a random position for scattering position
    if seed:
        np.random.seed(seed)
    u0 = np.random.random(length_events)
    cos_theta0 = np.random.random(length_events) * 2. - 1.
    phi0 = np.random.random(length_events) * 2. * np.pi
    theta0 = np.arccos(cos_theta0)
    r0 = radius_MB * u0**(1./3.)
    x0, y0, z0  = r0 * np.sin(theta0) * np.cos(phi0), r0 * np.sin(theta0) * np.sin(phi0), r0 * np.cos(theta0)

    # compute final position and radius
    xf, yf, zf = x0 + x, y0 + y, z0 + z
    decay_position_ = np.sqrt(xf*xf + yf*yf + zf*zf)
    df['decay_position'] = decay_position_
    df['in_detector'] = df.decay_position.values <= radius_MB
    df['reco_w_pre_decay'] = df.reco_w.values
    df.loc[:,'reco_w'] = df.reco_w.values * df.in_detector.values

    return df


# This programs multiplies the probability of decaying inside the detector by the reco_w. The scattering point is random
def select_MB_decay_expo_prob(df,seed=0,coupling_factor=1.,l_decay_proper_cm =0):
    df = df.copy(deep=True)

    # get momenta and decay length for decay_N
    pN = df.P_decay_N_parent.values
    if not(l_decay_proper_cm):
        l_decay_proper_cm = const.get_decay_rate_in_cm(np.sum(df.w_decay_rate_0)) / coupling_factor**2
    else:
        l_decay_proper_cm /= coupling_factor**2

    # compute the position of decay
    x,y,z = decay_position(pN, l_decay_proper_cm,random_gen = False)[1:]
    decay_rate_lab = np.sqrt(x*x + y*y + z*z)
    x_norm, y_norm, z_norm = x/decay_rate_lab, y/decay_rate_lab, z/decay_rate_lab
    length_events = len(x)

    # generate a random position for scattering position
    if seed:
        np.random.seed(seed)
    u0 = np.random.random(length_events)
    cos_theta0 = np.random.random(length_events) * 2. - 1.
    phi0 = np.random.random(length_events) * 2. * np.pi
    theta0 = np.arccos(cos_theta0)
    r0 = radius_MB * u0**(1./3.)
    x0, y0, z0  = r0 * np.sin(theta0) * np.cos(phi0), r0 * np.sin(theta0) * np.sin(phi0), r0 * np.cos(theta0)

    # compute the distance to the point of exit from the detector
    x0_dot_p = x_norm * x0 + y_norm * y0 + z_norm * z0
    distance_traveled = np.sqrt(x0_dot_p*x0_dot_p - r0*r0 + radius_MB*radius_MB) - x0_dot_p
    probabilities = expon.cdf(distance_traveled,0,decay_rate_lab)

    # new reconstructed weights
    df['reco_w_pre_decay'] = df.reco_w.values
    df.loc[:,'reco_w'] = df.reco_w.values * probabilities

    return df


# Select for MiniBooNE considering the outer spheres
def select_muB_decay(df,seed=0,coupling_factor=1.,l_decay_proper_cm =0):
    df = df.copy(deep=True)

    # get momenta and decay length for decay_N
    pN = df.P_decay_N_parent.values
    if not(l_decay_proper_cm):
        l_decay_proper_cm = const.get_decay_rate_in_cm(np.sum(df.w_decay_rate_0)) / coupling_factor**2
    else:
        l_decay_proper_cm /= coupling_factor**2
    
    # compute the position of decay
    x,y,z = decay_position(pN, l_decay_proper_cm)[1:]
    length_events = len(x)
    num_sphere = round(sphere_cut_muB*length_events)
    num_cylinder = length_events - 2*num_sphere
    
    r_cylinder = random_cylinder(num=num_cylinder)
    r_sphere_up = random_cut_sphere(num=num_sphere,radius=r_s_muB,up=True)
    r_sphere_down = random_cut_sphere(num=num_sphere,radius=r_s_muB,up=False)
    x0,y0,z0 = np.concatenate([r_cylinder,r_sphere_up,r_sphere_down],axis=1)
    
    # compute final position and radius
    xf, yf, zf = x0 + x, y0 + y, z0 + z
    df['decay_position_x'] = xf
    df['decay_position_y'] = yf
    df['decay_position_z'] = zf
    df['in_detector'] = (-x_muB/2. <= df.decay_position_x.values) & (df.decay_position_x.values <= x_muB/2.) & (-y_muB/2. <= df.decay_position_y.values) & (df.decay_position_y.values <= y_muB/2) & (df.decay_position_z.values <= z_muB + dif_z/2) & (df.decay_position_z.values >= dif_z/2)
    df['reco_w_pre_decay'] = df.reco_w.values
    df.loc[:,'reco_w'] = df.reco_w.values * df.in_detector.values

    df['scatt_x'] = x0
    df['scatt_y'] = y0
    df['scatt_z'] = z0

    return df


def set_params(df):

    df = df.copy(deep=True)

    df['reco_Enu'] = df[(  'P_projectile', '0')]
    p1 = [df[( 'P_projectile', '1')],df[( 'P_projectile', '2')],df[( 'P_projectile', '3')]]

    p21 = np.array([df[( 'P_decay_ell_minus', '1')].values,df[( 'P_decay_ell_minus', '2')].values,df[( 'P_decay_ell_minus', '3')].values])
    p22 = np.array([df[( 'P_decay_ell_plus', '1')].values,df[( 'P_decay_ell_plus', '2')].values,df[( 'P_decay_ell_plus', '3')].values])
    p2 = (p21+p22)/2

    angle = get_angle(p1,p2)

    df['reco_theta_beam'] = angle * 180 / np.pi

    df['reco_Evis'] = df[( 'P_decay_ell_minus', '0')].values + df[( 'P_decay_ell_plus', '0')].values

    df=df.rename(columns = {'w_event_rate':'reco_w'})

    #df['reco_w'] = df.w_event_rate

    return df


def filter_angle_ee(df,angle_max=5):
    df = df.copy(deep=True)

    p1 = np.array([df[( 'P_decay_ell_minus', '1')].values,df[( 'P_decay_ell_minus', '2')].values,df[( 'P_decay_ell_minus', '3')].values])
    p2 = np.array([df[( 'P_decay_ell_plus', '1')].values,df[( 'P_decay_ell_plus', '2')].values,df[( 'P_decay_ell_plus', '3')].values])

    angle_ee = get_angle(p1,p2)
    df['angle_ee'] = angle_ee * 180 / np.pi

    mask = df.angle_ee <= angle_max

    df.loc[:,'reco_w'] = df.reco_w * mask

    return df


def set_opening_angle(df):
    df = df.copy(deep=True)

    p1 = np.array([df[( 'P_decay_ell_minus', '1')].values,df[( 'P_decay_ell_minus', '2')].values,df[( 'P_decay_ell_minus', '3')].values])
    p2 = np.array([df[( 'P_decay_ell_plus', '1')].values,df[( 'P_decay_ell_plus', '2')].values,df[( 'P_decay_ell_plus', '3')].values])

    angle_ee = get_angle(p1,p2)
    df['opening_angle'] = angle_ee * 180 / np.pi

    return df
