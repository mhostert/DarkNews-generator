import numpy as np
import pandas as pd
from scipy.stats import expon

from DarkNews import const

#precision
precision = 1e-10

# baselines
baselines = {'miniboone': 541e2, 'miniboone_dirt': 541e2, 'microboone': 470e2, 'microboone_dirt': 470e2, 'sbnd': 110e2, 'sbnd_dirt': 110e2, 'icarus': 600e2, 'icarus_dirt': 600e2}

# radius of MB
radius_MB = 610 #cm
radius_MB_fid = 500 #cm

# geometry of cylinder_MB for dirt
radius_MB_outer = 1370 / 2.
radius_cyl_MB = 1.5 * radius_MB_outer
l_cyl_MB = 47400.
end_point_cyl_MB = -1320.
start_point_cyl_MB = end_point_cyl_MB - l_cyl_MB

# geometry of steal for MB
x_steal_MB = 100.
y_steal_MB = 100.
z_steal_MB = 380.
start_point_steal_MB = start_point_cyl_MB - z_steal_MB

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

#SBND detector
z_sbnd = 5e2
x_sbnd = 4e2
y_sbnd = 4e2
dif_z_sbnd = 20 # 20 cm between TPC and wall of detector


# Icarus
l_icarus = 600e2
x_icarus = 3.6e2*2
y_icarus = 3.9e2
z_icarus = 19.6e2
dif_z_icarus = 1e2



def get_angle(p1, p2):
    x1,y1,z1 = p1
    x2,y2,z2 = p2
    return np.arccos((x1*x2+y1*y2+z1*z2)/(np.sqrt(x1*x1+y1*y1+z1*z1)*np.sqrt(x2*x2+y2*y2+z2*z2)))

def dot3(p1, p2):
    return p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]

def normalize_3D_vec(v):
    return v / np.sqrt(dot3(v,v))

def cross3(p1, p2):
    px = p1[1]*p2[2] - p1[2]*p2[1]
    py = p1[2]*p2[0] - p1[0]*p2[2]
    pz = p1[0]*p2[1] - p1[1]*p2[0]
    return np.array([px,py,pz])

# rotate v by an angle of theta on the plane perpendicular to k using Rodrigues' rotation formula
def rotate_by_theta(v,k,theta):
    # we first normalize k
    k = normalize_3D_vec(k)
    
    # Rodrigues' rotation formula
    return np.cos(theta) * v + np.sin(theta) * cross3(k,v) + dot3(k,v) * (1 - np.cos(theta)) * k

# rotate a 4D-vector v using the same minimum rotation to take vector a into vector b
def rotate_similar_to(v,a,b):
    # normalize vectors a and b
    a = normalize_3D_vec(a)
    b = normalize_3D_vec(b)
    
    # compute normal vector to those and angle
    k = cross3(a,b)
    theta = dot3(a,b)
    
    # use previous function to compute new vector
    return rotate_by_theta(v,k,theta)

def dot4(p1, p2):
    return p1[0]*p2[0] - p1[1]*p2[1] - p1[2]*p2[2] - p1[3]*p2[3]

def get_3direction(p0):
    p = p0.T[1:]
    norm = np.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
    p /= norm
    return p

def rotate_dataframe(df):
    particles = ['P_target','P_recoil','P_decay_N_parent','P_decay_ell_plus','P_decay_ell_minus','P_decay_N_daughter','P_decay_photon','P_projectile']
    
    for particle in particles:
        try:
            df.loc[:,(particle,['1','2','3'])] = rotate_similar_to(df[particle].values.T[1:],df.P_projectile.values.T[1:],df.pos_scatt.values.T[1:] - df['pos_prod'].values.T).T
        except:
            continue
    
    return df

def get_beta(p):
    """get_beta get the velocity in Lab frame

    Parameters
    ----------
    p : np.ndarray
        4 momentum of the particle

    Returns
    -------
    np.ndarray
        array of particle velocities in the LAB frame
    """
    M = np.sqrt(np.abs(dot4(p.T, p.T)))
    return np.sqrt(np.abs(p[:,0]**2 -  M**2))/p[:,0]

def get_decay_length_in_lab(p, l_decay_proper_cm):
    """get_decay_length_in_lab given the proper decay length in cm and
    the 4-momentum in lab frame, get the decay length in the lab frame in cm

    Parameters
    ----------
    p : np.ndarray
        4 momentum of the particle
    l_decay_proper_cm : float
        proper decay length in cm

    Returns
    -------
    np.ndarray
        lab frame decay length in cm
    """
    M = np.sqrt(dot4(p.T, p.T))
    gammabeta = (np.sqrt(p[:,0]**2 -  M**2))/M
    return l_decay_proper_cm*gammabeta

def get_distances(p0, phat, experiment):
    """get_distances_in_muB gets the distance to the entry and exit
    of the MicroBooNE fiducial volume

    NOTE: this function uses a coordinate system with (0,0,0) being the center 
    of the MicroBooNE detector in the xy plane AND the most upstream point of the 
    detector in the z position (the tip of the first spherical cap).

    Parameters
    ----------
    p0 : np.ndarray
        point of HNL production
    phat : np.ndarray
        direction of travel of HNL
    experiment :
        what experiment to find the distances for 'microboone', 'sbnd'

    Returns
    -------
    2 np.ndarrays
        one array for the distance of entry and one for the exit
    """

    # number of events
    n = len(p0.T)

    # positions of the 6 walls of the cryostat in order (2 for X, 2 for Y, 2 for Z)
    if experiment == 'microboone' or experiment == 'microboone_dirt':
        planes = np.array([-x_muB/2,x_muB/2,-y_muB/2,y_muB/2,-z_muB/2,z_muB/2])
    elif experiment == 'sbnd' or experiment == 'sbnd_dirt':
        planes = np.array([-x_sbnd/2,x_sbnd/2,-y_sbnd/2,y_sbnd/2,-z_sbnd/2,z_sbnd/2])
    elif experiment == 'icarus' or experiment == 'icarus_dirt':
        planes = np.array([-x_icarus/2,x_icarus/2,-y_icarus/2,y_icarus/2,-z_icarus/2,z_icarus/2])

    # suitable forms for parameters
    p0_6 = np.array([p0[0], p0[0], p0[1], p0[1], p0[2], p0[2]]).T
    phat_6 = np.array([phat[0], phat[0], phat[1], phat[1], phat[2], phat[2]]).T

    # find solutions and intersections of P0 + phat*t = planes, for parameter t
    solutions = (planes - p0_6) / phat_6
    intersections = [[p0[:,i] + solutions[i,j] * phat[:,i] for j in range(6)] for i in range(n)]

    # create a mask with invalid intersections
    mask_inter = np.array([[
        (planes[0] - precision <= intersections[i][j][0] <= planes[1] + precision)\
      & (planes[2] - precision <= intersections[i][j][1] <= planes[3] + precision)\
      & (planes[4] - precision <= intersections[i][j][2] <= planes[5] + precision)\
      & (solutions[i,j] > -precision) for j in range(6)] for i in range(n)])

    # compute the distances from the previous calculations
    distances = np.zeros((n,2))
    for i in range(n):
        dist_temp = solutions[i][mask_inter[i]]
        if len(dist_temp) == 2:
            distances[i] = [dist_temp.min(),dist_temp.max()]
        elif len(dist_temp) == 1:
            distances[i] = [0, dist_temp[0]]
        else:
            distances[i] = [0,0]

    # return the distances
    return distances


# This programs multiplies the probability of decaying inside the detector by the reco_w. The scattering point is random
def decay_selection(df, l_decay_proper_cm, experiment, weights='w_event_rate'):
    """select_muB_decay_prob this applies a probability of decay inside MicroBooNE to the event weights

        Use scattering positions from generated by DarkNews translating
        the production points from the DarkNews coordinate system
        to the coordinate system in this function (see NOTE below)

        Then we apply the probably of decay inside the fiducial volume
        given by the prob that it decays before exiting minus
        the prob that it decays before entering

    NOTE: the pos_decay using this method is not well defined and should be ignored, since only the probability
    of decaying inside the fiducial volume makes sense.

    Parameters
    ----------
    df : DarkNews pd.DataFrame
        the events 
    l_decay_proper_cm : int
        enforces this l_decay_proper_cm
    experiment : str
        what experiment to do the selectio for: 'miniboone', 'microboone', 'sbnd'
    weights : str, optional
        what pandas dataframe column to reweight with probability, by default 'w_event_rate'

    Returns
    -------
    pd.DataFrame
        the new dataframe with w_event_rate takiung into account the probability of decaying inside the fiducial vol
        and the new weights "w_pre_decay" which do not take that into account.

    """

    df = df.copy(deep=True)
    
    # rotate all momenta by changing the direction of the incoming beam like coming from the center of the target
    #df = rotate_dataframe(df)

    pN = df.P_decay_N_parent.values
    l_decay_lab_cm = get_decay_length_in_lab(pN, l_decay_proper_cm)

    # direction of travel of the HNL.
    phat = get_3direction(pN)

    # production point
    p0 = np.array([df['pos_scatt','1'], df['pos_scatt','2'], df['pos_scatt','3']])


    if experiment == 'miniboone' or experiment == 'miniboone_dirt':
        # compute the distance to the point of exit from the detector using intersection of line with sphere
        #  p0 . phat
        x0_dot_p = dot3(p0, phat)
        x0_square = p0[0]**2 + p0[1]**2 + p0[2]**2
        discriminant = (x0_dot_p * x0_dot_p) - (x0_square - radius_MB_fid**2)
        mask_in_detector = (discriminant > 0)

        # only keep those events that have phat directions that intersect with the detector at least once
        df = df[mask_in_detector].reset_index()

        p0 = p0[:, mask_in_detector]
        phat = phat[:, mask_in_detector]
        discriminant = discriminant[mask_in_detector]
        x0_dot_p = x0_dot_p[mask_in_detector]
        l_decay_lab_cm = l_decay_lab_cm[mask_in_detector]

        dist1 = (- x0_dot_p - np.sqrt(discriminant))
        dist1 = dist1 * (dist1 >= 0)
        dist2 = - x0_dot_p + np.sqrt(discriminant)

        # prob of decay inside the fiducial vol
        probabilities = expon.cdf(dist2, 0, l_decay_lab_cm) - expon.cdf(dist1, 0, l_decay_lab_cm)

        # in this method, no well-defined decay position, so we take the mean of entry and exit points
        df.loc[:,('pos_decay', '0')] = df['pos_scatt', '0'] + (dist2 + dist1)/2 / const.c_LIGHT / get_beta(pN[mask_in_detector])
        df.loc[:,('pos_decay', '1')] = p0[0] + (dist2 + dist1)/2 * phat[0]
        df.loc[:,('pos_decay', '2')] = p0[1] + (dist2 + dist1)/2 * phat[1]
        df.loc[:,('pos_decay', '3')] = p0[2] + (dist2 + dist1)/2 * phat[2]

    else: 
        # dist1 is the distance between the point of production and the entrance of the FIDUCIAL vol
        # dist2 is the distance between the point of production and the exit of the FIDUCIAL vol
        dist1, dist2 = get_distances(p0,phat, experiment).T
        
        #now sample a random distance inside the detector
        frac = np.random.uniform(size=len(df))
        decay_dist = (dist2 - dist1)*frac
        probabilities = expon.cdf(decay_dist, 0, l_decay_lab_cm)

        # in this method, no well-defined decay position, so we take the mean of entry and exit points
        df['pos_decay', '0'] = df['pos_scatt', '0'] + decay_dist / const.c_LIGHT / get_beta(pN)
        df['pos_decay', '1'] = df['pos_scatt', '1'] + decay_dist * phat[0]
        df['pos_decay', '2'] = df['pos_scatt', '2'] + decay_dist * phat[1]
        df['pos_decay', '3'] = df['pos_scatt', '3'] + decay_dist * phat[2]
        
    # new reconstructed weights
    df['w_pre_decay'] = df[weights].values
    df.loc[:,weights] = df[weights].values * probabilities

    return df



def set_params(df, showers='e+e-'):

    df = df.copy(deep=True)
    
    if showers == 'e+e-':
        p21 = np.array([df[( 'P_decay_ell_minus', '1')].values,df[( 'P_decay_ell_minus', '2')].values,df[( 'P_decay_ell_minus', '3')].values])
        p22 = np.array([df[( 'P_decay_ell_plus', '1')].values,df[( 'P_decay_ell_plus', '2')].values,df[( 'P_decay_ell_plus', '3')].values])
        p2 = (p21+p22)/2

        p1 = np.array([0,0,1])
        angle = get_angle(p1,p2)

        df['reco_theta_beam'] = angle * 180 / np.pi

        df['reco_Evis'] = df[( 'P_decay_ell_minus', '0')].values + df[( 'P_decay_ell_plus', '0')].values

        df['reco_w'] = df.w_event_rate
        
        df['reco_Enu'] = const.m_proton * (df['reco_Evis']) / ( const.m_proton - (df['reco_Evis'])*(1.0 - np.cos(angle)))

        return df
    
    elif showers == 'photon':
        p2 = np.array([df[( 'P_decay_photon', '1')].values,df[( 'P_decay_photon', '2')].values,df[( 'P_decay_photon', '3')].values])
        
        p1 = np.array([0,0,1])
        angle = get_angle(p1,p2)

        df['reco_theta_beam'] = angle * 180 / np.pi

        df['reco_Evis'] = df[( 'P_decay_photon', '0')].values

        df['reco_w'] = df.w_event_rate
        
        df['reco_Enu'] = const.m_proton * (df['reco_Evis']) / ( const.m_proton - (df['reco_Evis'])*(1.0 - np.cos(angle)))

        return df


def filter_angle_ee(df, angle_max=5):
    df = df.copy(deep=True)

    p1 = np.array([df[( 'P_decay_ell_minus', '1')].values,df[( 'P_decay_ell_minus', '2')].values,df[( 'P_decay_ell_minus', '3')].values])
    p2 = np.array([df[( 'P_decay_ell_plus', '1')].values,df[( 'P_decay_ell_plus', '2')].values,df[( 'P_decay_ell_plus', '3')].values])

    angle_ee = get_angle(p1,p2)
    df['angle_ee'] = angle_ee * 180 / np.pi

    mask = df.angle_ee <= angle_max

    df.loc[:,'reco_w'] = df.reco_w * mask

    return df


def out_of_active_volume(df,experiment='microboone'):
    df = df.copy(deep=True)
    
    if experiment == 'microboone':
        # filtering out those scatterings inside the active volume
        mask = (-x_muB/2. <= df['pos_scatt','1'].values) & (df['pos_scatt','1'].values <= x_muB/2.) & (-y_muB/2. <= df['pos_scatt','2'].values) & (df['pos_scatt','2'].values<= y_muB/2.) & (-z_muB/2. <= df['pos_scatt','3'].values) & (df['pos_scatt','3'].values <= z_muB/2.)
        not_mask = np.array([bool(1-mask[j]) for j in range(len(mask))])
        df = df[not_mask]
    elif experiment == 'sbnd':
        # filtering out those scatterings inside the active volume
        mask = (-x_sbnd/2. <= df['pos_scatt','1'].values) & (df['pos_scatt','1'].values <= x_sbnd/2.) & (-y_sbnd/2. <= df['pos_scatt','2'].values) & (df['pos_scatt','2'].values<= y_sbnd/2.) & (-z_sbnd/2. <= df['pos_scatt','3'].values) & (df['pos_scatt','3'].values <= z_sbnd/2.)
        not_mask = np.array([bool(1-mask[j]) for j in range(len(mask))])
        df = df[not_mask]
    elif experiment == 'icarus':
        # filtering out those scatterings inside the active volume
        mask = (-x_icarus/2. <= df['pos_scatt','1'].values) & (df['pos_scatt','1'].values <= x_icarus/2.) & (-y_icarus/2. <= df['pos_scatt','2'].values) & (df['pos_scatt','2'].values<= y_icarus/2.) & (-z_icarus/2. <= df['pos_scatt','3'].values) & (df['pos_scatt','3'].values <= z_icarus/2.)
        not_mask = np.array([bool(1-mask[j]) for j in range(len(mask))])
        df = df[not_mask]
    else:
        raise NotImplementedError("This experiment is not implemented")
    
    return df

def set_opening_angle(df):
    df = df.copy(deep=True)

    p1 = np.array([df[( 'P_decay_ell_minus', '1')].values,df[( 'P_decay_ell_minus', '2')].values,df[( 'P_decay_ell_minus', '3')].values])
    p2 = np.array([df[( 'P_decay_ell_plus', '1')].values,df[( 'P_decay_ell_plus', '2')].values,df[( 'P_decay_ell_plus', '3')].values])

    angle_ee = get_angle(p1,p2)
    df['opening_angle'] = angle_ee * 180 / np.pi

    return df




######################################
# Old functions that use MC method to compute the decay positions


# def decay_position(pN, l_decay_proper_cm, random_gen = True):
#     """decay_position get the decay position of particle

#     Parameters
#     ----------
#     pN : np.ndarray
#         4 momentum
#     l_decay_proper_cm : float
#         proper decay length in cm
#     random_gen : bool, optional
#         whether to get decay positions by sampling exponential decay probability, by default True.
#         NOTE: if false, assume all particles decay exactly at the lab frame decay length (unphysical!)

#     Returns
#     -------
#     4 np.ndarrays
#         the time, x, y, and z coordinates of the decay
#     """
#     if random_gen:
#         # Sample from decay propability
#         d_decay = np.random.exponential(scale=get_decay_length_in_lab(pN, l_decay_proper_cm)) # centimeters
#     else:
#         d_decay = get_decay_length_in_lab(pN, l_decay_proper_cm)

#     # time of travel
#     t = d_decay/const.c_LIGHT
#     # position of decay
#     x,y,z = get_3direction(pN)*d_decay

#     return t,x,y,z

# def random_cylinder(num=100):

#     # generate a random position for scattering position
#     u0 = np.random.random(num)
#     phi0 = np.random.random(num) * 2. * np.pi
#     r0 = r_muB * u0**(1./2.)

#     x0 = r0 * np.cos(phi0)
#     y0 = r0 * np.sin(phi0)
#     z0 = np.random.random(num) * l_muB

#     return np.array([x0,y0,z0])

# # Generate random uniformly distributed points in sphere
# def random_sphere(num=100, seed=None, radius=1, theta=[0,np.pi], phi=[0,2*np.pi]):
#     if seed is not None:
#         np.random.seed(seed)
#     u0 = np.random.random(num)
#     cos_theta = [np.cos(theta[0]),np.cos(theta[1])]
#     cos_theta0 = np.random.random(num) * (cos_theta[0]-cos_theta[1]) + cos_theta[1]
#     phi0 = phi[0] + np.random.random(num) * (phi[1]-phi[0])
#     theta0 = np.arccos(cos_theta0)
#     r0 = radius * u0**(1./3.)
    
#     return np.array([r0,theta0,phi0])
    
# # Generate random uniformly distributed points in cut sphere
# def random_cut_sphere(num=100, seed=None, radius=1, up=True):
#     if seed is not None:
#         np.random.seed(seed)
#     output = random_sphere(num=int(3.25*num),radius=radius,theta=[0,theta_lim_muB])
#     mask = output[0] > r_s_muB * np.cos(theta_lim_muB) / np.cos(output[1])
#     output = (output.T[mask]).T
#     while len(output[0]) < num:
#         output_2 = random_sphere(num=10,radius=radius,theta=[0,theta_lim_muB])
#         mask = output_2[0] > r_s_muB * np.cos(theta_lim_muB) / np.cos(output_2[1])
#         output_2 = (output_2.T[mask]).T
#         output = np.concatenate([output,output_2],axis=1)

#     output = (output.T[:num]).T
#     output = np.array([output[0] * np.sin(output[1]) * np.cos(output[2]), output[0] * np.sin(output[1]) * np.sin(output[2]), output[0] * np.cos(output[1])])

#     if up:
#         output[2] += l_muB - radius * np.cos(theta_lim_muB)
#     else:
#         output[2] = radius * np.cos(theta_lim_muB) - output[2]

#     return output

# # Select for MiniBooNE considering the outer spheres
# def select_muB_decay(df, seed=0, coupling_factor=1., l_decay_proper_cm =0, weights='w_event_rate'):
#     df = df.copy(deep=True)

#     # get momenta and decay length for decay_N
#     pN = df.P_decay_N_parent.values
#     if not(l_decay_proper_cm):
#         l_decay_proper_cm = const.get_decay_rate_in_cm(np.sum(df.w_decay_rate_0)) / coupling_factor**2
#     else:
#         l_decay_proper_cm /= coupling_factor**2
    
#     # compute the position of decay
#     x,y,z = decay_position(pN, l_decay_proper_cm)[1:]
#     # length_events = len(x)
#     # num_sphere = round(sphere_cut_muB*length_events)
#     # num_cylinder = length_events - 2*num_sphere
#     # r_cylinder = random_cylinder(num=num_cylinder)
#     # r_sphere_up = random_cut_sphere(num=num_sphere,radius=r_s_muB,up=True)
#     # r_sphere_down = random_cut_sphere(num=num_sphere,radius=r_s_muB,up=False)
#     # x0,y0,z0 = np.concatenate([r_cylinder,r_sphere_up,r_sphere_down],axis=1)
    
#     # compute final position and radius
#     df['decay_position_x'] = df['pos_scatt','1'] + x
#     df['decay_position_y'] = df['pos_scatt','2'] + y
#     df['decay_position_z'] = df['pos_scatt','3'] + z
#     df['in_detector'] = (-x_muB/2. <= df.decay_position_x.values) & (df.decay_position_x.values <= x_muB/2.) & (-y_muB/2. <= df.decay_position_y.values) & (df.decay_position_y.values <= y_muB/2) & (df.decay_position_z.values <= z_muB/2) & (df.decay_position_z.values >= -z_muB/2)
#     df['w_pre_decay'] = df[weights].values
#     df.loc[:,weights] = df[weights].values * df.in_detector.values

#     df['scatt_x'] = df['pos_scatt','1']
#     df['scatt_y'] = df['pos_scatt','2']
#     df['scatt_z'] = df['pos_scatt','3']

#     return df


# # Select for MicroBooNE considering the outer spheres
# def select_muB_decay_dirt_MC(df, seed=0, coupling_factor=1., l_decay_proper_cm =0, weights='w_event_rate'):
#     df = df.copy(deep=True)

#     # get momenta and decay length for decay_N
#     pN = df.P_decay_N_parent.values
#     if not(l_decay_proper_cm):
#         l_decay_proper_cm = const.get_decay_rate_in_cm(np.sum(df.w_decay_rate_0)) / coupling_factor**2
#     else:
#         l_decay_proper_cm /= coupling_factor**2
    
#     # compute the position of decay
#     x,y,z = decay_position(pN, l_decay_proper_cm,random_gen = False)[1:]
#     decay_rate_lab = np.sqrt(x*x + y*y + z*z)
#     x_norm, y_norm, z_norm = x/decay_rate_lab, y/decay_rate_lab, z/decay_rate_lab
#     phat = np.array([x_norm,y_norm,z_norm])
#     length_events = len(x)
    
#     # generate a random position for scattering position
#     if seed:
#         np.random.seed(seed)
#     x0 = np.random.random(length_events)*(x_muB_dirt_max - x_muB_dirt_min)  + x_muB_dirt_min
#     y0 = np.random.random(length_events)*(y_muB_dirt_max - y_muB_dirt_min)  + y_muB_dirt_min
#     z0 = np.random.random(length_events)*(z_muB_dirt_max - z_muB_dirt_min)  + z_muB_dirt_min

#     # compute final position and radius
#     xf, yf, zf = x0 + x, y0 + y, z0 + z
#     df['decay_position_x'] = xf
#     df['decay_position_y'] = yf
#     df['decay_position_z'] = zf
#     df['in_detector'] = (-x_muB/2. <= df.decay_position_x.values) & (df.decay_position_x.values <= x_muB/2.) & (-y_muB/2. <= df.decay_position_y.values) & (df.decay_position_y.values <= y_muB/2) & (df.decay_position_z.values <= z_muB + dif_z/2) & (df.decay_position_z.values >= dif_z/2)
#     df['w_pre_decay'] = df[weights].values
#     df.loc[:,weights] = df[weights].values * df.in_detector.values

#     df['scatt_x'] = x0
#     df['scatt_y'] = y0
#     df['scatt_z'] = z0

#     return df


# def select_MB_decay(df, seed=0, coupling_factor=1., l_decay_proper_cm =0, weights='w_event_rate'):
#     df = df.copy(deep=True)

#     # get momenta and decay length for decay_N
#     pN = df.P_decay_N_parent.values
#     if not(l_decay_proper_cm):
#         l_decay_proper_cm = const.get_decay_rate_in_cm(np.sum(df.w_decay_rate_0)) / coupling_factor**2
#     else:
#         l_decay_proper_cm /= coupling_factor**2

#     # compute the position of decay
#     t,x,y,z = decay_position(pN, l_decay_proper_cm)
#     length_events = len(x)

#     # generate a random position for scattering position
#     if seed:
#         np.random.seed(seed)
#     u0 = np.random.random(length_events)
#     cos_theta0 = np.random.random(length_events) * 2. - 1.
#     phi0 = np.random.random(length_events) * 2. * np.pi
#     theta0 = np.arccos(cos_theta0)
#     r0 = radius_MB * u0**(1./3.)
#     x0, y0, z0  = r0 * np.sin(theta0) * np.cos(phi0), r0 * np.sin(theta0) * np.sin(phi0), r0 * np.cos(theta0)

#     # compute final position and radius
#     xf, yf, zf = x0 + x, y0 + y, z0 + z
#     decay_position_ = np.sqrt(xf*xf + yf*yf + zf*zf)
#     df['decay_time'] = t + df['pos_scatt','0']
#     df['decay_position'] = decay_position_
#     df['in_detector'] = df.decay_position.values <= radius_MB_fid
#     df['w_pre_decay'] = df[weights].values
#     df.loc[:,weights] = df[weights] * df.in_detector.values

#     return df



# # This programs multiplies the probability of decaying inside the detector by the reco_w. The scattering point is random
# def select_MB_decay_dirt(df, seed=0, coupling_factor=1., l_decay_proper_cm =0, weights='w_event_rate'):
#     df = df.copy(deep=True)

#     # get momenta and decay length for decay_N
#     pN = df.P_decay_N_parent.values
#     if not(l_decay_proper_cm):
#         l_decay_proper_cm = const.get_decay_rate_in_cm(np.sum(df.w_decay_rate_0)) / coupling_factor**2
#     else:
#         l_decay_proper_cm /= coupling_factor**2

#     # compute the position of decay
#     t,x,y,z = decay_position(pN, l_decay_proper_cm,random_gen = False)
#     decay_rate_lab = np.sqrt(x*x + y*y + z*z)
#     x_norm, y_norm, z_norm = x/decay_rate_lab, y/decay_rate_lab, z/decay_rate_lab
#     length_events = len(x)

#     # generate a random position for scattering position
#     if seed:
#         np.random.seed(seed)
#     u0 = np.random.random(length_events)
#     phi0 = np.random.random(length_events) * 2. * np.pi
#     r0 = radius_cyl_MB * u0**(1./2.)
#     x0, y0  = r0 * np.cos(phi0), r0 * np.sin(phi0)
#     z0 = np.random.random(length_events) * l_cyl_MB + start_point_cyl_MB

#     # compute the distance to the point of exit from the detector
#     x0_dot_p = x_norm * x0 + y_norm * y0 + z_norm * z0
#     x0_square = r0*r0 + z0*z0
#     discriminant = (x0_dot_p * x0_dot_p) - (x0_square - radius_MB_fid**2)
#     mask_in_detector = ((discriminant > 0) & (z_norm > 0))
    
#     df = df[mask_in_detector]
#     discriminant = discriminant[mask_in_detector]
#     x0_dot_p = x0_dot_p[mask_in_detector]
#     decay_rate_lab = decay_rate_lab[mask_in_detector]

#     distance_traveled_1 = - x0_dot_p - np.sqrt(discriminant)
#     distance_traveled_2 = - x0_dot_p + np.sqrt(discriminant)
#     probabilities = expon.cdf(distance_traveled_2,0,decay_rate_lab) - expon.cdf(distance_traveled_1,0,decay_rate_lab)

#     # new reconstructed weights
#     df['w_event_rate'] = df[weights].values
#     df.loc[:,weights] = df[weights].values * probabilities
#     df.loc[:,('pos_scatt', '0')] = z0[mask_in_detector]/const.c_LIGHT
#     df.loc[:,('pos_scatt', '1')] = x0[mask_in_detector]
#     df.loc[:,('pos_scatt', '2')] = y0[mask_in_detector]
#     df.loc[:,('pos_scatt', '3')] = z0[mask_in_detector]
#     df.loc[:,('pos_decay', '0')] = df.loc[:,('pos_scatt', '0')] + (distance_traveled_1+distance_traveled_2)/2/const.c_LIGHT
#     df.loc[:,('pos_decay', '1')] = x0[mask_in_detector] + x[mask_in_detector]
#     df.loc[:,('pos_decay', '2')] = y0[mask_in_detector] + y[mask_in_detector]
#     df.loc[:,('pos_decay', '3')] = z0[mask_in_detector] + z[mask_in_detector]

#     return df




# # This programs multiplies the probability of decaying inside the detector by the reco_w. The scattering point is random
# def select_MB_decay_dirt_no_filt(df, seed=0, coupling_factor=1., l_decay_proper_cm =0, weights='w_event_rate'):
#     df = df.copy(deep=True)

#     # get momenta and decay length for decay_N
#     pN = df.P_decay_N_parent.values
#     if not(l_decay_proper_cm):
#         l_decay_proper_cm = const.get_decay_rate_in_cm(np.sum(df.w_decay_rate_0)) / coupling_factor**2
#     else:
#         l_decay_proper_cm /= coupling_factor**2

#     # compute the position of decay
#     x,y,z = decay_position(pN, l_decay_proper_cm,random_gen = False)[1:]
#     decay_rate_lab = np.sqrt(x*x + y*y + z*z)
#     x_norm, y_norm, z_norm = x/decay_rate_lab, y/decay_rate_lab, z/decay_rate_lab
#     length_events = len(x)

#     # generate a random position for scattering position
#     if seed:
#         np.random.seed(seed)
#     u0 = np.random.random(length_events)
#     phi0 = np.random.random(length_events) * 2. * np.pi
#     r0 = radius_cyl_MB * u0**(1./2.)
#     x0, y0  = r0 * np.cos(phi0), r0 * np.sin(phi0)
#     z0 = np.random.random(length_events) * l_cyl_MB + start_point_cyl_MB

#     # compute the distance to the point of exit from the detector
#     x0_dot_p = x_norm * x0 + y_norm * y0 + z_norm * z0
#     x0_square = r0*r0 + z0*z0
#     discriminant = (x0_dot_p * x0_dot_p) - (x0_square - radius_MB**2)
#     mask_in_detector = ((discriminant > 0) & (z_norm > 0))
#     discriminant = mask_in_detector * discriminant
    
#     distance_traveled_1 = (- x0_dot_p - np.sqrt(discriminant)) * mask_in_detector
#     distance_traveled_2 = (- x0_dot_p + np.sqrt(discriminant)) * mask_in_detector
#     probabilities = expon.cdf(distance_traveled_2,0,decay_rate_lab) - expon.cdf(distance_traveled_1,0,decay_rate_lab)

#     # new reconstructed weights
#     df['w_pre_decay'] = df[weights].values
#     df.loc[:,weights] = df[weights].values * probabilities
#     df.loc[:,('pos_scatt', '0')] = z0/const.c_LIGHT
#     df.loc[:,('pos_scatt', '1')] = x0
#     df.loc[:,('pos_scatt', '2')] = y0
#     df.loc[:,('pos_scatt', '3')] = z0
#     # take the average of the entry and exit points
#     df.loc[:,('pos_decay', '0')] = df.loc[:,('pos_scatt', '0')] + (distance_traveled_1+distance_traveled_2)/2/const.c_LIGHT
#     df.loc[:,('pos_decay', '1')] = x0 + x
#     df.loc[:,('pos_decay', '2')] = y0 + y
#     df.loc[:,('pos_decay', '3')] = z0 + z

#     return df


# # This programs multiplies the probability of decaying inside the detector by the reco_w. The scattering point is random
# def select_MB_decay_steel(df, seed=0, coupling_factor=1., l_decay_proper_cm =0, weights='w_event_rate'):
#     df = df.copy(deep=True)

#     # get momenta and decay length for decay_N
#     pN = df.P_decay_N_parent.values
#     if not(l_decay_proper_cm):
#         l_decay_proper_cm = const.get_decay_rate_in_cm(np.sum(df.w_decay_rate_0)) / coupling_factor**2
#     else:
#         l_decay_proper_cm /= coupling_factor**2

#     # compute the position of decay
#     x,y,z = decay_position(pN, l_decay_proper_cm,random_gen = False)[1:]
#     decay_rate_lab = np.sqrt(x*x + y*y + z*z)
#     x_norm, y_norm, z_norm = x/decay_rate_lab, y/decay_rate_lab, z/decay_rate_lab
#     length_events = len(x)

#     # generate a random position for scattering position
#     if seed:
#         np.random.seed(seed)
#     x0, y0, z0  = np.random.random(length_events) * x_steal_MB - x_steal_MB/2., np.random.random(length_events) * y_steal_MB - y_steal_MB/2., np.random.random(length_events) * z_steal_MB + start_point_steal_MB
    
#     # compute the distance to the point of exit from the detector
#     x0_dot_p = x_norm * x0 + y_norm * y0 + z_norm * z0
#     x0_square = x0*x0 + y0*y0 + z0*z0
#     discriminant = (x0_dot_p * x0_dot_p) - (x0_square - radius_MB**2)
#     mask_in_detector = ((discriminant > 0) & (z_norm > 0))
    
#     df = df[mask_in_detector]
#     discriminant = discriminant[mask_in_detector]
#     x0_dot_p = x0_dot_p[mask_in_detector]
#     decay_rate_lab = decay_rate_lab[mask_in_detector]

#     distance_traveled_1 = - x0_dot_p - np.sqrt(discriminant)
#     distance_traveled_2 = - x0_dot_p + np.sqrt(discriminant)
#     probabilities = expon.cdf(distance_traveled_2,0,decay_rate_lab) - expon.cdf(distance_traveled_1,0,decay_rate_lab)

#     # new reconstructed weights
#     df['w_pre_decay'] = df[weights].values
#     df.loc[:,weights] = df[weights].values * probabilities
#     df.loc[:,('pos_scatt', '1')] = x0[mask_in_detector]
#     df.loc[:,('pos_scatt', '2')] = y0[mask_in_detector]
#     df.loc[:,('pos_scatt', '3')] = z0[mask_in_detector]
#     df.loc[:,('pos_decay', '1')] = x0[mask_in_detector] + x[mask_in_detector]
#     df.loc[:,('pos_decay', '2')] = y0[mask_in_detector] + y[mask_in_detector]
#     df.loc[:,('pos_decay', '3')] = z0[mask_in_detector] + z[mask_in_detector]

#     return df
