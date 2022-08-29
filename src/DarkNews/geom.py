import numpy as np
from . import const
from . import Cfourvec as Cfv

class Chisel:
    def __init__(self, nsamples, box=np.array(3 * [[-0.5, 0.5]]), name="my_mold"):
        """
        Sculptor of detector geometries. We start with a cuboid full of uniformly generated points and
        start sculpting the desired detector geometry by iteratively cutting events in given 3D objects.
        One can then find the union or intersection of these objects. The available objects are:
                * rectangle
                * sphere
                * cylinder
                * spherical_cap
        More complex and final pre-defined objects are also available:
                * the microboone cryostat as a junction of a tube with two spherical caps.
        A desired set of points randomly distributed across a final 3D object can then found by iteratively
        creating more sculptors with the same Chisel, until the desired number of points is reached.
        Args:
                nsamples (int): number of locations to generate
                box (numpy.ndarray, optional): size of the initial cuboid. Defaults to a cube: np.array(3*[[-0.5,0.5]]).
                name (str, optional): _description_. Defaults to 'my_mold'.
        Raises:
                ValueError: raised when the chisel suspects that the detector geometry is not specified correctly.
        """

        self.name = name
        self.nsamples = nsamples

        # initialize the ivory box
        if (box[:, 1] - box[:, 0] < 0).any():
            raise ValueError(f"Box axis reversed, x_0 > x_1 for dimensions: {[i for i, x in enumerate(box[:,1] - box[:,0]) if x < 0 ]}.")
        self.x = (box[0, 1] - box[0, 0]) * np.random.rand(nsamples) + box[0, 0]
        self.y = (box[1, 1] - box[1, 0]) * np.random.rand(nsamples) + box[1, 0]
        self.z = (box[2, 1] - box[2, 0]) * np.random.rand(nsamples) + box[2, 0]

        self.events = np.array([self.x, self.y, self.z])

    def union(self, mold1, mold2):
        return mold1 | mold2

    def intersection(self, mold1, mold2):
        return mold1 & mold2

    def translate(self, xvec, translation):
        return xvec + translation

    # simple geometrical objects
    def rectangle(self, dx, dy, dz, origin=np.zeros((3, 1))):
        x, y, z = self.translate(origin, self.events)
        return ( (-dx / 2 < x) & (x < dx / 2)) & ((-dy / 2 < y) & (y < dy / 2)) & ((-dz / 2 < z) & (z < dz / 2))

    def sphere(self, radius, origin=np.zeros((3, 1))):
        x, y, z = self.translate(origin, self.events)
        return x ** 2 + y ** 2 + z ** 2 < radius ** 2

    def cylinder(self, radius, height, origin=np.zeros((3, 1))):
        x, y, z = self.translate(origin, self.events)
        return (x ** 2 + y ** 2 < radius ** 2) & (-height / 2 < z) & (z < height / 2)

    def spherical_cap(self, radius, zenith_max, origin=np.zeros((3, 1))):
        x, y, z = self.translate(origin, self.events)
        rpolar = np.sqrt(x ** 2 + y ** 2)
        r = np.sqrt(rpolar ** 2 + z ** 2)
        zenith = np.arctan2(z, rpolar)
        h = radius * (1 - np.cos(zenith_max))
        return (r < radius) | (zenith < zenith_max) | (z < -radius + h)


    def microboone_cryostat(self):
        x = self.x
        y = self.y
        z = self.z

        # accept in tube
        r_polar = np.sqrt(x ** 2 + y ** 2)
        mask_tube = (-z_t / 2 < z) & (z < z_t / 2) & (r_polar < r_t)

        # coordinates in sphere 1
        z1 = z - (-z_t / 2 - cap_gap + r_c * ctheta_c)
        r1 = np.sqrt(r_polar ** 2 + z1 ** 2)
        inc1 = np.arctan2(z1, r_polar)  # inclination angle

        # accept in front cap
        mask_cap1 = (r1 < r_c) & (inc1 < theta_c) & (z1 < -r_c + h)

        # coordinates in sphere 2
        z2 = z + (-z_t / 2 - cap_gap - h + r_c)
        r2 = np.sqrt(r_polar ** 2 + z2 ** 2)
        inc2 = np.arctan2(-z2, r_polar)  # inclination angle

        # accept in back cap
        mask_cap2 = (r2 < r_c) & (inc2 < theta_c) & (z2 > r_c - h)
        
        fraction_of_height_wLAr = 0.85
        mask_full = (mask_tube + mask_cap1 + mask_cap2) & ( y + r_t < 2*r_t * fraction_of_height_wLAr)

        return mask_full



# @dataclass
# class MicroBooNE:
# geometry of tpc
z_muB = 1036.8
x_muB = 256.35
y_muB = 116.5*2

# geometry of cylinder_muB for dirt
l_muB  = 1086.49
#z_muB = 1040.

# target to front of detector
baseline_muB = 470e2
l_dirt_muB = 400e2
x_muB_dirt_min = -1100.
x_muB_dirt_max = 1100.
y_muB_dirt_min = -1600.
y_muB_dirt_max = 600.
z_muB_dirt_max = -215 - l_muB / 2
z_muB_dirt_min = z_muB_dirt_max - l_dirt_muB

# Tube parameters
r_t = 191.61
z_t = 1086.49

# cap parameters
r_c = 305.250694958
theta_c = 38.8816337686 * const.deg_to_rad
ctheta_c = np.cos(theta_c)
zend_c = 305.624305042
h = r_c * (1.0 - ctheta_c)
cap_gap = 0.37361008

# Parameters from Mark -- neutrino time spill, overlapping w Genie BNB
MicroBooNEGlobalTimeOffset = 3125.0
MicroBooNERandomTimeOffset = 1600.0

# @dataclass
# class MiniBooNE:
# geometry of cylinder_MB for dirt
radius_MB_outer = 1370 / 2.
radius_cyl_MB = 1.5 * radius_MB_outer
l_cyl_MB = 47400.

end_point_cyl_MB = -1320.
start_point_cyl_MB = end_point_cyl_MB - l_cyl_MB

# SBND - FIX ME, are SBND and MicroBooNE centered along the same Z axis?
# No, but close enough?
booster_decay_tunnel = 50e2
l_baseline_sbnd = 110e2
gap_sbnd_wall_TPC = 1e2
dx_sbnd = 4e2
dy_sbnd = 4e2
dz_sbnd = 5e2

x_sbnd_dirt_min = x_muB_dirt_min
x_sbnd_dirt_max = x_muB_dirt_max
y_sbnd_dirt_min = y_muB_dirt_min
y_sbnd_dirt_max = y_muB_dirt_max
z_sbnd_dirt_max = - dz_sbnd/2 - gap_sbnd_wall_TPC
z_sbnd_dirt_min = -l_baseline_sbnd + booster_decay_tunnel 

# distribute events in df accross the pre-defined MicroBooNE (cryostat) volume
def microboone_geometry(df):

    nsamples = len(df.index)

    detector_box = np.array([[-200, 200], [-200, 200], [-800, 800]])

    tries = 0
    npoints = 0
    events = np.array(3 * [[]])
    while npoints < nsamples:

        new_detector = Chisel(nsamples=nsamples, box=detector_box)
        new_events = new_detector.events[:, new_detector.microboone_active_tpc_geometry_benchmark()]
        events = np.concatenate((events, new_events), axis=1)

        npoints += np.shape(new_events)[1]
        tries += nsamples
        if tries > 1e3 * nsamples:
            raise ValueError("Geometry sampled too inefficiently. Wrong setup?")

    # Parameters from Mark -- neutrino time spill, overlapping w Genie BNB
    time = MicroBooNEGlobalTimeOffset + (MicroBooNERandomTimeOffset) * np.random.rand(nsamples)

    # guarantee that array has number of samples asked (nsamples)
    df["pos_scatt", "0"] = time
    df["pos_scatt", "1"] = events[0, :nsamples]
    df["pos_scatt", "2"] = events[1, :nsamples]
    df["pos_scatt", "3"] = events[2, :nsamples]


# distribute events in df accross the pre-defined spherical MiniBooNE volume
def sbnd_geometry(df):

    nsamples = len(df)

    detector_box = np.array([[-300, 300], [-300, 300], [-300, 300]])

    tries = 0
    npoints = 0
    events = np.array(3 * [[]])
    while npoints < nsamples:

        new_detector = Chisel(nsamples=nsamples, box=detector_box)
        new_events = new_detector.events[:, new_detector.rectangle(dx=dx_sbnd, dy=dy_sbnd, dz=dz_sbnd)]
        events = np.concatenate((events, new_events), axis=1)

        npoints += np.shape(new_events)[1]
        tries += nsamples
        if tries > 1e3 * nsamples:
            raise ValueError(f"Geometry sampled too inefficiently, tries = {tries} and npoints = {npoints}. Wrong setup?")

    # guarantee that array has number of samples asked (nsamples)
    df["pos_scatt", "0"] = (events[2, :nsamples] + l_baseline_sbnd + gap_sbnd_wall_TPC + dz_sbnd/2)/const.c_LIGHT
    df["pos_scatt", "1"] = events[0, :nsamples]
    df["pos_scatt", "2"] = events[1, :nsamples]
    df["pos_scatt", "3"] = events[2, :nsamples]


def miniboone_dirt_geometry(df):

    length_events = len(df)
    u0 = np.random.random(length_events)
    phi0 = np.random.random(length_events) * 2. * np.pi
    r0 = radius_cyl_MB * u0**(1./2.)
    z0 = np.random.random(length_events) * l_cyl_MB + start_point_cyl_MB
    
    df["pos_scatt", "0"] = (z0 - start_point_cyl_MB + booster_decay_tunnel)/const.c_LIGHT
    df["pos_scatt", "1"] = r0 * np.cos(phi0)
    df["pos_scatt", "2"] = r0 * np.sin(phi0)
    df["pos_scatt", "3"] = z0

def microboone_dirt_geometry(df):

    # geometry of cylinder_MB for dirt
    length_events = len(df)
    z0 = np.random.random(length_events)*(z_muB_dirt_max - z_muB_dirt_min)  + z_muB_dirt_min

    time = MicroBooNEGlobalTimeOffset + (MicroBooNERandomTimeOffset) * np.random.rand(length_events)
    df["pos_scatt", "0"] = time + (z0)/const.c_LIGHT*1e9 # z0 is negative
    df["pos_scatt", "1"] = np.random.random(length_events)*(x_muB_dirt_max - x_muB_dirt_min)  + x_muB_dirt_min
    df["pos_scatt", "2"] = np.random.random(length_events)*(y_muB_dirt_max - y_muB_dirt_min)  + y_muB_dirt_min
    df["pos_scatt", "3"] = z0

def microboone_active_tpc_geometry_benchmark(df):

    # geometry of cylinder_MB for dirt
    length_events = len(df)
    z0 = np.random.random(length_events)*(z_muB) - z_muB/2.

    time = MicroBooNEGlobalTimeOffset + (MicroBooNERandomTimeOffset) * np.random.rand(length_events)
    df["pos_scatt", "0"] = time  #+ (z0)/const.c_LIGHT*1e9 # z0 is negative no need for this
    df["pos_scatt", "1"] = np.random.random(length_events)*(x_muB) - x_muB/2.
    df["pos_scatt", "2"] = np.random.random(length_events)*(y_muB) - y_muB/2.
    df["pos_scatt", "3"] = z0



def microboone_tpc_geometry(df):

    # geometry of cylinder_MB for dirt
    length_events = len(df)
    z0 = np.random.random(length_events)*(z_muB) - z_muB/2.

    time = MicroBooNEGlobalTimeOffset + (MicroBooNERandomTimeOffset) * np.random.rand(length_events)
    df["pos_scatt", "0"] = time + (z0)/const.c_LIGHT*1e9 # z0 is negative
    df["pos_scatt", "1"] = np.random.random(length_events)*(x_muB) - x_muB/2.
    df["pos_scatt", "2"] = np.random.random(length_events)*(y_muB) - y_muB/2.
    df["pos_scatt", "3"] = z0


def sbnd_dirt_geometry(df):

    # geometry of cylinder_MB for dirt
    length_events = len(df)
    z0 = np.random.random(length_events)*(z_sbnd_dirt_max - z_sbnd_dirt_min) + z_sbnd_dirt_min

    df["pos_scatt", "0"] = (z0 + dz_sbnd/2 + l_baseline_sbnd + gap_sbnd_wall_TPC)/const.c_LIGHT
    df["pos_scatt", "1"] = np.random.random(length_events)*(x_sbnd_dirt_max - x_sbnd_dirt_min)  + x_sbnd_dirt_min
    df["pos_scatt", "2"] = np.random.random(length_events)*(y_sbnd_dirt_max - y_sbnd_dirt_min)  + y_sbnd_dirt_min
    df["pos_scatt", "3"] = z0


# distribute events in df accross the pre-defined spherical MiniBooNE volume
def miniboone_geometry(df):

    nsamples = len(df.index)

    detector_box = np.array([[-600, 600], [-600, 600], [-600, 600]])

    tries = 0
    npoints = 0
    events = np.array(3 * [[]])
    while npoints < nsamples:

        new_detector = Chisel(nsamples=nsamples, box=detector_box)
        new_events = new_detector.events[:, new_detector.sphere(radius=574.6)]
        events = np.concatenate((events, new_events), axis=1)

        npoints += np.shape(new_events)[1]
        tries += nsamples
        if tries > 1e3 * nsamples:
            raise ValueError(f"Geometry sampled too inefficiently, tries = {tries} and npoints = {npoints}. Wrong setup?")

    # guarantee that array has number of samples asked (nsamples)
    df["pos_scatt", "0"] = (events[2, :nsamples] - start_point_cyl_MB + booster_decay_tunnel)/const.c_LIGHT
    df["pos_scatt", "1"] = events[0, :nsamples]
    df["pos_scatt", "2"] = events[1, :nsamples]
    df["pos_scatt", "3"] = events[2, :nsamples]


# assing all events in df a scattering position at 4-position (0,0,0,0)
def point_geometry(df):
    nsamples = len(df.index)
    df["pos_scatt", "0"] = np.zeros((nsamples,))
    df["pos_scatt", "1"] = np.zeros((nsamples,))
    df["pos_scatt", "2"] = np.zeros((nsamples,))
    df["pos_scatt", "3"] = np.zeros((nsamples,))


def place_decay(df, df_column, l_decay_proper_cm, label="decay_pos"):
    """find decay positions given scattering positions and lifetime (sample from exponential) and
            save that in the dataframe (in-place)
    Args:
            df (pd.DataFrame): _description_
            df_column (str): column of the dataframe to use -- this determines which particle we are using
            l_decay_proper_cm (float): the proper decay length of the particle (obtained from vegas or otherwise)
            label (str, optional): label to be used for the dataframe new columns. Defaults to 'decay_pos'.
    """
    # four momentum as numpy array
    p = df[df_column].to_numpy()

    # parent mass
    M = np.sqrt(Cfv.dot4(p, p))
    # momentum
    pvec = np.sqrt(p[:, 0] ** 2 - M ** 2)
    beta = pvec / p[:, 0]
    gamma = 1 / np.sqrt(1 - beta ** 2)
    gammabeta = gamma * beta

    ######################
    # sample displacement from decay propability
    d_decay = np.random.exponential(scale=l_decay_proper_cm * gammabeta)  # centimeters

    # direction of N
    df[label, "0"] = df["pos_scatt", "0"] + d_decay / (beta * const.c_LIGHT)
    df[label, "1"] = df["pos_scatt", "1"] + Cfv.get_3direction(p)[:, 0] * d_decay
    df[label, "2"] = df["pos_scatt", "2"] + Cfv.get_3direction(p)[:, 1] * d_decay
    df[label, "3"] = df["pos_scatt", "3"] + Cfv.get_3direction(p)[:, 2] * d_decay
