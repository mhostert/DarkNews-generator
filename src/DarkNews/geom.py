import numpy as np
from DarkNews import const
from DarkNews import Cfourvec as Cfv
from numpy.random import choice

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files


################## ROTATION FUNCTIONS ######################
def dot3(p1, p2):
    return p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]


def normalize_3D_vec(v):
    return v / np.sqrt(dot3(v, v))


def cross3(p1, p2):
    px = p1[1] * p2[2] - p1[2] * p2[1]
    py = p1[2] * p2[0] - p1[0] * p2[2]
    pz = p1[0] * p2[1] - p1[1] * p2[0]
    return np.array([px, py, pz])


# rotate v by an angle of theta on the plane perpendicular to k using Rodrigues' rotation formula
def rotate_by_theta(v, k, theta):
    # we first normalize k
    k = normalize_3D_vec(k)

    # Rodrigues' rotation formula
    return np.cos(theta) * v + np.sin(theta) * cross3(k, v) + dot3(k, v) * (1 - np.cos(theta)) * k


# rotate a 3D-vector v using the same minimum rotation to take vector a into vector b
def rotate_similar_to(v, a, b):
    # normalize vectors a and b
    a = normalize_3D_vec(a)
    b = normalize_3D_vec(b)

    # compute normal vector to those and angle
    k = cross3(a, b)
    theta = np.arccos(dot3(a, b))

    # use previous function to compute new vector
    return rotate_by_theta(v, k, theta)


def rotate_dataframe(df):
    particles = ["P_target", "P_recoil", "P_decay_N_parent", "P_decay_ell_plus", "P_decay_ell_minus", "P_decay_N_daughter", "P_decay_photon", "P_projectile"]

    for particle in particles:
        try:
            df.loc[:, (particle, ["1", "2", "3"])] = rotate_similar_to(
                df[particle].to_numpy().T[1:], df.P_projectile.to_numpy().T[1:], df.pos_scatt.to_numpy().T[1:] - df["pos_prod"].to_numpy().T
            ).T
        except:
            continue

    return df


##################################################

# BNB flux and decay pipe DATA
# get flux angle normalization and decay position of pions for BNB
n_ebins = 99
BNB_enu_max = 4.245909093808516  # GeV
BNB_fluxes = np.genfromtxt(files("DarkNews.include.fluxes").joinpath("BNB_angle_energy_normalization.dat").open("r"))
BNB_energies_positions = np.genfromtxt(files("DarkNews.include.fluxes").joinpath("BNB_energy_distances.dat").open("r"))
BNB_energy_nu = BNB_energies_positions[1:, 0]
BNB_energy_nu_bins = np.linspace(0, BNB_enu_max, n_ebins + 1)
BNB_e_bins_angle = np.linspace(0, BNB_enu_max, 100)
BNB_th_bins_angle = np.linspace(-np.pi, np.pi, 100)
BNB_distances_nu = BNB_energies_positions[0, 1:]
BNB_e_vs_z_dist = BNB_energies_positions[1:, 1:]

radius_decay_pipe = 35.0  # cm


def sample_neutrino_origin_at_MiniBooNE(E_nu):
    # Define number of E bins in MiniBooNE flux simulation
    n_ebins = 99

    # Creating e_bins using np.searchsorted and np.clip
    e_bins = np.clip(np.searchsorted(BNB_energy_nu_bins, E_nu, side="right") - 1, 0, n_ebins - 1)

    # Ensure probs_distance is correctly normalized
    probs_distance = np.where(
        np.sum(BNB_e_vs_z_dist, axis=1, keepdims=True) != 0, BNB_e_vs_z_dist / (np.sum(BNB_e_vs_z_dist, axis=1, keepdims=True) + 1e-18), BNB_e_vs_z_dist
    )

    # Flattened choice to handle vectorized approach more efficiently
    choices = np.random.random(size=len(e_bins))

    # Cumulative distribution for more efficient sampling
    cumulative_probs = np.cumsum(probs_distance, axis=1)

    # Allocating space for results
    origin = np.zeros(len(e_bins))

    # Vectorized sampling using cumulative probabilities
    for i, e in enumerate(e_bins):
        if cumulative_probs[e, -1] > 0:  # Ensure it’s not an all-zero row
            origin[i] = BNB_distances_nu[np.searchsorted(cumulative_probs[e], choices[i])]
        else:
            origin[i] = np.random.choice(BNB_distances_nu)

    return origin * 1e2


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
        return ((-dx / 2 < x) & (x < dx / 2)) & ((-dy / 2 < y) & (y < dy / 2)) & ((-dz / 2 < z) & (z < dz / 2))

    def sphere(self, radius, origin=np.zeros((3, 1))):
        x, y, z = self.translate(origin, self.events)
        return x**2 + y**2 + z**2 < radius**2

    def cylinder(self, radius, height, origin=np.zeros((3, 1))):
        x, y, z = self.translate(origin, self.events)
        return (x**2 + y**2 < radius**2) & (-height / 2 < z) & (z < height / 2)

    def spherical_cap(self, radius, zenith_max, origin=np.zeros((3, 1))):
        x, y, z = self.translate(origin, self.events)
        rpolar = np.sqrt(x**2 + y**2)
        r = np.sqrt(rpolar**2 + z**2)
        zenith = np.arctan2(z, rpolar)
        h = radius * (1 - np.cos(zenith_max))
        return (r < radius) | (zenith < zenith_max) | (z < -radius + h)

    def microboone_cryostat(self):
        x = self.x
        y = self.y
        z = self.z

        # accept in tube
        r_polar = np.sqrt(x**2 + y**2)
        mask_tube = (-z_t / 2 < z) & (z < z_t / 2) & (r_polar < r_t)

        # coordinates in sphere 1
        z1 = z - (-z_t / 2 - cap_gap + r_c * ctheta_c)
        r1 = np.sqrt(r_polar**2 + z1**2)
        inc1 = np.arctan2(z1, r_polar)  # inclination angle

        # accept in front cap
        mask_cap1 = (r1 < r_c) & (inc1 < theta_c) & (z1 < -r_c + h)

        # coordinates in sphere 2
        z2 = z + (-z_t / 2 - cap_gap - h + r_c)
        r2 = np.sqrt(r_polar**2 + z2**2)
        inc2 = np.arctan2(-z2, r_polar)  # inclination angle

        # accept in back cap
        mask_cap2 = (r2 < r_c) & (inc2 < theta_c) & (z2 > r_c - h)

        fraction_of_height_wLAr = 0.85
        mask_full = (mask_tube + mask_cap1 + mask_cap2) & (y + r_t < 2 * r_t * fraction_of_height_wLAr)

        return mask_full


# @dataclass
# class MicroBooNE:
# geometry of tpc
z_muB = 1040.0
x_muB = 256.0
y_muB = 232.0

# Tube parameters - cryostat
r_t = 191.61
z_t = 1086.49

# geometry of cone_muB for dirt
l_cone_muB = 400e2
end_point_cone_muB = -215 - z_t / 2
l_baseline_muB = 470e2
radius_cone_outer_muB = 1.5 * r_t
radius_cone_inner_muB = 38.7955

l_cone_excluded_muB = 7000 + end_point_cone_muB
start_point_cone_muB = end_point_cone_muB - l_cone_muB


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
# geometry of cone_MB for dirt
l_baseline_MB = 541e2
radius_MB_outer = 1370 / 2.0
radius_cone_outer_MB = 1.5 * radius_MB_outer
radius_cone_inner_MB = 104.736
l_cone_MB = 47400.0
l_cone_excluded = 5380.0

end_point_cone_MB = -1320.0
start_point_cone_MB = end_point_cone_MB - l_cone_MB

# SBND - FIX ME, are SBND and MicroBooNE centered along the same Z axis?
booster_decay_tunnel = 50e2
l_baseline_sbnd = 110e2
gap_sbnd_wall_TPC = 1e2
dx_sbnd = 4e2
dy_sbnd = 4e2
dz_sbnd = 5e2

x_sbnd_dirt_min = -1100.0
x_sbnd_dirt_max = 1100.0
y_sbnd_dirt_min = -1600.0
y_sbnd_dirt_max = 600.0
z_sbnd_dirt_max = -dz_sbnd / 2 - gap_sbnd_wall_TPC
z_sbnd_dirt_min = -l_baseline_sbnd + booster_decay_tunnel

# geometry of cone_sbnd for dirt
l_cone_sbnd = 56.5e2
sbnd_gap = 1e2
end_point_cone_sbnd = -sbnd_gap - dz_sbnd / 2.0
radius_cone_outer_sbnd = 1.0 * dx_sbnd
radius_cone_inner_sbnd = 1.87793e2

l_cone_excluded_sbnd = 50e2
start_point_cone_sbnd = end_point_cone_sbnd - l_cone_sbnd

# Icarus
l_baseline_icarus = 600e2
x_icarus = 3.6e2 * 2
y_icarus = 3.9e2
z_icarus = 19.6e2

# geometry of cone_icarus for dirt
l_cone_icarus = 400e2
icarus_gap = 1e2
end_point_cone_icarus = -icarus_gap - z_icarus / 2.0
radius_cone_outer_icarus = 1.5 * x_icarus
radius_cone_inner_icarus = 346.79

l_cone_excluded_icarus = 18919
start_point_cone_icarus = end_point_cone_icarus - l_cone_icarus


# distribute events in df accross the pre-defined MicroBooNE (cryostat) volume
def microboone_geometry(df):

    nsamples = len(df.index)

    detector_box = np.array([[-200, 200], [-200, 200], [-800, 800]])

    tries = 0
    npoints = 0
    events = np.array(3 * [[]])
    while npoints < nsamples:

        new_detector = Chisel(nsamples=nsamples, box=detector_box)
        new_events = new_detector.events[:, new_detector.microboone_cryostat()]
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

    # Compute the mean position where the pions decayed
    n_ebins = 99
    E_nu = df["P_projectile", "0"].values
    e_bins = np.searchsorted(BNB_energy_nu_bins, E_nu, side="right") - 1
    if n_ebins in e_bins:
        mask = e_bins >= n_ebins
        e_bins[mask] = n_ebins - 1

    probs_distance = np.ones_like(BNB_e_vs_z_dist)
    for i in range(len(probs_distance)):
        if BNB_e_vs_z_dist[e_bins[i], :].sum() != 0:
            probs_distance[i] = BNB_e_vs_z_dist[e_bins[i], :]
    origin = np.array([choice(BNB_distances_nu, 1, p=probs_distance[e_bins[i]] / probs_distance[e_bins[i]].sum())[0] * 1e2 for i in range(len(e_bins))])

    length_events = len(df)
    u_normal = np.random.random(length_events)
    phi_normal = np.random.random(length_events) * 2.0 * np.pi
    r_normal = radius_decay_pipe * np.sqrt(u_normal)
    df["pos_prod", "1"] = r_normal * np.cos(phi_normal)
    df["pos_prod", "2"] = r_normal * np.sin(phi_normal)
    df["pos_prod", "3"] = origin - l_baseline_muB

    # RESCALE WEIGHTS
    # rescale the weights with respect to the distance
    distances = np.sqrt(df["pos_scatt", "1"].values ** 2 + df["pos_scatt", "2"].values ** 2 + (df["pos_scatt", "3"].values - df["pos_prod", "3"].values) ** 2)
    df.w_event_rate *= ((l_baseline_muB - origin) / distances) ** 2

    # rotate momenta
    df = rotate_dataframe(df)


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
    df["pos_scatt", "0"] = (events[2, :nsamples] + l_baseline_sbnd + gap_sbnd_wall_TPC + dz_sbnd / 2) / const.c_LIGHT
    df["pos_scatt", "1"] = events[0, :nsamples]
    df["pos_scatt", "2"] = events[1, :nsamples]
    df["pos_scatt", "3"] = events[2, :nsamples]

    origin = sample_neutrino_origin_at_MiniBooNE(df["P_projectile", "0"].values)

    length_events = len(df)
    u_normal = np.random.random(length_events)
    phi_normal = np.random.random(length_events) * 2.0 * np.pi
    r_normal = radius_decay_pipe * np.sqrt(u_normal)
    df["pos_prod", "1"] = r_normal * np.cos(phi_normal)
    df["pos_prod", "2"] = r_normal * np.sin(phi_normal)
    df["pos_prod", "3"] = origin - l_baseline_sbnd

    # RESCALE WEIGHTS
    # rescale the weights with respect to the distance
    distances = np.sqrt(df["pos_scatt", "1"].values ** 2 + df["pos_scatt", "2"].values ** 2 + (df["pos_scatt", "3"].values - df["pos_prod", "3"].values) ** 2)
    df.w_event_rate *= ((l_baseline_sbnd - origin) / distances) ** 2

    # rotate momenta
    df = rotate_dataframe(df)


def icarus_geometry(df):

    nsamples = len(df)

    detector_box = np.array([[-300, 300], [-300, 300], [-300, 300]])

    tries = 0
    npoints = 0
    events = np.array(3 * [[]])
    while npoints < nsamples:

        new_detector = Chisel(nsamples=nsamples, box=detector_box)
        new_events = new_detector.events[:, new_detector.rectangle(dx=x_icarus, dy=y_icarus, dz=z_icarus)]
        events = np.concatenate((events, new_events), axis=1)

        npoints += np.shape(new_events)[1]
        tries += nsamples
        if tries > 1e3 * nsamples:
            raise ValueError(f"Geometry sampled too inefficiently, tries = {tries} and npoints = {npoints}. Wrong setup?")

    # guarantee that array has number of samples asked (nsamples)
    df["pos_scatt", "0"] = (events[2, :nsamples] + l_baseline_icarus + icarus_gap + z_icarus / 2) / const.c_LIGHT
    df["pos_scatt", "1"] = events[0, :nsamples]
    df["pos_scatt", "2"] = events[1, :nsamples]
    df["pos_scatt", "3"] = events[2, :nsamples]

    origin = sample_neutrino_origin_at_MiniBooNE(df["P_projectile", "0"].values)

    length_events = len(df)
    u_normal = np.random.random(length_events)
    phi_normal = np.random.random(length_events) * 2.0 * np.pi
    r_normal = radius_decay_pipe * np.sqrt(u_normal)
    df["pos_prod", "1"] = r_normal * np.cos(phi_normal)
    df["pos_prod", "2"] = r_normal * np.sin(phi_normal)
    df["pos_prod", "3"] = origin - l_baseline_icarus

    # RESCALE WEIGHTS
    # rescale the weights with respect to the distance
    distances = np.sqrt(df["pos_scatt", "1"].values ** 2 + df["pos_scatt", "2"].values ** 2 + (df["pos_scatt", "3"].values - df["pos_prod", "3"].values) ** 2)
    df.w_event_rate *= ((l_baseline_icarus - origin) / distances) ** 2

    # rotate momenta
    df = rotate_dataframe(df)


def miniboone_dirt_geometry(df):

    length_events = len(df)
    a = l_cone_MB + l_cone_excluded  # height of the cone
    b = radius_cone_outer_MB  # base of the cone
    fraction_dirt = ((a * b**2) - (l_cone_excluded * radius_cone_inner_MB**2)) / (a * b**2)
    correction_fraction = 1.5  # just to be sure we produce more than we need
    n_sample = int(length_events / fraction_dirt * correction_fraction)

    a = l_cone_MB + l_cone_excluded  # height of the cone
    b = radius_cone_outer_MB  # base of the cone
    h = a * np.random.random(n_sample) ** (1.0 / 3.0)
    r = (b / a) * h * np.sqrt(np.random.random(n_sample))
    phi = np.random.random(n_sample) * 2.0 * np.pi

    x0 = r * np.cos(phi)
    y0 = r * np.sin(phi)
    z0 = h

    mask_truncate_cone = z0 >= l_cone_excluded
    x0 = x0[mask_truncate_cone][:length_events]
    y0 = y0[mask_truncate_cone][:length_events]
    z0 = (z0[mask_truncate_cone][:length_events]) - a + end_point_cone_MB

    df["pos_scatt", "0"] = (z0 - start_point_cone_MB + booster_decay_tunnel) / const.c_LIGHT
    df["pos_scatt", "1"] = x0
    df["pos_scatt", "2"] = y0
    df["pos_scatt", "3"] = z0

    origin = sample_neutrino_origin_at_MiniBooNE(df["P_projectile", "0"].values)

    length_events = len(df)
    u_normal = np.random.random(length_events)
    phi_normal = np.random.random(length_events) * 2.0 * np.pi
    r_normal = radius_decay_pipe * np.sqrt(u_normal)
    df["pos_prod", "1"] = r_normal * np.cos(phi_normal)
    df["pos_prod", "2"] = r_normal * np.sin(phi_normal)
    df["pos_prod", "3"] = origin - l_baseline_MB

    # RESCALE WEIGHTS
    # rescale the weights with respect to the distance
    distances = np.sqrt(df["pos_scatt", "1"].values ** 2 + df["pos_scatt", "2"].values ** 2 + (df["pos_scatt", "3"].values - df["pos_prod", "3"].values) ** 2)
    df.w_event_rate *= ((l_baseline_MB - origin) / distances) ** 2

    # rotate momenta
    df = rotate_dataframe(df)


def microboone_dirt_geometry(df):

    # geometry of cylinder_MB for dirt
    length_events = len(df)
    a = l_cone_muB + l_cone_excluded_muB  # height of the cone
    b = radius_cone_outer_muB  # base of the cone
    fraction_dirt = ((a * b**2) - (l_cone_excluded_muB * radius_cone_inner_muB**2)) / (a * b**2)
    correction_fraction = 2.0  # just to be sure we produce more than we need
    n_sample = int(length_events / fraction_dirt * correction_fraction)

    h = a * np.random.random(n_sample) ** (1.0 / 3.0)
    r = (b / a) * h * np.sqrt(np.random.random(n_sample))
    phi = np.random.random(n_sample) * 2.0 * np.pi

    x0 = r * np.cos(phi)
    y0 = r * np.sin(phi)
    z0 = h

    mask_truncate_cone = z0 >= l_cone_excluded_muB
    x0 = x0[mask_truncate_cone][:length_events]
    y0 = y0[mask_truncate_cone][:length_events]
    z0 = (z0[mask_truncate_cone][:length_events]) - a + end_point_cone_muB

    df["pos_scatt", "0"] = (z0 - start_point_cone_muB + booster_decay_tunnel) / const.c_LIGHT
    df["pos_scatt", "1"] = x0
    df["pos_scatt", "2"] = y0
    df["pos_scatt", "3"] = z0

    origin = sample_neutrino_origin_at_MiniBooNE(df["P_projectile", "0"].values)

    u_normal = np.random.random(length_events)
    phi_normal = np.random.random(length_events) * 2.0 * np.pi
    r_normal = radius_decay_pipe * np.sqrt(u_normal)
    df["pos_prod", "1"] = r_normal * np.cos(phi_normal)
    df["pos_prod", "2"] = r_normal * np.sin(phi_normal)
    df["pos_prod", "3"] = origin - l_baseline_muB

    # RESCALE WEIGHTS
    # rescale the weights with respect to the distance
    distances = np.sqrt(df["pos_scatt", "1"].values ** 2 + df["pos_scatt", "2"].values ** 2 + (df["pos_scatt", "3"].values - df["pos_prod", "3"].values) ** 2)
    df.w_event_rate *= ((l_baseline_muB - origin) / distances) ** 2

    # rotate momenta
    df = rotate_dataframe(df)


def sbnd_dirt_cone_geometry(df):

    # geometry of cone_sbnd for dirt
    length_events = len(df)
    a = l_cone_sbnd + l_cone_excluded_sbnd  # height of the cone
    b = radius_cone_outer_sbnd  # base of the cone
    fraction_dirt = ((a * b**2) - (l_cone_excluded_sbnd * radius_cone_inner_sbnd**2)) / (a * b**2)
    correction_fraction = 2.0  # just to be sure we produce more than we need
    n_sample = int(length_events / fraction_dirt * correction_fraction)

    h = a * np.random.random(n_sample) ** (1.0 / 3.0)
    r = (b / a) * h * np.sqrt(np.random.random(n_sample))
    phi = np.random.random(n_sample) * 2.0 * np.pi

    x0 = r * np.cos(phi)
    y0 = r * np.sin(phi)
    z0 = h

    mask_truncate_cone = z0 >= l_cone_excluded_sbnd
    x0 = x0[mask_truncate_cone][:length_events]
    y0 = y0[mask_truncate_cone][:length_events]
    z0 = (z0[mask_truncate_cone][:length_events]) - a + end_point_cone_sbnd

    df["pos_scatt", "0"] = (z0 - start_point_cone_sbnd + booster_decay_tunnel) / const.c_LIGHT
    df["pos_scatt", "1"] = x0
    df["pos_scatt", "2"] = y0
    df["pos_scatt", "3"] = z0

    origin = sample_neutrino_origin_at_MiniBooNE(df["P_projectile", "0"].values)

    u_normal = np.random.random(length_events)
    phi_normal = np.random.random(length_events) * 2.0 * np.pi
    r_normal = radius_decay_pipe * np.sqrt(u_normal)
    df["pos_prod", "1"] = r_normal * np.cos(phi_normal)
    df["pos_prod", "2"] = r_normal * np.sin(phi_normal)
    df["pos_prod", "3"] = origin - l_baseline_sbnd

    # RESCALE WEIGHTS
    # rescale the weights with respect to the distance
    distances = np.sqrt(df["pos_scatt", "1"].values ** 2 + df["pos_scatt", "2"].values ** 2 + (df["pos_scatt", "3"].values - df["pos_prod", "3"].values) ** 2)
    df.w_event_rate *= ((l_baseline_sbnd - origin) / distances) ** 2

    # rotate momenta
    df = rotate_dataframe(df)


def icarus_dirt_geometry(df):

    # geometry of cylinder_icarus for dirt
    length_events = len(df)
    a = l_cone_icarus + l_cone_excluded_icarus  # height of the cone
    b = radius_cone_outer_icarus  # base of the cone
    fraction_dirt = ((a * b**2) - (l_cone_excluded_icarus * radius_cone_inner_icarus**2)) / (a * b**2)
    correction_fraction = 2.0  # just to be sure we produce more than we need
    n_sample = int(length_events / fraction_dirt * correction_fraction)

    h = a * np.random.random(n_sample) ** (1.0 / 3.0)
    r = (b / a) * h * np.sqrt(np.random.random(n_sample))
    phi = np.random.random(n_sample) * 2.0 * np.pi

    x0 = r * np.cos(phi)
    y0 = r * np.sin(phi)
    z0 = h

    mask_truncate_cone = z0 >= l_cone_excluded_icarus
    x0 = x0[mask_truncate_cone][:length_events]
    y0 = y0[mask_truncate_cone][:length_events]
    z0 = (z0[mask_truncate_cone][:length_events]) - a + end_point_cone_icarus

    df["pos_scatt", "0"] = (z0 - start_point_cone_icarus + booster_decay_tunnel) / const.c_LIGHT
    df["pos_scatt", "1"] = x0
    df["pos_scatt", "2"] = y0
    df["pos_scatt", "3"] = z0

    origin = sample_neutrino_origin_at_MiniBooNE(df["P_projectile", "0"].values)

    u_normal = np.random.random(length_events)
    phi_normal = np.random.random(length_events) * 2.0 * np.pi
    r_normal = radius_decay_pipe * np.sqrt(u_normal)
    df["pos_prod", "1"] = r_normal * np.cos(phi_normal)
    df["pos_prod", "2"] = r_normal * np.sin(phi_normal)
    df["pos_prod", "3"] = origin - l_baseline_icarus

    # RESCALE WEIGHTS
    # rescale the weights with respect to the distance
    distances = np.sqrt(df["pos_scatt", "1"].values ** 2 + df["pos_scatt", "2"].values ** 2 + (df["pos_scatt", "3"].values - df["pos_prod", "3"].values) ** 2)
    df.w_event_rate *= ((l_baseline_icarus - origin) / distances) ** 2

    # rotate momenta
    df = rotate_dataframe(df)


def microboone_tpc_geometry(df):

    # geometry of cylinder_MB for dirt
    length_events = len(df)
    z0 = np.random.random(length_events) * (z_muB) - z_muB / 2.0

    time = MicroBooNEGlobalTimeOffset + (MicroBooNERandomTimeOffset) * np.random.rand(length_events)
    df["pos_scatt", "0"] = time + (z0) / const.c_LIGHT * 1e9  # z0 is negative
    df["pos_scatt", "1"] = np.random.random(length_events) * (x_muB) - x_muB / 2.0
    df["pos_scatt", "2"] = np.random.random(length_events) * (y_muB) - y_muB / 2.0
    df["pos_scatt", "3"] = z0

    origin = sample_neutrino_origin_at_MiniBooNE(df["P_projectile", "0"].values)

    u_normal = np.random.random(length_events)
    phi_normal = np.random.random(length_events) * 2.0 * np.pi
    r_normal = radius_decay_pipe * np.sqrt(u_normal)
    df["pos_prod", "1"] = r_normal * np.cos(phi_normal)
    df["pos_prod", "2"] = r_normal * np.sin(phi_normal)
    df["pos_prod", "3"] = origin - l_baseline_muB

    # RESCALE WEIGHTS
    # rescale the weights with respect to the distance
    distances = np.sqrt(df["pos_scatt", "1"].values ** 2 + df["pos_scatt", "2"].values ** 2 + (df["pos_scatt", "3"].values - df["pos_prod", "3"].values) ** 2)
    df.w_event_rate *= ((l_baseline_muB - origin) / distances) ** 2

    # rotate momenta
    df = rotate_dataframe(df)


def sbnd_dirt_geometry(df):

    # geometry of cylinder_MB for dirt
    length_events = len(df)
    z0 = np.random.random(length_events) * (z_sbnd_dirt_max - z_sbnd_dirt_min) + z_sbnd_dirt_min

    df["pos_scatt", "0"] = (z0 + dz_sbnd / 2 + l_baseline_sbnd + gap_sbnd_wall_TPC) / const.c_LIGHT
    df["pos_scatt", "1"] = np.random.random(length_events) * (x_sbnd_dirt_max - x_sbnd_dirt_min) + x_sbnd_dirt_min
    df["pos_scatt", "2"] = np.random.random(length_events) * (y_sbnd_dirt_max - y_sbnd_dirt_min) + y_sbnd_dirt_min
    df["pos_scatt", "3"] = z0

    E_nu = df["P_projectile", "0"].values
    origin = sample_neutrino_origin_at_MiniBooNE(E_nu)

    u_normal = np.random.random(length_events)
    phi_normal = np.random.random(length_events) * 2.0 * np.pi
    r_normal = radius_decay_pipe * np.sqrt(u_normal)
    df["pos_prod", "1"] = r_normal * np.cos(phi_normal)
    df["pos_prod", "2"] = r_normal * np.sin(phi_normal)
    df["pos_prod", "3"] = origin - l_baseline_sbnd

    # RESCALE WEIGHTS
    # rescale the weights with respect to the distance
    distances = np.sqrt(df["pos_scatt", "1"].values ** 2 + df["pos_scatt", "2"].values ** 2 + (df["pos_scatt", "3"].values - df["pos_prod", "3"].values) ** 2)
    df.w_event_rate *= ((l_baseline_sbnd - origin) / distances) ** 2
    # rescaling with respect to angle
    theta_nu = np.arccos(
        (df["pos_scatt", "3"] - df["pos_prod", "3"])
        / np.sqrt(
            ((df["pos_scatt", "1"] - df["pos_prod", "1"])) ** 2
            + ((df["pos_scatt", "2"] - df["pos_prod", "2"])) ** 2
            + ((df["pos_scatt", "3"] - df["pos_prod", "3"])) ** 2
        )
    )

    e_bins = np.searchsorted(BNB_e_bins_angle, E_nu, side="right") - 1
    n_ebins = len(BNB_fluxes)
    if n_ebins in e_bins:
        mask = e_bins >= n_ebins
        e_bins[mask] = n_ebins - 1
    th_bins = np.searchsorted(BNB_th_bins_angle, theta_nu, side="right") - 1
    n_ebins = len(BNB_fluxes[0])
    if n_ebins in e_bins:
        mask = e_bins >= n_ebins
        th_bins[mask] = n_ebins - 1

    renorm_flux_angle = np.array([BNB_fluxes[e_bins[i], th_bins[i]] for i in range(length_events)])
    df.w_event_rate *= renorm_flux_angle

    # rotate momenta
    df = rotate_dataframe(df)


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
    df["pos_scatt", "0"] = (events[2, :nsamples] - start_point_cone_MB + booster_decay_tunnel) / const.c_LIGHT
    df["pos_scatt", "1"] = events[0, :nsamples]
    df["pos_scatt", "2"] = events[1, :nsamples]
    df["pos_scatt", "3"] = events[2, :nsamples]

    # Compute the mean position where the pions decayed
    origin = sample_neutrino_origin_at_MiniBooNE(df["P_projectile", "0"].values)

    length_events = len(df)
    u_normal = np.random.random(length_events)
    phi_normal = np.random.random(length_events) * 2.0 * np.pi
    r_normal = radius_decay_pipe * np.sqrt(u_normal)
    df["pos_prod", "1"] = r_normal * np.cos(phi_normal)
    df["pos_prod", "2"] = r_normal * np.sin(phi_normal)
    df["pos_prod", "3"] = origin - l_baseline_MB

    # RESCALE WEIGHTS
    # rescale the weights with respect to the distance
    distances = np.sqrt(df["pos_scatt", "1"].values ** 2 + df["pos_scatt", "2"].values ** 2 + (df["pos_scatt", "3"].values - df["pos_prod", "3"].values) ** 2)
    df.w_event_rate *= ((l_baseline_MB - origin) / distances) ** 2

    # rotate momenta
    df = rotate_dataframe(df)


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
    pvec = np.sqrt(p[:, 0] ** 2 - M**2)
    beta = pvec / p[:, 0]
    gamma = 1 / np.sqrt(1 - beta**2)
    gammabeta = gamma * beta

    ######################
    # sample displacement from decay propability
    d_decay = np.random.exponential(scale=l_decay_proper_cm * gammabeta)  # centimeters

    # direction of N
    df[label, "0"] = df["pos_scatt", "0"] + d_decay / (beta * const.c_LIGHT)
    df[label, "1"] = df["pos_scatt", "1"] + Cfv.get_3direction(p)[:, 0] * d_decay
    df[label, "2"] = df["pos_scatt", "2"] + Cfv.get_3direction(p)[:, 1] * d_decay
    df[label, "3"] = df["pos_scatt", "3"] + Cfv.get_3direction(p)[:, 2] * d_decay
