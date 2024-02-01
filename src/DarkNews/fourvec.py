from DarkNews import Cfourvec as Cfv

X = 0
Y = 1
Z = 2

##############################
# numpy functions
def cos_opening_angle(x, y):
    return Cfv.get_cos_opening_angle(x, y)

def inv_mass(x, y):
    return Cfv.inv_mass(x, y)

##############################
# dataframe functions
def df_dot4(dx, dy):
    return Cfv.dot4(dx.to_numpy(), dy.to_numpy())

def df_dot3(dx, dy):
    return Cfv.dot3(dx.to_numpy(), dy.to_numpy())

def df_cos_opening_angle(dx, dy):
    return Cfv.get_cos_opening_angle(dx.to_numpy(), dy.to_numpy())

def df_cos_azimuthal(dx):
    return Cfv.get_cosTheta(dx.to_numpy())

def df_inv_mass(dx, dy):
    return inv_mass(dx.to_numpy(), dy.to_numpy())
