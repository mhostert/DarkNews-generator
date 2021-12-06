
	# ALL UNITS -- radius and cm 

	# Tube parameters
	r_t = 191.61
	z_t  = 1086.49

	# cap parameters
	r_c = 305.250694958
	theta_c = 38.8816337686*const.degrees_to_rad
	ctheta_c = np.cos(theta_c)
	stheta_c = np.sin(theta_c)
	zend_c = 305.624305042
	# dz = r_c*(1.-np.sqrt(1.0-(r_t/r_c)**2))
	dz = r_c*(1.-ctheta_c)
	cap_gap = 0.37361008

	############################
	# Generating events in tube
	# tube z
	zmin = 0.0
	zmax = z_t
	z = (zmin + (zmax -zmin)*np.random.rand(size))
	
	# tube radius
	rmin=0.0
	rmax=r_t
	r = ((rmin**2 + (rmax**2 -rmin**2)*np.random.rand(size)))**(1./2.0)
	
	# tube phi
	phimin=0.0
	phimax=2*np.pi
	phi = (phimin + (phimax - phimin)*np.random.rand(size))
	cphi=np.cos(phi)

	x=r*cphi
	y=r*np.sin(phi)

	############################
	# Generating events in caps
	# cap phi
	phicmin=0.0
	phicmax=2*np.pi
	phic = (phicmin + (phicmax -phicmin)*np.random.rand(size))

	# cap theta
	ctcmax=np.cos(0.0)
	ctcmin=ctheta_c
	t = np.arccos(ctcmin + (ctcmax -ctcmin)*np.random.rand(size))
	
	ct = np.cos(t)
	st = np.sin(t)
	# cap truncated radius
	#radius spanned between end of cylinder and the cap
	rprime = (r_c - dz)/ct
	rcmin=rprime
	rcmax=r_c

	# first cap
	rc1 = ((rcmin**3 + (rcmax**3 -rcmin**3)*np.random.rand(size)))**(1.0/3.0)
	# second cap
	rc2 = ((rcmin**3 + (rcmax**3 -rcmin**3)*np.random.rand(size)))**(1.0/3.0)

	xc=rc1*st*np.cos(phic)
	yc=rc1*st*np.sin(phic)


	# Translate
	z_c1 = -rc1*ct + (-cap_gap-dz+r_c)
	z_c2 = rc2*ct + (z_t+cap_gap+dz-r_c)

	
	#############
	# decide which volume to pick from
	vol_tube = np.pi*r_t**2*(z_t+2*cap_gap)
	
	#  V_(cap)=1/6 pih(3a^2+h^2). 
	vol_cap = 1.0/6.0*np.pi*dz*(3*(r_c*stheta_c)**2+dz**2)

	weights = np.concatenate([vol_tube*np.ones(size)*0, vol_cap*np.ones(size), vol_cap*np.ones(size)*0])
	
	indices_comb = random.choices(range(3*size), weights=weights, k = size)
	x_comb = np.concatenate([x,xc,xc])[indices_comb]
	y_comb = np.concatenate([y,yc,yc])[indices_comb]
	z_comb = np.concatenate([z,z_c1,z_c2])[indices_comb]

	#######
	# Using initial time of the det only! 
	# CROSS CHECK THIS VALUE
	tmin=3.200e3;
	tmax=3.200e3 # ticks? ns?
	time = (tmin + (tmax -tmin)*np.random.rand(size))

	return np.array([time,x_comb,y_comb,z_comb])
	# return np.array([time,xc,yc,z_c1]).T



# def geometry_muboone(size):
# 	######################### HEPevt format
# 	# Detector geometry -- choose random position
# 	xmin=0;xmax=256. # cm
# 	ymin=-115.;ymax=115. # cm
# 	zmin=0.;zmax=1045. # cm

# 	#######
# 	# Using initial time of the det only! 
# 	# CROSS CHECK THIS VALUE
# 	tmin=3.200e3;
# 	tmax=3.200e3 # ticks? ns?

# 	####################
# 	# scaling it to a smaller size around the central value
# 	restriction = 0.3 

# 	xmax = xmax - restriction*(xmax -xmin)
# 	ymax = ymax - restriction*(ymax -ymin)
# 	zmax = zmax - restriction*(zmax -zmin)
# 	tmax = tmax - restriction*(tmax -tmin)

# 	xmin = xmin + restriction*(xmax-xmin)
# 	ymin = ymin + restriction*(ymax-ymin)
# 	zmin = zmin + restriction*(zmax-zmin)
# 	tmin = tmin + restriction*(tmax-tmin)

# 	# generating entries
# 	x = (xmin + (xmax -xmin)*np.random.rand(size))
# 	y = (ymin + (ymax -ymin)*np.random.rand(size))
# 	z = (zmin + (zmax -zmin)*np.random.rand(size))
# 	t = (tmin + (tmax -tmin)*np.random.rand(size))

# 	return t,x,y,z