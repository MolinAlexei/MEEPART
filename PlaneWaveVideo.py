from meep_optics import OpticalSystem, AsphericLens, ApertureStop, ImagePlane, TelescopeTube, Absorber, Sim, Analysis
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import h5py
import mpi4py
from scipy import optimize
#from mpi4py import MPI
    


coeff = 10 

lens1 = AsphericLens(name = 'Lens 1', 
                     r1 = 327.365, 
                     r2 = np.inf, 
                     c1 = -0.66067, 
                     c2 = 0, 
                     thick = 40, 
                     x = (130.+50.), 
                     y = 0., 
                     diameter = 300)
    
lens2 = AsphericLens(name = 'Lens 2', 
                     r1 = 269.190, 
                     r2 = 6398.02, 
                     c1 = -2.4029, 
                     c2 = 1770.36,
                     thick = 40,
                     diameter = 300, 
                     x = (40.+130.+369.408+50.), 
                     y = 0.)
    
aperture_stop = ApertureStop(name = 'Aperture Stop',
                             pos_x = 50,
                             diameter = 200,
                             thickness = 5,
                             n_refr = 1., 
                             conductivity = 1e7)
    
image_plane = ImagePlane(name = 'Image Plane',
                         pos_x = (50+714.704),
                         diameter = 300,
                         thickness = 2,
                         n_refr = 1, 
                         conductivity = 0)

tube = TelescopeTube(name = 'Tube', thick = 10, center = 165)
absorber = Absorber(name = 'Absorber', thick = 10, center = 155)

#PARAMS
wvl = 10
resolution = 1
dpml = 5

opt_sys = OpticalSystem('test')
opt_sys.set_size(750,340)
opt_sys.add_component(lens1)
opt_sys.add_component(lens2)
opt_sys.add_component(aperture_stop)
opt_sys.add_component(image_plane)
opt_sys.add_component(tube)
opt_sys.assemble_system(dpml = dpml, res = resolution)
opt_sys.write_h5file()



sim = Sim(opt_sys)

sim.define_source(wvl = wvl, x = 20, y = 0, size_x = 0, size_y = 300, sourcetype = 'Plane wave', rot_angle = 14)

sim.run_sim(runtime = 800, 
    simres = resolution, 
    get_mp4 = True, 
    movie_name = 'plane_wave', 
    Nfps = 24, 
    image_every = 5,
    dpi = 150)
