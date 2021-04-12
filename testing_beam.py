from meep_optics import OpticalSystem, AsphericLens, ApertureStop, ImagePlane, TelescopeTube, Absorber, Sim, Analysis
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import h5py
from scipy import optimize
from mpi4py import MPI
    
lens1 = AsphericLens(name = 'Lens 1', 
                     r1 = 327.365, 
                     r2 = np.inf, 
                     c1 = -0.66067, 
                     c2 = 0, 
                     thick = 40, 
                     x = 130.+10., 
                     y = 0., 
                     AR_left = 0, AR_right = 0)
    
lens2 = AsphericLens(name = 'Lens 2', 
                     r1 = 269.190, 
                     r2 = 6398.02, 
                     c1 = -2.4029, 
                     c2 = 1770.36,
                     thick = 40, 
                     x = 40.+130.+369.408+10., 
                     y = 0.,
                     AR_left = 0, AR_right = 0)
    
aperture_stop = ApertureStop(name = 'Aperture Stop',
                             pos_x = 10,
                             diameter = 200,
                             thickness = 5,
                             n_refr = 1, 
                             conductivity = 1e7)
    
image_plane = ImagePlane(name = 'Image Plane',
                         pos_x = 10+714.704,
                         diameter = 300,
                         thickness = 2,
                         n_refr = 1, 
                         conductivity = 0)

tube = TelescopeTube('Tube')
absorber = Absorber('Absorber')



def system_assembly(lens1, lens2, aperture_stop, image_plane, res, dpml, 
                    bub_radius = 0, bub_nb = 0, r_factor = 0,
                    telescope_tube = False,
                    absorb = False):
    opt_sys = OpticalSystem('test')
    opt_sys.set_size(750,300)
    opt_sys.add_component(lens1)
    opt_sys.add_component(lens2)
    opt_sys.add_component(aperture_stop)
    opt_sys.add_component(image_plane)

    if telescope_tube :
        opt_sys.add_component(tube)
    if absorb :
        opt_sys.add_component(absorber)

    opt_sys.assemble_system(dpml = dpml, resolution = res)
    if bub_radius>0 and bub_nb >0 and r_factor >0:
        opt_sys.make_lens_bubbles(bub_radius, bub_nb, 15, r_factor = r_factor)
    opt_sys.write_h5file()
    
    return opt_sys



#PARAMS
wvl = 10
beam_w0 = 30
resolution = 1
dpml = 5


#RUN SIM
opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane, resolution, dpml)
sim = Sim(opt_sys)
analysis = Analysis(sim)  

analysis.image_plane_beams(wavelength = wvl, sim_resolution = resolution, beam_w0 = beam_w0) 

#PLOT FIELD
analysis.sim.plot_efield()

freq, fft = analysis.beam_FT(precision_factor = 15)

#PLOT BEAM
analysis.plotting(freq, fft, wvl, deg_range= 40, print_fwhm = True, savefig = True, path_name = 'testing_w0')