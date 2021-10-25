#from meep_optics import OpticalSystem, AsphericLens, ApertureStop, ImagePlane, TelescopeTube, Absorber, Sim, Analysis
import meep_optics as mpo
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import h5py
from scipy import optimize
#from mpi4py import MPI

do_sim = True
do_analysis = False

lens1 = mpo.AsphericLens(name = 'Lens 1',
                     r1 = 327.365, 
                     r2 = np.inf, 
                     c1 = -0.66067, 
                     c2 = 0, 
                     thick = 40, 
                     x = (130.+50.), 
                     y = 0., 
                     diameter = 300)
    
lens2 = mpo.AsphericLens(name = 'Lens 2',
                     r1 = 269.190, 
                     r2 = 6398.02, 
                     c1 = -2.4029, 
                     c2 = 1770.36,
                     thick = 40,
                     diameter = 300, 
                     x = (40.+130.+369.408+50.), 
                     y = 0.)
    
aperture_stop = mpo.ApertureStop(name = 'Aperture Stop',
                             pos_x = 50,
                             diameter = 200,
                             thickness = 5,
                             n_refr = 1., 
                             conductivity = 1e7)
    
image_plane = mpo.ImagePlane(name = 'Image Plane',
                         pos_x = (50+714.704),
                         diameter = 300,
                         thickness = 2,
                         n_refr = 1, 
                         conductivity = 0)

tube = mpo.TelescopeTube('Tube', center=0.)
absorber = mpo.Absorber('Absorber', center=0.)

def system_assembly(lens1, lens2, aperture_stop, image_plane, res, dpml):
    opt_sys = mpo.OpticalSystem('test')
    opt_sys.set_size(750, 350)
    #opt_sys.set_size(1500, 700)
    #opt_sys.set_size(3000, 1500)
    opt_sys.add_component(lens1)
    opt_sys.add_component(lens2)
    opt_sys.add_component(aperture_stop)
    opt_sys.add_component(image_plane)
    opt_sys.add_component(tube)
    opt_sys.assemble_system(dpml = dpml, res = res)
    opt_sys.write_h5file()
    
    return opt_sys

#PARAMS
wvl = 10
resolution = 8
dpml = 5
runtime = 100

#wvl = 50
#resolution = 2
#dpml = 5
#runtime = 100

print('asdf')

FFT_list = []
opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane,
    resolution, dpml)

dist_unit = 0.001
opt_sys.sys_info(dist_unit, wvl=wvl)

sim = mpo.Sim(opt_sys)

sim.define_source(wvl = wvl, x = 20, y = 0, size_x = 0, size_y = 300,
    sourcetype = 'Plane wave', rot_angle = 14)

if do_sim:

    #sim.run_sim(runtime = 100, get_mp4 = True,
    #        # dpml=dpml, sim_resolution = resolution
    #        movie_name = 'plane_wave_linear', Nfps = 24, image_every = 5, dpi=300)

    sim.run_sim(runtime = 1000, get_mp4 = True,
            movie_name = 'plane_wave_linear_short',
            Nfps = 24, image_every = 5, dpi=300)

### Analysis part of the run
if do_analysis:

    analysis = mpo.Analysis(sim)
    analysis.image_plane_beams(wvl=wvl, simres=resolution, runtime=runtime)

    print('analysis.list_efields')
    print(analysis.list_efields)

    sim.get_MEEP_ff()

