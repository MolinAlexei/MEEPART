from meep_optics import OpticalSystem, AsphericLens, ApertureStop, ImagePlane, TelescopeTube, Absorber, Sim, Analysis
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import h5py
from scipy import optimize
from mpi4py import MPI
    
coeff = 10

lens1 = AsphericLens(name = 'Lens 1', 
                     r1 = 327.365*coeff, 
                     r2 = np.inf, 
                     c1 = -0.66067, 
                     c2 = 0, 
                     thick = 40*coeff, 
                     x = (130.+10.)*coeff, 
                     y = 0., 
                     diameter = 300*coeff,
                     AR_left = 5., AR_right = 5.)
    
lens2 = AsphericLens(name = 'Lens 2', 
                     r1 = 269.190*coeff, 
                     r2 = 6398.02*coeff, 
                     c1 = -2.4029, 
                     c2 = 1770.36,
                     thick = 40*coeff,
                     diameter = 300*coeff, 
                     x = (40.+130.+369.408+10.)*coeff, 
                     y = 0.,
                     AR_left = 5., AR_right = 5.)
    
aperture_stop = ApertureStop(name = 'Aperture Stop',
                             pos_x = 10*coeff,
                             diameter = 200*coeff,
                             thickness = 5*coeff,
                             n_refr = 1., 
                             conductivity = 1e7)
    
image_plane = ImagePlane(name = 'Image Plane',
                         pos_x = (10+714.704)*coeff,
                         diameter = 300*coeff,
                         thickness = 2*coeff,
                         n_refr = 1, 
                         conductivity = 0)

tube = TelescopeTube('Tube')
absorber = Absorber('Absorber')



def system_assembly(lens1, lens2, aperture_stop, image_plane, res, dpml):
    opt_sys = OpticalSystem('test')
    opt_sys.set_size(750*coeff,300*coeff)
    opt_sys.add_component(lens1)
    opt_sys.add_component(lens2)
    opt_sys.add_component(aperture_stop)
    opt_sys.add_component(image_plane)

    opt_sys.assemble_system(dpml = dpml, resolution = res)

    opt_sys.write_h5file(parallel = True)
    
    return opt_sys



#PARAMS
wvl = 10
resolution = 1
dpml = 5

"""
FFT_list = []


opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane, resolution, dpml)
w0_list = [10., 30., 50.] #np.linspace(10,50,5)


legend = ['1', '3', '5']

for k in range(len(w0_list)):
    #RUN SIM
    
    sim = Sim(opt_sys)
    analysis = Analysis(sim)  

    analysis.image_plane_beams(wavelength = wvl, runtime = 800*coeff, 
        sim_resolution = resolution, beam_w0 = w0_list[k], 
        plot_amp = True, plotname = 'w0_{}_wvl1'.format(int(w0_list[k])),
        aperture_size = 200*coeff) 

    #PLOT FIELD
    analysis.sim.plot_efield()

    #freq, fft = analysis.beam_FT(precision_factor = 15)

    #FFT_list.append(fft[0])
"""

#TEST N2FAR
opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane, resolution, dpml)
w0 = 30

w0_list = [10., 30., 50.] #np.linspace(10,50,5)


legend = ['1', '3', '5']

for k in range(len(w0_list)):
    sim = Sim(opt_sys)
    analysis = Analysis(sim) 
    analysis.image_plane_beams(wavelength = wvl, runtime = 800*coeff, sim_resolution = resolution, beam_w0 = w0, plotname = 'w0_{}_wvl1'.format(int(w0_list[k]))) 
    analysis.sim.plot_efield()

#freq, fft = analysis.beam_FT(precision_factor = 15)

#analysis.plotting(freq, fft, wvl, deg_range= 40, print_fwhm = True, savefig = True, path_name = 'hey')

#PLOT BEAM
#analysis.plotting(freq, FFT_list, wvl, deg_range= 40, print_fwhm = True, savefig = True, path_name = 'testing_w0', legend = legend)


"""
w0_list_precise = np.linspace(8,13,11)
legend = ['8', '8.5', '9', '9.5', '10','10.5','11', '11.5', '12', '12.5', '13']
for k in range(11):
    #RUN SIM
    
    sim = Sim(opt_sys)
    analysis = Analysis(sim)  

    analysis.image_plane_beams(wavelength = wvl, sim_resolution = resolution, beam_w0 = w0_list_precise[k]) 

    #PLOT FIELD
    analysis.sim.plot_efield()

    freq, fft = analysis.beam_FT(precision_factor = 15)

    FFT_list.append(fft[0])

#PLOT BEAM
analysis.plotting(freq, FFT_list, wvl, deg_range= 40, print_fwhm = True, savefig = True, path_name = 'testing_w0_precise', legend = legend)
"""


### TEST SINGLE VALUE

"""
w0 = 40


sim = Sim(opt_sys)
analysis = Analysis(sim)  

analysis.image_plane_beams(wavelength = wvl, sim_resolution = resolution, beam_w0 = w0, plot_amp = True, plotname = 'test')  

#PLOT FIELD
analysis.sim.plot_efield()

#freq, fft = analysis.beam_FT(precision_factor = 15)

#FFT_list.append(fft[0])

#analysis.plotting(freq, FFT_list, wvl, deg_range= 40, print_fwhm = True, savefig = True, path_name = 'w0_8dot5')
"""