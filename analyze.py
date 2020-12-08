from meep_optics import OpticalSystem, AsphericLens, ApertureStop, ImagePlane, Sim, Analysis
import numpy as np
import matplotlib.pyplot as plt
    
lens1 = AsphericLens(name = 'Lens 1', 
                     r1 = 327.365, 
                     r2 = np.inf, 
                     c1 = -0.66067, 
                     c2 = 0, 
                     thick = 40, 
                     x = 130.+50., 
                     y = 0., 
                     AR_left = 5, AR_right = 5,
                     AR_delamination = .14)
    
lens2 = AsphericLens(name = 'Lens 2', 
                     r1 = 269.190, 
                     r2 = 6398.02, 
                     c1 = -2.4029, 
                     c2 = 1770.36,
                     thick = 40, 
                     x = 40.+130.+369.408+50., 
                     y = 0.,
                     AR_left = 5, AR_right = 5,
                     AR_delamination = .14)
    
aperture_stop = ApertureStop(name = 'Aperture Stop',
                             pos_x = 50,
                             diameter = 200,
                             thickness = 5,
                             index = 5., 
                             conductivity = 1e7)
    
image_plane = ImagePlane(name = 'Image Plane',
                         pos_x = 50+714.704,
                         diameter = 300,
                         thickness = 2,
                         index = 5., 
                         conductivity = 0.01)


study_freq = 0.5
dpml = np.int(np.around(0.5*1/study_freq))

def system_assembly(lens1, lens2, aperture_stop, image_plane):
    opt_sys = OpticalSystem('test')
    opt_sys.set_size(800,300)
    opt_sys.add_component(lens1)
    opt_sys.add_component(lens2)
    opt_sys.add_component(aperture_stop)
    opt_sys.add_component(image_plane)
    #print(opt_sys.list_components())
    
    
    opt_sys.assemble_system(dpml = dpml, resolution = 7)
    #opt_sys.plot_lenses()
    opt_sys.write_h5file()
    
    return opt_sys

opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane)    
    
sim = Sim(opt_sys)
    
analysis = Analysis(sim)
plt.figure()
analysis.image_plane_beams(study_freq, fwidth = 0.1, sourcetype='Gaussian beam multichromatic',
                           y_max = 100, Nb_sources = 2, linestyle='-', sim_resolution=7)
beams_delam = analysis.list_beams
freq, fft_delam = analysis.beam_FT()
plt.legend(fontsize = 9)

# lens1.delaminate = 0
# lens2.delaminate = 0
# opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane)   
# sim = Sim(opt_sys)
# analysis = Analysis(sim)
# analysis.image_plane_beams(study_freq, fwidth = 0.1, sourcetype='Gaussian beam multichromatic',
#                            y_max = 100, Nb_sources = 2, linestyle = '--', sim_resolution=7) 
# beams_no_delam = analysis.list_beams
# freq, fft_no_delam = analysis.beam_FT()

plt.legend(('Center beam w/ AR delam.', 'Border beam w/ AR delam.', 'Center beam w/o AR delam.',
            'Border beam w/o AR delam.'), fontsize= 9)
#plt.savefig('../delamination_multichromatic.pdf')  
plt.show()

fft_dB_onaxis = 10*np.log10(fft_delam[0].real**2)
fft_dB_offaxis = 10*np.log10(fft_delam[1].real**2)

plt.figure()
plt.plot(freq*360, fft_dB_onaxis, c='r', linestyle = '-')
plt.plot(freq*360, fft_dB_offaxis, c ='b', linestyle = '-')
# plt.semilogy(freq*200*360, fft_no_delam[0].real**2, c ='r', linestyle = '--')
# plt.semilogy(freq*200*360, fft_no_delam[1].real**2, c= 'b', linestyle = '--')
plt.legend(('FFT On axis w/ delamination', 'FFT Off-axis w/ delamination',
            'FFT On axis w/o delamination', 'FFT Off-axis w/o delamination'))
plt.ylim((-100, 1))
plt.show()
