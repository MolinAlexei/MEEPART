from meep_optics import OpticalSystem, AsphericLens, ApertureStop, ImagePlane, Sim, Analysis
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import h5py
    
lens1 = AsphericLens(name = 'Lens 1', 
                     r1 = 327.365, 
                     r2 = np.inf, 
                     c1 = -0.66067, 
                     c2 = 0, 
                     thick = 40, 
                     x = 130.+10., 
                     y = 0., 
                     AR_left = 0, AR_right = 0,
                     AR_delamination = 0)
    
lens2 = AsphericLens(name = 'Lens 2', 
                     r1 = 269.190, 
                     r2 = 6398.02, 
                     c1 = -2.4029, 
                     c2 = 1770.36,
                     thick = 40, 
                     x = 40.+130.+369.408+10., 
                     y = 0.,
                     AR_left = 0, AR_right = 0,
                     AR_delamination = 0)
    
aperture_stop = ApertureStop(name = 'Aperture Stop',
                             pos_x = 10,
                             diameter = 200,
                             thickness = 5,
                             n_refr = 1.1, 
                             conductivity = 1e7)
    
image_plane = ImagePlane(name = 'Image Plane',
                         pos_x = 10+714.704,
                         diameter = 300,
                         thickness = 2,
                         n_refr = 1.1, 
                         conductivity = 0.01)



def system_assembly(lens1, lens2, aperture_stop, image_plane, res, dpml):
    opt_sys = OpticalSystem('test')
    opt_sys.set_size(800,300)
    opt_sys.add_component(lens1)
    opt_sys.add_component(lens2)
    opt_sys.add_component(aperture_stop)
    opt_sys.add_component(image_plane)
    opt_sys.assemble_system(dpml = dpml, resolution = res)
    opt_sys.write_h5file()
    
    return opt_sys


def main():


    parser = ap.ArgumentParser(description='Pouet.')

    parser.add_argument('--wvl', dest = 'wvl', action = 'store', type = float, default = 2)
    parser.add_argument('--resolution', dest = 'resolution', action = 'store', type = int, default = 7)
    parser.add_argument('--file_name', dest = 'file_name', action = 'store', type = str, default = 'test')
    parser.add_argument('--beam_nb', dest = 'beam_nb', action = 'store', type = int, default = 1)
    parser.add_argument('--plot_FT', dest= 'plot_FT', action = 'store_true', default = False)
    parser.add_argument('--plot_name', dest= 'plot_name', action = 'store', type = str, default = 'FT')
    args = parser.parse_args()

    if not args.plot_FT :
        
        dpml = max(np.int(np.around(args.wvl/2)), 1)

        opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane, res = args.resolution, dpml = dpml)
        sim = Sim(opt_sys)
        analysis = Analysis(sim)  

        analysis.image_plane_beams(wavelength = args.wvl, fwidth = 0, sourcetype='Gaussian beam',
                                    y_max = 100, Nb_sources = args.beam_nb, sim_resolution = args.resolution) 

        freq, fft = analysis.beam_FT(aperture_size = 200, precision_factor = 15)
        degrees = np.arctan(freq*args.wvl)*180/np.pi
        fwhm_fft = analysis.FWHM_fft[0]
        fwhm_ap = analysis.FWHM_ap[0]

        name = args.file_name + '.h5'

        h = h5py.File(name, 'w')
        h.create_dataset('degrees', data=degrees)
        h.create_dataset('ffts', data=fft)
        h.create_dataset('fwhm_fft', data=[fwhm_fft])
        h.create_dataset('fwhm_ap', data=[fwhm_ap])
        h.close()

    elif args.plot_FT :

        name = args.file_name + '.h5'
        data = h5py.File(name, 'r')
        fft = data['ffts']
        degrees = data['degrees']
        fwhm_ap = data['fwhm_ap']
        fwhm_fft = data['fwhm_fft']
        plt.figure(figsize = (8,6))

        

        for k in range(len(fft)):
            fft_dB = 10*np.log10(np.abs(fft[k].real))
            if len(fft) >1 :
                y = k*100/(len(fft)-1)
            elif len(fft) ==1 :
                y = 0

            plt.plot(degrees, fft_dB, label = '{:.2f}mm'.format(y))
        plt.ylim((-60, 0))
        plt.xlabel('Angle [deg]')
        plt.ylabel('Power [dB]')
        if len(fft)>1 :
            plt.legend(loc = 'lower right')
            plt.xlim((-20,20))
        elif len(fft) == 1:
            plt.xlim((0,10))
        fwhm = args.wvl*0.28648
        plt.vlines([-fwhm/2, fwhm/2], -100, 0, color = 'grey', linestyle = 'dashdot')
        plt.vlines([fwhm_fft[0]/2], -100, 0, color='grey', linestyle = '--', alpha = 0.7)
        plt.annotate('Expected FWHM : {:.2f}deg'.format(fwhm), 
            xy = (.1, .9), xycoords='figure fraction', color = 'grey')
        plt.annotate('Beam FWHM : {:.2f}deg'.format(fwhm_fft[0]), 
            xy = (.1, .87), xycoords='figure fraction', color = 'grey', alpha = 0.7)
        #plt.annotate('Field FWHM : {:.2f}mm'.format(fwhm_ap[0]), 
        #    xy = (.1, .84), xycoords='figure fraction')
        plt.tight_layout()
        plt.savefig('plots/{}.png'.format(args.file_name))
        plt.close()

    


if __name__ == '__main__':
    main()

