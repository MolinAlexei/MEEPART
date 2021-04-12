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


def main():


    parser = ap.ArgumentParser(description='Pouet.')


    #REQUIRED FOR EACH CALL
    parser.add_argument('--wvl', dest = 'wvl', action = 'store', type = float, default = 2)
    parser.add_argument('--resolution', dest = 'resolution', action = 'store', type = int, default = 7)
    parser.add_argument('--file_name', dest = 'file_name', action = 'store', type = str, default = 'test')

    #ADDITIONNAL ARGUMENTS FOR ANALYSIS
    parser.add_argument('--beam_nb', dest = 'beam_nb', action = 'store', type = int, default = 1)
    parser.add_argument('--plot_name', dest= 'plot_name', action = 'store', type = str, default = 'FT')
    parser.add_argument('--AR_thick', dest= 'AR_thick', action = 'append', type = float)
    parser.add_argument('--AR_delam', dest= 'AR_delam', action = 'append', type = float)
    parser.add_argument('--lens_idx', dest= 'lens_idx', action = 'append', type = float)
    parser.add_argument('--bubbles_r', dest= 'bubbles_r', action = 'append', type = float)
    parser.add_argument('--bubbles_nb', dest= 'bubbles_nb', action = 'append', type = float)
    parser.add_argument('--axial_grad', dest= 'axial_grad', action = 'append', type = float)
    parser.add_argument('--radial_grad', dest= 'radial_grad', action = 'append', type = float)

    #ENABLES A SPECFIFIC TYPE OF ANALYSIS
    parser.add_argument('--plot_FT', dest= 'plot_FT', action = 'store_true', default = False)
    parser.add_argument('--AR_analysis', dest= 'AR_analysis', action = 'store_true', default = False)
    parser.add_argument('--wvl_analysis', dest= 'wvl_analysis', action = 'store_true', default = False)
    parser.add_argument('--index_anal', dest= 'index_anal', action = 'store_true', default = False)
    parser.add_argument('--bubbles', dest= 'bubbles', action = 'store_true', default = False)
    parser.add_argument('--gradient', dest= 'gradient', action = 'store_true', default = False)
    parser.add_argument('--thermal_def', dest= 'thermal_def', action = 'store_true', default = False)
    parser.add_argument('--tube', dest= 'tube', action = 'store_true', default = False)

    args = parser.parse_args()

    if args.wvl_analysis :
        
        dpml = max(int(np.around(args.wvl/2)), 1)


        opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane, res = args.resolution, dpml = dpml)
        sim = Sim(opt_sys)
        analysis = Analysis(sim)  

        analysis.image_plane_beams(wavelength = args.wvl, fwidth = 0, sourcetype='Gaussian beam',
                                    y_max = 100, Nb_sources = args.beam_nb, sim_resolution = args.resolution) 

        analysis.sim.plot_efield()

        freq, fft = analysis.beam_FT(precision_factor = 15)
        degrees = np.arctan(freq*args.wvl)*180/np.pi
        fwhm_fft = analysis.FWHM_fft[0]
        fwhm_ap = analysis.FWHM_ap[0]

        name = 'FFT_data/' + args.file_name + '.h5'

        h = h5py.File(name, 'w')#, driver ='mpio', comm=MPI.COMM_WORLD)
        h.create_dataset('degrees', data=degrees)
        h.create_dataset('ffts', data=fft)
        h.create_dataset('fwhm_fft', data=[fwhm_fft])
        h.create_dataset('fwhm_ap', data=[fwhm_ap])
        h.close()

    if args.tube :
        dpml = max(int(np.around(args.wvl/2)), 1)
        fft = []

        is_tube = False
        is_abs = False
        for k in range(2):

            if k==1 :
                is_abs = False

            opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane, 
                res = args.resolution, dpml = dpml, telescope_tube = is_tube, absorb = is_abs)

            sim = Sim(opt_sys)
            analysis = Analysis(sim)  
            analysis.image_plane_beams(wavelength = args.wvl, fwidth = 0, sourcetype='Gaussian beam',
                                    y_max = 100, Nb_sources = args.beam_nb, sim_resolution = args.resolution) 

            freq, fft_k = analysis.beam_FT(precision_factor = 15)
            fft.append(fft_k[1].real)

            analysis.sim.plot_efield()

        fft = np.array(fft)
        degrees = np.arctan(freq*args.wvl)*180/np.pi
        fwhm_fft = analysis.FWHM_fft[0]
        fwhm_ap = analysis.FWHM_ap[0]

        name = 'FFT_data/' + args.file_name + '.h5'

        h = h5py.File(name, 'w', driver ='mpio', comm=MPI.COMM_WORLD)
        h.create_dataset('degrees', data=degrees)
        h.create_dataset('ffts', data=fft)
        h.create_dataset('fwhm_fft', data=[fwhm_fft])
        h.create_dataset('fwhm_ap', data=[fwhm_ap])
        h.close()

    if args.thermal_def :
        dpml = max(int(np.around(args.wvl/2)), 1)
        fft = []
        beam_solid_angle = []

        for k in range(2):
            
            if k == 1 :
                lens1.therm_def = True

            opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane, res = args.resolution, dpml = dpml)
            #opt_sys.plot_lenses()
            sim = Sim(opt_sys)
            analysis = Analysis(sim)  
            analysis.image_plane_beams(wavelength = args.wvl, fwidth = 0, sourcetype='Gaussian beam',
                                    y_max = 0, Nb_sources = args.beam_nb, sim_resolution = args.resolution) 

            #sim.plot_efield()

            freq, fft_k = analysis.beam_FT(aperture_size = 200, precision_factor = 15)
            fft.append(fft_k[0])

            degrees = np.arctan(freq*args.wvl)*180/np.pi

        fft = np.array(fft)
        
        fwhm_fft = analysis.FWHM_fft[0]
        fwhm_ap = analysis.FWHM_ap[0]

        name = 'FFT_data/' + args.file_name + '.h5'

        h = h5py.File(name, 'w', driver ='mpio', comm=MPI.COMM_WORLD)
        h.create_dataset('degrees', data=degrees)
        h.create_dataset('ffts', data=fft)
        h.create_dataset('fwhm_fft', data=[fwhm_fft])
        h.create_dataset('fwhm_ap', data=[fwhm_ap])
        h.close()

    if args.bubbles :
        dpml = max(int(np.around(args.wvl/2)), 1)
        fft = []
        beam_solid_angle = []

        for k in range(len(args.bubbles_r)):
            bub_radius = np.float(args.bubbles_r[0])
            for i in range(len(args.bubbles_nb)):

            
                bub_nb = int(args.bubbles_nb[i])
            
                if k == 1 :
                    opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane, res = args.resolution, dpml = dpml,
                                            bub_radius = bub_radius, bub_nb = bub_nb)
                elif k != 1 :
                    r_factor = args.bubbles_r[k]/args.bubbles_r[0]
                    opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane, res = args.resolution, dpml = dpml,
                                            bub_radius = bub_radius, bub_nb = bub_nb, r_factor = r_factor)
                #opt_sys.plot_lenses()
                sim = Sim(opt_sys)
                analysis = Analysis(sim)  
                analysis.image_plane_beams(wavelength = args.wvl, fwidth = 0, sourcetype='Gaussian beam',
                                    y_max = 0, Nb_sources = args.beam_nb, sim_resolution = args.resolution) 

                #sim.plot_efield()

                freq, fft_k = analysis.beam_FT(aperture_size = 200, precision_factor = 15)
                fft.append(fft_k[0])

                degrees = np.arctan(freq*args.wvl)*180/np.pi

        fft = np.array(fft)
        
        fwhm_fft = analysis.FWHM_fft[0]
        fwhm_ap = analysis.FWHM_ap[0]

        name = 'FFT_data/' + args.file_name + '.h5'

        h = h5py.File(name, 'w', driver ='mpio', comm=MPI.COMM_WORLD)
        h.create_dataset('degrees', data=degrees)
        h.create_dataset('ffts', data=fft)
        h.create_dataset('fwhm_fft', data=[fwhm_fft])
        h.create_dataset('fwhm_ap', data=[fwhm_ap])
        h.close()

    if args.gradient: 
        dpml = max(int(np.around(args.wvl/2)), 1)
        fft = []

        for k in range(len(args.radial_grad)):

            radial_grad = np.float(args.radial_grad[k])
            lens1.radial_slope = radial_grad
            lens2.radial_slope = radial_grad


            for i in range(len(args.axial_grad)):
                axial_grad = np.float(args.axial_grad[i])
                lens1.axial_slope = axial_grad
                lens2.axial_slope = axial_grad

                opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane, res = args.resolution, dpml = dpml)
                sim = Sim(opt_sys)
                analysis = Analysis(sim)  
                analysis.image_plane_beams(wavelength = args.wvl, fwidth = 0, sourcetype='Gaussian beam',
                                    y_max = 100, Nb_sources = args.beam_nb, sim_resolution = args.resolution) 

                freq, fft_k = analysis.beam_FT(aperture_size = 200, precision_factor = 15)
                fft.append(fft_k[0])

                degrees = np.arctan(freq*args.wvl)*180/np.pi

        fft = np.array(fft)
        
        fwhm_fft = analysis.FWHM_fft[0]
        fwhm_ap = analysis.FWHM_ap[0]

        name = 'FFT_data/' + args.file_name + '.h5'

        h = h5py.File(name, 'w', driver ='mpio', comm=MPI.COMM_WORLD)
        h.create_dataset('degrees', data=degrees)
        h.create_dataset('ffts', data=fft)
        h.create_dataset('fwhm_fft', data=[fwhm_fft])
        h.create_dataset('fwhm_ap', data=[fwhm_ap])
        h.close()

    if args.AR_analysis : 
        dpml = max(int(np.around(args.wvl/2)), 1)
        fft = []

        for k in range(len(args.AR_thick)):

            AR = np.float(args.AR_thick[k])
            lens1.AR_right = AR
            lens1.AR_left = AR
            lens2.AR_right = AR
            lens2.AR_left = AR

            for i in range(len(args.AR_delam)):
                d = np.float(args.AR_delam[i])
                lens1.delam_thick = 0
                lens1.delam_width = d
                lens2.delam_thick = 0
                lens2.delam_width = d 
            
                opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane, res = args.resolution, dpml = dpml)
                sim = Sim(opt_sys)
                analysis = Analysis(sim)  
                analysis.image_plane_beams(wavelength = args.wvl, fwidth = 0, sourcetype='Gaussian beam',
                                    y_max = 100, Nb_sources = args.beam_nb, sim_resolution = args.resolution) 

                freq, fft_k = analysis.beam_FT(aperture_size = 200, precision_factor = 15)
                fft.append(fft_k[0])

                degrees = np.arctan(freq*args.wvl)*180/np.pi

        fft = np.array(fft)
        
        fwhm_fft = analysis.FWHM_fft[0]
        fwhm_ap = analysis.FWHM_ap[0]

        name = 'FFT_data/' + args.file_name + '.h5'

        h = h5py.File(name, 'w', driver ='mpio', comm=MPI.COMM_WORLD)
        h.create_dataset('degrees', data=degrees)
        h.create_dataset('ffts', data=fft)
        h.create_dataset('fwhm_fft', data=[fwhm_fft])
        h.create_dataset('fwhm_ap', data=[fwhm_ap])
        h.close()

    if args.index_anal :
        dpml = max(int(np.around(args.wvl/2)), 1)
        fft = []

        for k in range(len(args.lens_idx)):

            idx = np.float(args.lens_idx[k])
            lens1.material = idx**2
            lens2.material = idx**2

            opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane, res = args.resolution, dpml = dpml)
            sim = Sim(opt_sys)
            analysis = Analysis(sim)  
            analysis.image_plane_beams(wavelength = args.wvl, fwidth = 0, sourcetype='Gaussian beam',
                                    y_max = 100, Nb_sources = args.beam_nb, sim_resolution = args.resolution) 

            freq, fft_k = analysis.beam_FT(aperture_size = 200, precision_factor = 15)
            fft.append(fft_k[0].real)

        fft = np.array(fft)
        degrees = np.arctan(freq*args.wvl)*180/np.pi
        fwhm_fft = analysis.FWHM_fft[0]
        fwhm_ap = analysis.FWHM_ap[0]

        name = 'FFT_data/' + args.file_name + '.h5'

        h = h5py.File(name, 'w', driver ='mpio', comm=MPI.COMM_WORLD)
        h.create_dataset('degrees', data=degrees)
        h.create_dataset('ffts', data=fft)
        h.create_dataset('fwhm_fft', data=[fwhm_fft])
        h.create_dataset('fwhm_ap', data=[fwhm_ap])
        h.close()


    elif args.plot_FT:

        name = 'FFT_data/' + args.file_name + '.h5'
        data = h5py.File(name, 'r')
        fft = data['ffts']
        degrees = data['degrees']
        fwhm_ap = data['fwhm_ap']
        fwhm_fft = data['fwhm_fft']

        deg = np.copy(degrees)
        plt.figure(figsize = (8,6))

        
        legend = ['0mm', '0.5mm', '1.25mm','2.5mm']
        legend = ['No delamination', '0.5mm thick lumps']
        legend = ['1e-3 $mm^{-1}$','5e-4 $mm^{-1}$','1e-4 $mm^{-1}$','0 $mm^{-1}$']
        legend = ['1mm', '0.5mm', 'No bubbles']
        legend = ['No defects', 'Therm. deform.']
        legend = ['+2 $\%$ change', '0 $\%$ change', '-2 $\%$ change']
        legend = ['Without absorber', 'With absorber']
        
        
        def gaussian(x, stddev, mean):
            return np.exp(-(((x-mean)/4/stddev)**2))
        for k in range(len(fft)):

            i = 0
            while degrees[i]<1 : 
                i+=1

            middle = int((len(fft[k])+1)/2)

            fft_k = fft[k]

            

            

            fft_dB = 10*np.log10(np.abs(fft[k]))
            if len(fft) >1 :
                y = k*100/(len(fft)-1)
            elif len(fft) ==1 :
                y = 0

            rads = np.array(degrees) * np.pi/180
            integrand = fft_k* np.sin(rads)
            integrand = np.append(integrand, integrand[0])
            rads = np.append(rads, 0)
            right_part = np.trapz(integrand[0:middle], x = rads[0:middle])
            left_part = np.trapz(integrand[middle:-0], x = rads[middle:-0])
            solid_angle = right_part + left_part
            print('{:.2e}'.format(solid_angle*2*np.pi))
            


            plt.plot(deg, fft_dB, label = '{}'.format(legend[k]))

        plt.ylim((-60, 0))
        plt.xlabel('Angle [deg]', fontsize = 14)
        plt.ylabel('Power [dB]', fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)

        plt.xlim((0,45))

        if len(fft)>1 :
            plt.legend(loc = 'upper right', fontsize = 12)


        #elif len(fft) == 1:
        #    plt.xlim((0,10))

        #plt.legend(('+2 $\%$ change', '0 $\%$ change', '-2 $\%$ change'))
        #plt.legend(('1mm', '0.5mm', '0.25mm'))

        
        fwhm = args.wvl*0.28648

        plt.hlines(-3,0,90)
        plt.vlines([-fwhm/2, fwhm/2], -100, 0, color = 'grey', linestyle = 'dashdot')
        #plt.vlines([fwhm_fft[0]/2], -100, 0, color='grey', linestyle = '--', alpha = 0.7)
        #plt.annotate('Expected FWHM : {:.2f}deg'.format(fwhm), 
        #    xy = (.25, .9), xycoords='figure fraction', color = 'grey')
        #plt.annotate('Beam FWHM : {:.2f}deg'.format(fwhm_fft[0]), 
        #    xy = (.25, .87), xycoords='figure fraction', color = 'grey', alpha = 0.7)
        

        #plt.annotate('Field FWHM : {:.2f}mm'.format(fwhm_ap[0]), 
        #    xy = (.1, .84), xycoords='figure fraction')
        plt.tight_layout()
        plt.savefig('plots/{}.png'.format(args.file_name))
        plt.close()

        """
        color = ['blue', 'red', 'black']
        plt.figure()
        for k in range(len(fft)):
            fft_k = fft[k]
            deg = np.array(degrees)*1.
            popt, _ = optimize.curve_fit(gaussian, deg, np.abs(fft_k))
            print(2*popt[1] + 2*4*popt[0]*np.sqrt(np.log(2)))
            plt.plot(degrees, fft_k, color = color[k])
            plt.plot(degrees, gaussian(deg, popt[0], popt[1]), color = color[k], linestyle = '--')
            plt.xlim((-2,2))
        plt.savefig('pouetpouet')
        plt.close()
        """

if __name__ == '__main__':
    main()

