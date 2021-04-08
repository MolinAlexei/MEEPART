import numpy as np
from meep_optics3D import OpticalSystem, AsphericLens, ApertureStop, ImagePlane, TelescopeTube, Sim, Analysis
import matplotlib.pyplot as plt
import argparse as ap
import h5py
from mpi4py import MPI
from scipy.integrate import simps
    
lens1 = AsphericLens(name = 'Lens 1', 
                    r1 = 327.365, 
                    r2 = np.inf, 
                    c1 = -0.66067, 
                    c2 = 0, 
                    thick = 40, 
                    x = 10.+130.,
                    diameter = 300,
                    AR_left = 0, AR_right = 0,
                    delam_thick = 0,
                    delam_width = 00,
                    axial_slope = 0,
                    radial_slope = 0)
    
lens2 = AsphericLens(name = 'Lens 2', 
                    r1 = 269.190, 
                    r2 = 6398.02, 
                    c1 = -2.4029, 
                    c2 = 1770.36, 
                    thick = 40., 
                    x = 40.+130.+369.408+10.,
                    diameter = 300,
                    AR_left = 0, AR_right = 0,
                    delam_thick = 0,
                    delam_width = 00,
                    axial_slope = 0, 
                    radial_slope = 0)

aperture_stop = ApertureStop(name = 'Aperture Stop',
                            pos_x = 10,
                            diameter = 200,
                            thickness = 5,
                            n_refr = 1.1, 
                            conductivity = 1e7)
    

    
image_plane = ImagePlane(name = 'Image Plane',
                        pos_x = 10+714.704,
                        side_size = 300,
                        thickness = 2,
                        n_refr = 1., 
                        conductivity = 0)

tube = TelescopeTube(name = 'Tube', 
                        diameter = 300,
                        dielec_perm = 1.7,
                        metal_cond = 1e7)



def system_assembly(lens1, lens2, aperture_stop, image_plane, res, dpml, 
                    bub_radius = 0, bub_nb = 4, r_factor = 1):
    
    opt_sys = OpticalSystem('test')
    opt_sys.set_size(800,300,300)
    opt_sys.add_component(lens1)
    opt_sys.add_component(lens2)
    opt_sys.add_component(aperture_stop)
    opt_sys.add_component(image_plane)
    #opt_sys.add_component(tube)
    if res < 1 :
        res = 1
    else :
        res = int(res)

    opt_sys.assemble_system(dpml = dpml, resolution = res)
    #if bub_radius>0 and bub_nb >0 and r_factor >0:
    #    opt_sys.make_lens_bubbles(bub_radius, bub_nb, 15)
    opt_sys.write_h5file()
    print('WritingH5Complete')
    return opt_sys


def main():


    parser = ap.ArgumentParser(description='Pouet.')


    #REQUIRED FOR EACH CALL
    parser.add_argument('--wvl', dest = 'wvl', action = 'store', type = float, default = 16)
    parser.add_argument('--resolution', dest = 'resolution', action = 'store', type = float, default = 0.5)
    parser.add_argument('--file_name', dest = 'file_name', action = 'store', type = str, default = 'test3D')

    #ADDITIONNAL ARGUMENTS FOR ANALYSIS
    parser.add_argument('--beam_nb', dest = 'beam_nb', action = 'store', type = int, default = 1)
    parser.add_argument('--plot_name', dest= 'plot_name', action = 'store', type = str, default = 'FT')
    parser.add_argument('--AR_thick', dest= 'AR_thick', action = 'append', type = float)
    parser.add_argument('--AR_delam', dest= 'AR_delam', action = 'append', type = float)
    parser.add_argument('--lens_idx', dest= 'lens_idx', action = 'append', type = float)
    parser.add_argument('--bubbles_r', dest= 'bubbles_r', action = 'append', type = float)
    parser.add_argument('--bubbles_nb', dest= 'bubbles_nb', action = 'append', type = float)
    parser.add_argument('--axial_grad', dest= 'axial_grad', action = 'append', type = float)
    parser.add_argument('--plot_nbs', dest= 'plot_nbs', action = 'append', type = float)
    parser.add_argument('--radial_grad', dest= 'radial_grad', action = 'append', type = float)

    #ENABLES A SPECFIFIC TYPE OF ANALYSIS
    parser.add_argument('--plot_beam', dest= 'plot_beam', action = 'store_true', default = False)
    parser.add_argument('--plot_difference', dest= 'plot_difference', action = 'store_true', default = False)
    parser.add_argument('--runsim', dest= 'runsim', action = 'store_true', default = False)
    parser.add_argument('--delam_analysis', dest= 'delam_analysis', action = 'store_true', default = False)
    parser.add_argument('--gradient', dest= 'gradient', action = 'store_true', default = False)
    parser.add_argument('--bubbles', dest= 'bubbles', action = 'store_true', default = False)


    args = parser.parse_args()

    if args.runsim :
        
        dpml = max(int(np.around(args.wvl/2)), 1)


        opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane, res = args.resolution, dpml = dpml)
        sim = Sim(opt_sys)
        analysis = Analysis(sim)  

        analysis.image_plane_beams(wavelength = args.wvl, fwidth = 0, sourcetype='Gaussian beam',
                                    y_max = 100, Nb_sources = args.beam_nb, sim_resolution = args.resolution) 

        analysis.sim.plot_system()
        freq, fft = analysis.beam_FT(aperture_size = 200, precision_factor = 15)
        degrees = np.arctan(freq*args.wvl)*180/np.pi
        opt_sys.delete_h5file()
        name = '3DFFT_data/' + args.file_name + '.h5'

        h = h5py.File(name, 'w', driver ='mpio', comm=MPI.COMM_WORLD)
        h.create_dataset('degrees', data=degrees)
        h.create_dataset('ffts', data=fft)
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
                opt_sys.delete_h5file() 
                degrees = np.arctan(freq*args.wvl)*180/np.pi

        fft = np.array(fft)
        
        name = '3DFFT_data/' + args.file_name + '.h5'

        h = h5py.File(name, 'w', driver ='mpio', comm=MPI.COMM_WORLD)
        h.create_dataset('degrees', data=degrees)
        h.create_dataset('ffts', data=fft)
        h.close()

    if args.delam_analysis : 
        dpml = max(int(np.around(args.wvl/2)), 1)
        fft = []
        ar_thick = np.float(args.AR_thick[0])
        lens1.AR_left = ar_thick
        lens1.AR_right = ar_thick
        lens2.AR_left = ar_thick
        lens2.AR_right = ar_thick

        for i in range(len(args.AR_delam)):
            d = np.float(args.AR_delam[i])
            lens1.delam_thick = d 
            lens1.delam_width = 10
            lens2.delam_thick = d 
            lens2.delam_width = 10
            
            opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane, res = args.resolution, dpml = dpml)
            sim = Sim(opt_sys)
            analysis = Analysis(sim)  
            analysis.image_plane_beams(wavelength = args.wvl, fwidth = 0, sourcetype='Gaussian beam',
                                    y_max = 100, Nb_sources = args.beam_nb, sim_resolution = args.resolution) 

            freq, fft_k = analysis.beam_FT(aperture_size = 200, precision_factor = 11)
            fft.append(fft_k[0])
            opt_sys.delete_h5file() 
            degrees = np.arctan(freq*args.wvl)*180/np.pi
        fft = np.array(fft)
        
        fwhm_fft = analysis.FWHM_fft[0]
        fwhm_ap = analysis.FWHM_ap[0]

        name = '3DFFT_data/' + args.file_name + '.h5'

        h = h5py.File(name, 'w', driver ='mpio', comm=MPI.COMM_WORLD)
        h.create_dataset('degrees', data=degrees)
        h.create_dataset('ffts', data=fft)
        h.close()

    if args.bubbles :
        dpml = max(int(np.around(args.wvl/2)), 1)
        fft = []
        beam_solid_angle = []

        for k in range(len(args.bubbles_r)):
            bub_radius = np.float(args.bubbles_r[0])
            for i in range(len(args.bubbles_nb)):

            
                bub_nb = int(args.bubbles_nb[i])
            
                opt_sys = system_assembly(lens1, lens2, aperture_stop, image_plane, res = args.resolution, dpml = dpml,
                                            bub_radius = bub_radius, bub_nb = bub_nb)
                #opt_sys.plot_lenses()
                sim = Sim(opt_sys)
                analysis = Analysis(sim)  
                analysis.image_plane_beams(wavelength = args.wvl, fwidth = 0, sourcetype='Gaussian beam',
                                    y_max = 0, Nb_sources = args.beam_nb, sim_resolution = args.resolution) 

                #sim.plot_efield()

                freq, fft_k = analysis.beam_FT(aperture_size = 200, precision_factor = 15)
                fft.append(fft_k[0])
                opt_sys.delete_h5file() 
                degrees = np.arctan(freq*args.wvl)*180/np.pi

        fft = np.array(fft)
        

        name = '3DFFT_data/' + args.file_name + '.h5'

        h = h5py.File(name, 'w', driver ='mpio', comm=MPI.COMM_WORLD)
        h.create_dataset('degrees', data=degrees)
        h.create_dataset('ffts', data=fft)

        h.close()


    elif args.plot_beam :

        name = '3DFFT_data/' + args.file_name + '.h5'
        data = h5py.File(name, 'r')
        fft = data['ffts']
        degrees = data['degrees']

        deg = np.array(degrees)
        #deg = np.append(deg, 0)
        #X = [deg for k in range(len(deg))]
        #Y = [np.ones(len(deg))*deg[k] for k in range(len(deg))]

        azimuth_rad = deg*np.pi/180
        azimuth_deg = deg

        theta_rad = deg*np.pi/180

        elevation_rad = np.arctan(np.tan(theta_rad)/np.sqrt(1+(np.tan(azimuth_rad)**2)))


        AZ_mat = np.array([azimuth_deg for k in range(len(azimuth_deg))])
        EL_mat = np.array([[np.arctan(np.tan(theta_rad[k])/np.sqrt(1+(np.tan(azimuth_rad[i])**2)))*180/np.pi for i in range(len(deg))] for k in range(len(deg))])
        #EL_mat = np.array([np.ones(len(azimuth_deg))*azimuth_deg[k] for k in range(len(azimuth_deg))])

        for k in range(len(fft)):

            fft_dB = 10*np.log10(np.abs(fft[k]))

            middle_idx = int(len(deg)/2)
            deg_reordered = np.sort(degrees)


            first_integ = [simps(np.abs(fft[k][:,i]), deg_reordered) for i in range(len(fft_dB))]
            solid_angle = simps(first_integ, deg_reordered)*(np.pi/180)**2

            """
            solid_angle = deg_reordered[-1] - deg_reordered[0]
            plt.figure()
            plt.plot(np.arange(len(deg_reordered)), deg_reordered)
            plt.savefig('pouet')
            plt.close()
            """


            fft_dB = 10*np.log10(np.abs(fft[0]))

            fft_dB2 = np.concatenate((fft_dB[middle_idx:,:], fft_dB[:middle_idx,:]), axis = 0)
            fft_reshaped = np.concatenate((fft_dB2[:, middle_idx:], fft_dB2[:, :middle_idx]), axis = 1)

            plt.figure(figsize=(10,10))
            ax = plt.subplot(111)
            im = ax.imshow(fft_reshaped, origin = 'lower', extent = (-90,90,-90,90), vmin = -40, vmax = 0)
            ax.set_xlabel('Angle [deg]')
            ax.set_ylabel('Angle [deg]')
            ax.set_aspect('equal')
            plt.annotate('solid Angle : {:.4e}srad'.format(solid_angle), xy = (.25, .9), xycoords='figure fraction')

            #pouet = np.hstack((fft_dB, np.transpose([fft_dB[:,0]])))
            #fft_test = np.vstack((pouet, [pouet[0,:]]))

            #c = ax.contourf(AZ_mat,EL_mat,fft_reshaped, cmap=plt.get_cmap('viridis'),levels=np.arange(-30,1)*2, vmin = -60, vmax = 0)

            cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
            cbar.ax.set_xlabel('Power [dB]')
            ax.set_xlim((-20,20))
            ax.set_ylim((-20,20))

            if k == 0 :
                circle = plt.Circle((0,0), args.wvl*180/(2*200*np.pi), fill = False, color = 'r', linestyle = '--')
                ax.add_artist(circle)

            plot_name = args.file_name + '_n{}'.format(k)
            plt.tight_layout()
            plt.savefig('3D_plots/{}.png'.format(plot_name))
            plt.close()

    elif args.plot_difference :

        name = '3DFFT_data/' + args.file_name + '.h5'
        data = h5py.File(name, 'r')
        fft = data['ffts']
        degrees = data['degrees']

        deg = np.array(degrees)
        deg = np.append(deg, 0)
        X = [deg for k in range(len(deg))]
        Y = [np.ones(len(deg))*deg[k] for k in range(len(deg))]

        idx0 = 0#np.int(args.plots_nbs[0])
        idx1 = 1#np.int(args.plots_nbs[1])

        fft_dB = 10*np.log10(np.abs(fft[idx0])) - 10*np.log10(np.abs(fft[idx1]))

        plt.figure(figsize=(10,10))
        ax = plt.subplot(111)

        pouet = np.hstack((fft_dB, np.transpose([fft_dB[:,0]])))
        fft_test = np.vstack((pouet, [pouet[0,:]]))

        middle_idx = int(len(fft_dB)/2)
        fft_dB2 = np.concatenate((fft_dB[middle_idx:,:], fft_dB[:middle_idx,:]), axis = 0)
        fft_reshaped = np.concatenate((fft_dB2[:, middle_idx:], fft_dB2[:, :middle_idx]), axis = 1)

        im = ax.imshow(fft_reshaped, vmin = -1, vmax = 1)
        #c = ax.contourf(X,Y,fft_test, cmap=plt.get_cmap('viridis'))#, vmin = -10, vmax = 10)

        ax.set_xlabel('Angle [deg]')
        ax.set_ylabel('Angle [deg]')
        ax.set_title('Difference')

        cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
        #cbar.ax.set_xlabel('Power [dB]')

        #ax.set_xlim((-15,15))
        #ax.set_ylim((-15,15))

        plot_name = args.file_name + 'diff'
        plt.tight_layout()
        plt.savefig('3D_plots/{}.png'.format(plot_name))
        plt.close()

    


if __name__ == '__main__':
    main()

