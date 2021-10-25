#from meep_optics import OpticalSystem, AsphericLens, ApertureStop, ImagePlane, TelescopeTube, Absorber, Sim, Analysis
import meep_optics as mpo
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import h5py
from scipy import optimize
#from mpi4py import MPI



def system_assembly(lens1, lens2, lens3, aperture_stop, stop1, stop2, stop3,
    image_plane, tube, absorber,
    res, dpml):

    opt_sys = mpo.OpticalSystem('test')
    opt_sys.set_size(1200, 420)
    #opt_sys.set_size(1500, 700)
    #opt_sys.set_size(3000, 1500)

    opt_sys.add_component(lens1)
    opt_sys.add_component(lens2)
    opt_sys.add_component(lens3)
    opt_sys.add_component(aperture_stop)
    opt_sys.add_component(stop1)
    opt_sys.add_component(stop2)
    opt_sys.add_component(stop3)
    opt_sys.add_component(image_plane)
    opt_sys.add_component(tube)
    opt_sys.add_component(absorber)
    opt_sys.assemble_system(dpml = dpml, res = res)
    opt_sys.write_h5file()
    
    return opt_sys

def main():

    do_sim = True
    do_analysis = False

    #PARAMS
    wvl = 10.
    resolution = 8
    dpml = 5
    runtime = 2000
    linear = False

    # wvl = 50
    # resolution = 2
    # dpml = 5
    # runtime = 2000
    # #runtime = 200

    rot_angle = 3.

    lens1 = mpo.AsphericLens(name = 'Lens 1',
                        n_refr = 3.38,
                        r1 = np.inf,
                        r2 = -1510.,
                        c1 = 0.,
                        c2 = 12.097,
                        thick = 18.680,
                        AR_left=wvl/4.,
                        AR_right=wvl/4.,
                        x = (130.),
                        y = 0.,
                        diameter = 2 * 189)

    lens2 = mpo.AsphericLens(name = 'Lens 2',
                        n_refr = 3.38,
                        r1 = np.inf,
                        r2 = -871.246,
                        c1 = 0.,
                        c2 = -10.201,
                        thick = 25.825,
                        AR_left=wvl/4.,
                        AR_right=wvl/4.,
                        diameter = 2 * 168,
                        x = (130. + 380.747),
                        y = 0.)

    lens3 = mpo.AsphericLens(name = 'Lens 3',
                        n_refr = 3.38,
                        r1 = 604.446,
                        r2 = 6608.297,
                        c1 = -3.177,
                        c2 = 414.143,
                        thick = 31.552,
                        AR_left=wvl/4.,
                        AR_right=wvl/4.,
                        diameter = 2 * 179,
                        x = (130. + 380.747 + 144.081 + 240.),
                        y = 0.)

    aperture_x = (130. + 380.747 + 144.081)
    aperture_stop = mpo.ApertureStop(name = 'Aperture Stop',
                                 pos_x = aperture_x,
                                 diameter = 2 * 68,
                                 thickness = 5,
                                 n_refr = 1.,
                                 conductivity = 1e7)

    image_plane = mpo.ImagePlane(name = 'Image Plane',
                             pos_x = (130. + 380.747 + 144.081 + 240. + 216.918),
                             diameter = 300,
                             thickness = 2,
                             n_refr = 1,
                             conductivity = 0)

    #tube = mpo.TelescopeTube2('Tube', 1200., 190.)




    rring = 189.2
    stop1 = mpo.ReflectingStop(aperture_x + 80., rring - 70., rring, name='Stop1')
    stop2 = mpo.ReflectingStop(aperture_x + 160, rring - 50., rring, name='Stop2')
    stop3 = mpo.ReflectingStop(aperture_x + 210., rring - 30., rring, name='Stop3')

    tube = mpo.TelescopeTube2(1210., 193., name='Tube')
    absorber = mpo.Absorber2(3.9, 1200., 191., name='Tube')
    #absorber = mpo.Absorber('Absorber', center=0.)


    print('asdf')

    FFT_list = []
    opt_sys = system_assembly(lens1, lens2, lens3, aperture_stop,
        stop1, stop2, stop3, image_plane, tube, absorber, resolution, dpml)

    dist_unit = 0.001
    opt_sys.sys_info(dist_unit, wvl=wvl)

    sim = mpo.Sim(opt_sys)

    # sim.define_source(wvl = wvl, x = 20, y = 0, size_x = 0, size_y = 300,
    #     sourcetype = 'Plane wave', rot_angle = rot_angle)

    sim.define_source(wvl = wvl, x = 130. - 43., y = 0, #size_x = 0, size_y = 300,
        sourcetype = 'Point source')

    if do_sim:

        #sim.run_sim(runtime = 100, get_mp4 = True,
        #        # dpml=dpml, sim_resolution = resolution
        #        movie_name = 'plane_wave_linear', Nfps = 24, image_every = 5, dpi=300)


        sim.run_sim(runtime = runtime, get_mp4 = True, linear=linear,
                movie_name = 'so_plane_wave_{}_wvl{:d}_rt{}'.\
                format('linear' if linear else 'log', int(wvl), int(runtime)),
                Nfps = 24, image_every = 5, dpi=300)

    ### Analysis part of the run
    if do_analysis:

        analysis = mpo.Analysis(sim)
        analysis.image_plane_beams(wvl=wvl, simres=resolution, runtime=runtime)

        print('analysis.list_efields')
        print(analysis.list_efields)

        sim.get_MEEP_ff()


if __name__ == '__main__':
    main()