![](misc/logo_large.png)

# MEEPART: MEEP for the Analysis of Refractor Telescopes

The goal of this code repository is to provide useful functions for simulating refractor telescopes using the publicly available Meep code library (see link below): https://github.com/NanoComp/meep
MEEP is a Finite Difference Time Domain simulation tool that allows for accurate simulation of Maxwell's equations in any media, i.e. electromagnetic wavec propagation. Using this and tools built in this code to create standard objects in the context of refractor telescopes, the following code allows to propagate EM waves in such a system. The goal is also to provide analysis tools to understand any changes in the system.

MEEPART's main functions in the current state are :
- Plotting a far field beam for a refractor single axis optical system
- Adding the following elements : asymetric lenses, aperture stop, image plane, half-wave plate, telescope tube and absorber material
- Adding defects to the lenses such as non homogeneous index, surface errors etc... (see Guide for more information)
- All of the above for 2D systems

A guide to the code is provided in 2 parts, MEEPART_GUIDE which shows the basic tools of the code and MEEPART_GUIDE_Pt2 which goes into more detail on the possibilities of MEEPART.

Built with MEEP 1.17, h5py 2.10.0, mpi4py 3.0.3, ffmpeg 4.3.1
