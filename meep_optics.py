import meep as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
import h5py
import os
import glob

class OpticalSystem(object):
    """
    This class is used to define the optical system, by creating the dielectric
    map associated to the system that can the be used within the simulation 
    class.
    """
    
    def __init__(self, name=''):
        
        """
        Give a name to the optical system and initialise a geometry that is 
        empty by default
        """
        
        self.name = name
        self.geometry = None
        self.components = []
        
    def __str__(self):
        """
        Returns the name of the system
        """
        return self.name
    
    def set_size(self, size_x, size_y):
        """
        Defines the size of the system:
            - size_x is the size along the optical axis
            - size_y is the size orthogonal to the optical axis in 2D
        """
        
        self.size_x = size_x
        self.size_y = size_y
    
    def add_component(self, component):
        """
        Adds a component to the list of components, these components can be :
            - Lenses
            - Aparture Stop (x1)
            - Image Plane
        """
        
        self.components.append(component)
        
    def list_components(self):
        """
        Returns the list of components by their names
        """
        out_str = ''
        for component in self.components:
            out_str += ' {}'.format(component.name)
        return out_str
    
    def write_lens(self, component, epsilon_map, resolution):
        # The lens equation returns a sag (distance from plane orth. to
        # optical axis) as a function of distance from optical axis y,
        # so the code cycles through the different y to change the 
        # dielectric map between left surface and right surface
        
        # The y axis has its zero in the middle of the cell, the offset
        # is mid_y
        mid_y = np.int(self.size_y*resolution/2)
        
        for y_res in range(mid_y) :
        
            thick = component.thick*resolution
            
            #Left surface sag
            x_left = np.int(np.around((
                        component.left_surface(y_res/resolution) + 
                        component.x)*resolution))
            #Right surface sag       
            x_right = np.int(np.around((
                        component.right_surface(y_res/resolution) + 
                        component.x)*resolution + 
                        thick))
            
            #Above and below the optical axis :
            y_positive = self.dpml*resolution + mid_y + y_res
            y_negative = self.dpml*resolution + mid_y - y_res
            
            #Write lens between left and right surface below optical axis
            epsilon_map[x_left:x_right+1, y_negative] *= component.material
            
            #So that the center line is not affectef twice :
            if y_res != 0 :
                #Write lens between left and right surface above optical axis
                epsilon_map[x_left:x_right+1, y_positive] *= component.material
            
            delam = np.int(np.around(component.delaminate*resolution))
            #Write AR coating on left surface
            if component.AR_left is not None :
                AR_thick = np.int(np.around(component.AR_left*resolution))
                epsilon_map[x_left - AR_thick - delam: x_left - delam, y_negative] *= component.AR_material
                if y_res != 0:
                     epsilon_map[x_left - AR_thick - delam: x_left - delam, y_positive] *= component.AR_material
            
            #Write AR coating on right surface                    
            if component.AR_right is not None :
                AR_thick = np.int(np.around(component.AR_right*resolution))

                epsilon_map[x_right + 1 + delam: AR_thick + x_right + 1 + delam, 
                            y_negative] *= component.AR_material

                if y_res != 0 :
                    epsilon_map[x_right + 1 + delam: AR_thick + x_right + 1 + delam, 
                                y_positive] *= component.AR_material
            
            
    
    def assemble_system(self, resolution = 1, dpml = None):
        
        """
        Creates the map that will be read by the simulation later, as well as
        the geometry object necessary for an absorbing aperture stop and an 
        image plane
        
        Inputs : 
            - resolution : defines how many points/unit of distance, a higher 
            res gives better precision but also longer coomputation
            - dpml : thickness of the outer absorbing layer
        """
        
        self.resolution = resolution
        self.dpml = dpml
        
        # Define the map size, so that the PML is outside of the working system
        epsilon_map = np.ones(((self.size_x + 2*dpml)*resolution+1, 
                               (self.size_y + 2*dpml)*resolution+1)) 
        
                
        #Goes through all the components to add them to the system
        for component in self.components:
            
            #The action is decided with the component type
            
            ### LENSES
            if component.object_type == 'Lens':
                
                self.write_lens(component, epsilon_map, resolution)
     
            ### APERTURE STOP
            elif component.object_type == 'AP_stop':
                
                #The aperture can be done with 2 blocks in 2D, as follows :
                c1 = mp.Block(size=mp.Vector3(component.thick, 
                                              (self.size_y - component.diameter)/2 + dpml, 0),
                              
                      center=mp.Vector3(component.x - self.size_x/2, 
                                        (component.diameter +  self.size_y + 2*dpml)/4, 0),
                      
                      material = mp.Medium(epsilon=component.permittivity, 
                                           D_conductivity = component.conductivity))
                
                c2 = mp.Block(size=mp.Vector3(component.thick, 
                                              (self.size_y - component.diameter)/2 + dpml, 0),
                              
                      center=mp.Vector3(component.x - self.size_x/2, 
                                        -(component.diameter + self.size_y + 2*dpml)/4, 0),
                      
                      material = mp.Medium(epsilon=component.permittivity, 
                                           D_conductivity = component.conductivity))
        
                
                if self.geometry is not None :
                    #If there are already objects in geometry, adds the aperture
                    #instead of replacing what was there
                    
                    self.geometry.append(c1)
                    self.geometry.append(c2)
                    
                else :
                    self.geometry = [c1,c2]
            
            ### IMAGE PLANE
            elif component.object_type == 'ImagePlane':
                
                #The image plane is just a single plane, made with a block :
                c1 = mp.Block(size=mp.Vector3(component.thick, component.diameter, 0),
                      center=mp.Vector3(component.x - self.size_x/2, 0, 0),
                      material = component.material)
                
                self.image_plane_pos = component.x - self.size_x/2
                if self.geometry is not None :
                    #If there are already objects in geometry, adds the image
                    #plane instead of replacing what was there
                    
                    self.geometry.append(c1)
                    
                else :
                    self.geometry = [c1]
                
            
        self.permittivity_map = epsilon_map
        
    def plot_lenses(self):
        #Only plots the lenses, allows to check their dispostion and shape
        
        plt.imshow(self.permittivity_map.transpose())
        plt.savefig('lenses')
        plt.close()
        
    def write_h5file(self):
        #Writes the file that will then be read within the sim function
        
        #rank = MPI.COMM_WORLD.rank

        h = h5py.File('epsilon_map.h5', 'w')#, driver ='mpio', comm=MPI.COMM_WORLD)
        h.create_dataset('eps', data=self.permittivity_map)
        h.close()
        
    def delete_h5file(self):
        #Deletes the h5 file, can be useful when the file is heavy and not to 
        #be kept after simulation
        
        file = glob.glob('epsilon_map.h5')
        os.remove(file[0])
        
    def sys_info(self, wavelength = None, frequency = None):

        c = 299792458.0
        if wavelength is not None :
           frequency = 1/wavelength

        if frequency is not None:
            wavelength = 1/frequency

        print('System size = {} x {} wavelengths'.format(self.size_x/wavelength, self.size_y/wavelength))


                
class AsphericLens(object):
    """
    This class is used to define an aspheric lens of arbitrary shape and 
    position, and creates the function of sag (curvature) that is used to create 
    the permitttivity map
    """
    
    def __init__(self, name = '', 
                 r1=None, r2=None, 
                 c1=None, c2=None, 
                 thick=None, 
                 x=0., y=0., 
                 n_refr = 1.52, 
                 AR_left = None, AR_right = None,
                 AR_delamination = 0):
        
        self.name = name                #NAME OF LENS  
        self.r1 = r1                    #LEFT SURFACE RADIUS
        self.r2 = r2                    #RIGHT SURFACE RADIUS
        self.c1 = c1                    #LEFT SURFACE ASPHERIC PARAMETER
        self.c2 = c2                    #RIGHT SURFACE ASPHERIC PARAMETER
        self.thick = thick              #THICKNESS AT CENTER
        self.x = x                      #X POSITION OF LEFT SURFACE CENTER
        self.y = y                      #Y POSITION OF LEFT SURFACE CENTER
        self.material = n_refr**2       #DIELECTRIC PERMITTIVITY
        self.object_type = 'Lens'
        self.AR_left = AR_left          #LEFT AR COATING THICKNESS
        self.AR_right = AR_right        #RIGHT AR COATING THICKNESS
        self.AR_material = n_refr       #AR COATING PERMITTIVITY
        self.delaminate = AR_delamination #AR COATING DELAMINATION THICKNESS
    
    def left_surface(self, y):
        """
        Aspheric lens equation

        Parameters
        ----------
        y : STR OR LIST
            Distance from optical axis at which the sag is computed

        Returns
        -------
        STR OR LIST
            Sag at at distance y from optical axis.

        """
        
        if self.r1 != np.inf :
            return (y**2/self.r1) / (1 + np.sqrt(1 - (1+ self.c1)*y**2/self.r1**2))
        else : 
            #If the radius is infinite, returns a flat surface, i.e. 0 sag
            return 0
    
    def right_surface(self, y):
        """
        Same as left_surface(self,y)
        """
        
        if self.r2 != np.inf :
            return (y**2/self.r2) / (1 + np.sqrt(1 - (1+ self.c2)*y**2/self.r2**2))
        else : 
            #If the radius is infinite, returns a flat surface, i.e. 0 sag
            return 0

    
class ApertureStop(object):
    """
    Defines an aperture stop of arbitrary position, material and size
    """

    def __init__(self, name = '', 
                 diameter = None, 
                 pos_x = None, 
                 thickness = None, 
                 n_refr  = None, 
                 conductivity = None):
        
        self.name = name                #NAME OF APERTURE STOP
        self.thick = thickness          #THICKNESS OF AP STOP
        self.x = pos_x                  #POSITION ON OPTICAL AXIS
        self.diameter = diameter        #DIAMETER OF AP ASTOP
        self.permittivity = n_refr**2    #INDEX OF MATERIAL
        self.conductivity = conductivity #CONDUCTIVITY
        self.object_type = 'AP_stop'

class ImagePlane(object):
    """
    Defines an image plane of arbitrary position, material and size
    """

    def __init__(self, name = '', 
                 diameter = None, 
                 pos_x = None, 
                 thickness = None, 
                 n_refr  = None, 
                 conductivity = None):
        
        self.name = name                #NAME OF IMAGE PLANE
        self.thick = thickness          #THICKNESS OF IMAGE PLANE
        self.x = pos_x                  #POSITION ON OPTICAL AXIS
        self.diameter = diameter        #DIAMETER OF IMAGE PLANE
        
        if conductivity != np.inf :
            #Defines the material with given properties
            self.material = mp.Medium(epsilon=n_refr**2, 
                                      D_conductivity = conductivity)
        
        else :
            #If the conductivity is infinite, Meep can define a perfect conductor
            self.material = mp.perfect_electric_conductor
        
        self.object_type = 'ImagePlane'
        
    
    
class Sim(object):
    """
    Runs the sim object from MEEP with the dielectric map created for the system
    and a source that can be specified as Gaussian (monochromatic or multichromatic)
    or plane wave.
    """
    
    def __init__(self, optical_system):
        ### Defines the optical system to be used for the simulation
        self.opt_sys = optical_system
           
    
    def PML(self, dpml):
        ### Defines the boundary layer of Perfectly Matched Layer as well as the 
        ### computational cell so that the PML doesn't overlap on the materials.
        
        self.pml_layers = [mp.PML(thickness = dpml)]
        self.cell = mp.Vector3(self.opt_sys.size_x+2*dpml, self.opt_sys.size_y+2*dpml)
        
    def define_source(self, frequency = None,
                      wavelength = None, 
                      sourcetype = 'Plane wave', 
                      x = 0, y = 0, 
                      size_x = 0, size_y = 300, 
                      beam_width = 0, 
                      focus_pt_x = 0, focus_pt_y = 0,
                      fwidth = 0,
                      wvl_width = 0):
        """
        Defines the source to be used by the simulation. Only does one source
        at a time.

        Parameters
        ----------
        frequency : FLOAT, optional
            Frequency of the source
        wavelength : FLOAT, optional
            Wavelength of the source 
        sourcetype : STR, optional
            A source can be a plane wave coming on the aperture
            or a gaussian beam on the image plane. The default is 'Plane wave'.
        x : FLOAT, optional
            x-Position of the source center. The default is 0.
        y : FLOAT, optional
            y-Position of the source center. The default is 0.
        size_x : FLOAT, optional
            x-size of the source. The default is 0.
        size_y : FLOAT, optional
            y-size of the source. The default is 300.
        beam_width : FLOAT, optional
            For a gaussian beam, defines its width. The default is 0.
        focus_pt_x : FLOAT, optional
            For a gaussian beam, defines where is the x position of the focus 
            of the waist. The default is 0.
        focus_pt_y : FLOAT, optional
            For a gaussian beam, defines where is the y position of the focus 
            of the waist. The default is 0..
        fwidth : FLOAT, optional
            If the beam is to be multichromatic, defines the frequency width 
            around the  center frequency. The default is 0.
        wvl_width : FLOAT, optional
            If the beam is to be multichromatic, defines the wavelength width
            around the center wavelength. fwdith = 1/width

        Returns
        -------
        self.source : MEEP source object
            Object that will be used in the sim function.

        """
        
        if wavelength is not None :
           frequency = 1/wavelength

        if frequency is not None:
            wavelength = 1/frequency

        if wvl_width == 0:
            wvl_width = 1/fwidth

        if fwidth == 0:
            fwidth = 1/wvl_width

        #Its easier for the user to define the system such that x=0 is the 
        #plane on the left and note the center of the cell, this allows for that :
        x_meep = x - self.opt_sys.size_x/2
        y_meep = y
        
        #Defines these objects so that they can be sued outside of the 
        #function later :
        self.frequency = frequency
        self.fwidth = fwidth
        
        #Sim is monochromatic by default
        self.multichromatic = False 
        
        #Different action for different source types
        if sourcetype == 'Plane wave':
            self.source = [mp.Source(mp.ContinuousSource(frequency, is_integrated=True),
                           component=mp.Ez,
                           center=mp.Vector3(x_meep, y_meep, 0),
                           size=mp.Vector3(size_x, size_y, 0))]
        
        elif sourcetype == 'Gaussian beam':
            self.source = [mp.GaussianBeamSource(mp.ContinuousSource(frequency),
                      component = mp.Ez,
                      center = mp.Vector3(x_meep, y_meep, 0),
                      beam_x0 = mp.Vector3(focus_pt_x, focus_pt_y),
                      beam_kdir = mp.Vector3(-1, 0),
                      beam_w0 = beam_width,
                      beam_E0 = mp.Vector3(0,0,1),
                      size=mp.Vector3(size_x, size_y, 0))]
            
        elif sourcetype == 'Gaussian beam multichromatic':
            #The multichromatic object is later used so that the averaging time
            #is set accordingly
            self.multichromatic = True
            
            #With a coordinate system defined with the aperture on the left, 
            #the wave vector has to be (-1,0) in 2D coordinates, so that it goes 
            #from the image plane to the system
            #The cutoff is the rise and decay time of the gaussian pulse, it's
            #set to 1 here so that is shorter, but may give rise to high frequency
            #artifacts
            self.source = [mp.GaussianBeamSource(mp.GaussianSource(frequency, fwidth = self.fwidth, cutoff = 5),
                      component = mp.Ez,
                      center = mp.Vector3(self.opt_sys.image_plane_pos-1, y_meep, 0),
                      beam_x0 = mp.Vector3(focus_pt_x, focus_pt_y),
                      beam_kdir = mp.Vector3(-1, 0),
                      beam_w0 = beam_width,
                      beam_E0 = mp.Vector3(0,0,1),
                      size=mp.Vector3(size_x, size_y, 0))]
        
        return self.source
    
    
    def run_sim(self, runtime = 0., dpml = None, sim_resolution = 1):
        """
        Creates the sim environment as defined by MEEP and then runs it.
        
        Parameters
        ----------
        runtime : FLOAT, optional
            Meep time for which the sim should be run. The default is 0..
        dpml : FLOAT, optional
            The PML layer thickness is set by the user when defining the system
            If not, it defaults then to half the main wavelength.
        sim_resolution : FLOAT, optional
            Resolution of the grid created by Meep. Recall that res/freq should 
            be at least 8 in the highest index material. The default is 1.

        Returns
        -------
        None.

        """
        
        
        dpml = self.opt_sys.dpml
        
        if dpml is None :
            #Closest integer to half wavelength
            dpml = np.int(np.around(0.5*1/self.frequency))
        
        
        self.PML(dpml)
        self.dpml = dpml
        self.sim_resolution = sim_resolution

        #Defines the simulation environment, using the various objects defined
        #previously
        self.sim = mp.Simulation(cell_size=self.cell,
                    boundary_layers=self.pml_layers,
                    geometry=self.opt_sys.geometry, 
                    sources=self.source,
                    resolution=self.sim_resolution,
                    epsilon_input_file = 'epsilon_map.h5:eps')     
        
        #Runs the sim
        self.sim.run(until = runtime)
         
        
    def plot_system(self):
        
        #Makes a plot of the various objects in the computational cell, with 
        #the objects in grey and the PML in purple.
        
        eps_data = self.sim.get_array(center=mp.Vector3(), size=self.cell, component=mp.Dielectric)
        pml = np.zeros((eps_data.transpose().shape))
        pml[0: self.dpml*self.sim_resolution, :] = 1
        pml[:, 0: self.dpml*self.sim_resolution] = 1
        pml[:, -self.dpml*self.sim_resolution : ] = 1
        pml[-self.dpml*self.sim_resolution : , :] = 1
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='Greys')
        plt.imshow(pml, cmap = 'Purples', alpha = 0.4)
        plt.xlabel('x times resolution')
        plt.ylabel('y times resolution')
        plt.show()
    
    def plot_efield(self):

        #Makes a plot of the Ez component of the electric field in the system
        
        eps_data = self.sim.get_array(center=mp.Vector3(), size=self.cell, component=mp.Dielectric)
        ez_data = self.sim.get_array(center=mp.Vector3(), size=self.cell, component=mp.Ez)
        plt.figure(figsize = (20,20))
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha = 0.9)
        plt.xlabel('x times resolution')
        plt.ylabel('y times resolution')
        plt.savefig('efield')
        plt.show()
        
    def plot_airy_spot(self):

        #Makes a plot of the evolution of the instantaneous Ez squared over the 
        #image plane, if the system is properly focused, for an incoming plane
        #wave, should give an Airy pattern

        plt.figure()
        for k in range(np.int(5/self.frequency*0.5)):
            self.sim.run(until = 1)
            #Gets the e-field just 1 unit before the image plane
            ez_data = self.sim.get_array(center=mp.Vector3(self.opt_sys.image_plane_pos-1, 0), 
                                         size=mp.Vector3(0, self.opt_sys.size_y), component=mp.Ez)
            plt.scatter(np.arange(len(ez_data)/self.sim_resolution), ez_data**2, marker = '+')
        plt.title('$E^2$ at image plane')
        plt.xlabel('y (mm)')
        plt.ylabel('$E^2$')
        plt.show()
    
    def plot_beam(self, single_plot = True, colors = ['r'], plot_n = 0, linestyle = '-',
                  aper_pos_x = 10):
        """
        Plots the Ez component of the electric field at the aperture. Is used when
        the source is a gaussian beam at the image plane.

        Parameters
        ----------
        single_plot : BOOLEAN, optional
            Default to True, so that it gives a single plot. If single_plot is 
            False, you can call the function several times so that the plots are
            on the same figure
        colors : LIST OF STR, optional
            When using this function for mutliple plots, the color can be chosen
            for plot clarity.
        plot_n : INT, optional
            When plotting for mutliple beams, allows to pick the right color
            within the list. The default is 0.
        linestyle : STR, optional
            When using this function for multiple plots, the linestyle can be chosen
            for plot clarity
        aper_pos_x : FLOAT, optional
            Position of aperture, at which the electric field is computed.
        Returns
        -------
        E_mean : LIST of ARRAYS
            List of the computed electric fields at the aperture

        """

        if single_plot :
            #Displays the plot if single_plot is True
            plt.figure()
        
        #MEEP's lowest timestep that is not rounded by the sim.run is .5
        timestep = .5

        if self.multichromatic:
            #When the beam is multichromatic, the average needs has to be done
            #on the gaussian pulse duration to get the maximum information 
            #The gaussian pulse duration is 2*1/fwidth
            n_iter = np.int(5/(self.fwidth*timestep))

        elif not self.multichromatic: 
            #When the beam is monochromatic, the average can be done on a single 
            #period
            n_iter = np.int(1/(2*self.frequency*timestep)) 
        
        ez_data = self.sim.get_array(center=mp.Vector3(-self.opt_sys.size_x/2+aper_pos_x, 0), 
                                         size=mp.Vector3(0, self.opt_sys.size_y), component=mp.Ez)**2
        ### Averaging
        for k in range(n_iter):
            self.sim.run(until = timestep)
            ez_data += self.sim.get_array(center=mp.Vector3(-self.opt_sys.size_x/2+aper_pos_x, 0), 
                                         size=mp.Vector3(0, self.opt_sys.size_y), component=mp.Ez)**2
        
        E_mean = ez_data/(n_iter+1)

        ### Plot
        plt.plot(np.arange(len(ez_data))/self.sim_resolution, 
                    E_mean, 
                    color = colors[plot_n],
                    alpha = .9,
                    linestyle = linestyle) 
        plt.title('Average $E^2$ in front of aperture, $\lambda = ${:2.1} mm'.format(1/self.frequency))
        plt.xlabel('y (mm)')
        plt.ylabel('$E^2$')
    
        if single_plot :
            #Displays the plot if single_plot is True
            plt.show()
        
        return E_mean
    
class Analysis(object):
    """
    When analyzing an optical system such as a telescope, we want to see the impact
    of changes on the far field beam. To that end, gaussian beams are sent from
    the image plane at different locations to recover different E fields squared
    at the aperture, of which the Fourier Transform can be taken. 
    """
    
    def __init__(self, sim):
        """
        Runs on a specific sim environment : the objects and their properties as 
        specified in the sim will be the same all throughout the analysis.
        """
        self.sim = sim
        
    def image_plane_beams(self, frequency = None, wavelength = None, 
                        fwidth = 0, wvl_width = 0,
                        sourcetype = 'Gaussian beam', 
                        y_max = 0., Nb_sources = 1., sim_resolution = 1,
                        linestyle = '-', runtime = 800):
        """
        Sends gaussian beams (mono or multichromatic) from the image plane and 
        recovers the E-field squared at the aperture. Also plots the electric 
        fields at the aperture.

        Parameters
        ----------
        frequency : FLOAT, optional
            Frequency of the source
        wavelength : FLOAT, optional
            Wavelength of the source 
        fwidth : FLOAT, optional
            If the beams are to be multichromatic, defines the frequency width 
            around the  center frequency. The default is 0.
        wvl_width : FLOAT, optional
            If the beams are to be multichromatic, defines the wavelength width
            around the center wavelength. fwdith = 1/wvl_width
        y_max : FLOAT, optional
            If the number of sources is more than one, the beams sent from the 
            image plane are equally spaced between y=0 on the optical axis and y_max.
        Nb_souyrces : FLOAT, optional
            Number of sources for the analysis. Ideally, 2 so that one is on the optical
            axis and the other on the edge of the image plane, to get a maximum of 
            information without running too many sims.
        sim_resolution : INT, optional
            Is set so that wavelength*resolution should be >8 in the highest 
            index materials
        sourcetype : STR, optional
            Either 'Gaussian beam' or 'Gaussian beam multichromatic.
        linestyle : STR, optional
            When running two analysis to compare some effect, linestyle can 
            be changed between the two so that comparison on the plot is
            easier
        runtime : FLOAT, optional
            This is the time for which the sim should run before aveaging the field 
            over the duration of the gaussian pulse. Ideally it is set so that
            the gaussian pulse arrives just before the aperture stop.
        Returns
        -------
        None, but registers self.list_beams in which the averaged electric fields
        are stored. self.list_beams[k] stores the averaged electric field for the
        k-th beam.

        """

        # Adaptation to specify either in wavelength or frequency :
        if wavelength is not None :
           frequency = 1/wavelength

        if frequency is not None:
            wavelength = 1/frequency

        if wvl_width == 0:
            wvl_width = 1/fwidth

        if fwidth == 0:
            fwidth = 1/wvl_width

        #For the plot of the electric fields, the cmpap is created for the colors
        #to represent the distance from the optical axis
        colors = plt.cm.viridis(np.linspace(0, 1, Nb_sources))

        #Initialize the electric fields list
        self.list_efields = [[] for k in range(Nb_sources)]

        #Iterates over the number of sources
        for k in range(Nb_sources):

            #If there is only one source, the beam is sent on the optical axis
            #Avoids division y zero
            if Nb_sources != 1 :
                height = y_max*k/(Nb_sources-1)
            else : 
                height = 0

            #Defines the source at the appropriate height on the image plane
            self.sim.define_source(frequency, 
                                   sourcetype = sourcetype,
                                   x=self.sim.opt_sys.size_x/2-10, y = height, 
                                   size_x = 0, size_y = 10, 
                                   beam_width = 10, 
                                   focus_pt_x = 0, focus_pt_y = 0,
                                   fwidth = fwidth)
            
            #Runs the sim
            self.sim.run_sim(runtime = runtime, sim_resolution = sim_resolution)

            #Gets the squared averaged electric field and adds it to the plot
            E_field = self.sim.plot_beam(single_plot = False, colors= colors, plot_n = k, linestyle = linestyle)

            #Updates the list of fields
            self.list_efields[k] = E_field
        
    
    def beam_FT(self, aperture_size = 200, precision_factor = 5):

        """
        Gets the Fourier Transforms of the averaged  squared electric fields at aperture.
        Parameters
        ----------
        aperture_size : FLOAT, optional
            Size of the aperture. Only the relevant part of the electric field is then
            used for the fourier transform
        precision_factor : FLOAT, optional
            If the gaussian pattern over the aperture is large, the precision_factor
            is used to add zeros to the electric field list so that the final list is
            bigger by a factor of precision_factor. This adds precision in the fourier
            transform
        Returns
        -------
        freq : List of FLOATS
            List of the frequencies at which the FFT has been done
        FFTs : List of arrays
            Each array contains the FFT for the k-th source.
        """

        #Initialize the list
        FFTs = [[] for k in range(len(self.list_efields))]

        res = self.sim.sim_resolution

        #List of frequencies
        freq = np.fft.fftfreq(aperture_size*res*precision_factor)

        #Iterate over the number of sources
        for k in range(len(self.list_beams)):

            #Indexes of the beginning and end of aperture
            index_ap = np.int(np.around((self.sim.opt_sys.size_y-aperture_size)/2))

            #Truncates field over aperture
            truncated_field = self.list_efields[k][index_ap*res+1:(index_ap+aperture_size)*res+1]

            #Extends field for precision factor
            efield_ext = np.zeros(aperture_size*res*precision_factor)
            n = np.int(precision_factor/2)
            efield_ext[aperture_size*res*n : aperture_size*res*(n+1)] += truncated_field 

            #FFT over the extended field
            FFTs[k] = np.fft.fft(efield_ext)
            FFTs[k] = FFTs[k]/FFTs[k][0]
        
        return freq, FFTs
        


if __name__ == '__main__':
    
    opt_sys = OpticalSystem('test')
    opt_sys.set_size(750,300)
    
    lens1 = AsphericLens(name = 'Lens 1', 
                         r1 = 327.365, 
                         r2 = np.inf, 
                         c1 = -0.66067, 
                         c2 = 0, 
                         thick = 40, 
                         x = 130.+10., 
                         y = 0., 
                         AR_left = .5, AR_right = .5,
                         AR_delamination = 1)
    
    lens2 = AsphericLens(name = 'Lens 2', 
                         r1 = 269.190, 
                         r2 = 6398.02, 
                         c1 = -2.4029, 
                         c2 = 1770.36, 
                         thick = 40, 
                         x = 40.+130.+369.408+10., 
                         y = 0.,
                         AR_left = .5, AR_right = .5,
                         AR_delamination = 1)
    
    aperture_stop = ApertureStop(name = 'Aperture Stop',
                                 pos_x = 10,
                                 diameter = 200,
                                 thickness = 5,
                                 n_refr = 5., 
                                 conductivity = 1e7)
    
    image_plane = ImagePlane(name = 'Image Plane',
                             pos_x = 10+714.704,
                             diameter = 300,
                             thickness = 2,
                             n_refr = 5., 
                             conductivity = 0.01)
    
    opt_sys.add_component(lens1)
    opt_sys.add_component(lens2)
    opt_sys.add_component(aperture_stop)
    opt_sys.add_component(image_plane)
    print(opt_sys.list_components())
    opt_sys.sys_info(frequency = 0.5)
    
    study_freq = 0.5
    dpml = np.int(np.around(0.5*1/study_freq))
    
    opt_sys.assemble_system(dpml = dpml, resolution = 2)
    opt_sys.plot_lenses()
    opt_sys.write_h5file()
    
    sim = Sim(opt_sys)
    sim.define_source(wavelength = 8, sourcetype = 'Gaussian beam', 
                      x=710, y= 0, beam_width = 10, 
                      focus_pt_x= 0, focus_pt_y=0, size_x = 0, size_y=10)
    sim.run_sim(runtime = 800, sim_resolution = 2)
    # sim.plot_system()
    sim.plot_efield()
    # sim.plot_airy_spot()
    # sim.plot_beam()
    """
    analysis = Analysis(sim)
    analysis.image_plane_beams(study_freq, fwidth = 0.01, sourcetype='Gaussian beam multichromatic',
                                y_max = 100, Nb_sources = 2, sim_resolution = 1)
    freq, ffts = analysis.beam_FT(aperture_size = 200)
    
    fft1_dB = 10*np.log10(ffts[0].real**2)
    fft2_dB = 10*np.log10(ffts[1].real**2)

    plt.figure()
    plt.plot(freq*360, fft1_dB)
    #plt.plot(freq, ffts[0].imag)
    plt.plot(freq*360, fft2_dB)
    plt.legend(('FFT Beam On axis', 'FFT Beam Off-axis'))
    plt.ylim((-100, 1))
    plt.xlim((-50,50))
    plt.savefig('Test',dpi=300, bbox_inches='tight')
    plt.close()
    """