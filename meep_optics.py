import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
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
            y_positive = dpml*resolution + mid_y + y_res
            y_negative = dpml*resolution + mid_y - y_res
            
            #Write lens between left and right surface below optical axis
            epsilon_map[x_left:x_right+1, y_negative] *= component.material
            
            #So that the center line is not affectef twice :
            if y_res != 0 :
                #Write lens between left and right surface above optical axis
                epsilon_map[x_left:x_right+1, y_positive] *= component.material
            
            delam = component.delaminate
            #Write AR coating on left surface
            if component.AR_left is not None :
                
                epsilon_map[x_left - component.AR_left - delam: x_left - delam, 
                            y_negative] *= component.AR_material
                
                if y_res != 0 :
                    epsilon_map[x_left - component.AR_left - delam: x_left - delam, 
                                y_positive] *= component.AR_material
            
            #Write AR coating on right surface                    
            if component.AR_right is not None :
                epsilon_map[x_right + 1 + delam: component.AR_right + x_right + 1 + delam, 
                            y_negative] *= component.AR_material
                
                if y_res != 0 :
                    epsilon_map[x_right + 1 + delam: component.AR_right + x_right + 1 + delam, 
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
                    self.geometry.append(c1)
                    self.geometry.append(c2)
                    
                else :
                    self.geometry = [c1,c2]
            
            ### IMAGE PLANE
            elif component.object_type == 'ImagePlane':
                
                c1 = mp.Block(size=mp.Vector3(component.thick, component.diameter, 0),
                      center=mp.Vector3(component.x - self.size_x/2, 0, 0),
                      material = component.material)
                
                self.image_plane_pos = component.x - self.size_x/2
                if self.geometry is not None :
                    self.geometry.append(c1)
                    
                else :
                    self.geometry = [c1]
                
            
        self.permittivity_map = epsilon_map
        
    def plot_lenses(self):
        plt.imshow(self.permittivity_map.transpose())
        
    def write_h5file(self):
        h = h5py.File('epsilon_map.h5', 'w')
        h.create_dataset('eps', data=self.permittivity_map)
        
    def delete_h5file(self):
        file = glob.glob('epsilon_map.h5')
        os.remove(file[0])
        
                
class AsphericLens(object):
    
    def __init__(self, name = '', 
                 r1=None, r2=None, 
                 c1=None, c2=None, 
                 thick=None, 
                 x=0., y=0., 
                 index = 1.52, 
                 AR_left = None, AR_right = None,
                 AR_delamination = 0):
        
        self.name = name
        self.r1 = r1
        self.r2 = r2
        self.c1 = c1
        self.c2 = c2
        self.thick = thick
        self.x = x
        self.y = y
        self.material = index**2
        self.object_type = 'Lens'
        self.AR_left = AR_left
        self.AR_right = AR_right
        self.AR_material = index
        self.delaminate = np.int(AR_delamination)
    
    def left_surface(self, y):
        if self.r1 != np.inf :
            return (y**2/self.r1) / (1 + np.sqrt(1 - (1+ self.c1)*y**2/self.r1**2))
        else : 
            return 0
    
    def right_surface(self, y):
        if self.r2 != np.inf :
            return (y**2/self.r2) / (1 + np.sqrt(1 - (1+ self.c2)*y**2/self.r2**2))
        else : 
            return 0

    
class ApertureStop(object):

    def __init__(self, name = '', 
                 diameter = None, 
                 pos_x = None, 
                 thickness = None, 
                 index  = None, 
                 conductivity = None):
        
        self.name = name
        self.thick = thickness
        self.x = pos_x
        self.diameter = diameter
        self.permittivity = index**2
        self.conductivity = conductivity
        self.object_type = 'AP_stop'

class ImagePlane(object):

    def __init__(self, name = '', 
                 diameter = None, 
                 pos_x = None, 
                 thickness = None, 
                 index  = None, 
                 conductivity = None):
        
        self.name = name
        self.thick = thickness
        self.x = pos_x
        self.diameter = diameter
        if conductivity != np.inf :
            self.material = mp.Medium(epsilon=index**2, 
                                      D_conductivity = conductivity)
        
        else :
            self.material = mp.perfect_electric_conductor
        
        self.object_type = 'ImagePlane'
        
    
    
class Sim(object):
    
    def __init__(self, optical_system):
        self.opt_sys = optical_system
           
    
    
    def PML(self, dpml):
        self.pml_layers = [mp.PML(thickness = dpml)]
        self.cell = mp.Vector3(self.opt_sys.size_x+2*dpml, self.opt_sys.size_y+2*dpml)
        
    def define_source(self, frequency, 
               sourcetype = 'Plane wave', 
               x = 0, y = 0, 
               size_x = 0, size_y = 300, 
               beam_width = 0, 
               focus_pt_x = 0, focus_pt_y = 0):
        
        x_meep = x - self.opt_sys.size_x/2
        y_meep = y
        self.frequency = frequency
        
        if sourcetype == 'Plane wave':
            self.source = [mp.Source(mp.ContinuousSource(frequency, is_integrated=True),
                      component=mp.Ez,
                      center=mp.Vector3(x_meep, y_meep, 0),
                      size=mp.Vector3(size_x, size_y, 0))]
        
        elif sourcetype == 'Gaussian beam':
            self.source = [mp.GaussianBeamSource(mp.ContinuousSource(frequency),
                      component = mp.Ez,
                      center = mp.Vector3(self.opt_sys.image_plane_pos-1, y_meep, 0),
                      beam_x0 = mp.Vector3(focus_pt_x, focus_pt_y),
                      beam_kdir = mp.Vector3(-1, 0),
                      beam_w0 = beam_width,
                      beam_E0 = mp.Vector3(0,0,1),
                      size=mp.Vector3(size_x, size_y, 0))]
        
        return self.source
    
    
    def run_sim(self, runtime = 0., dpml = None, sim_resolution = 1):
        
        dpml = self.opt_sys.dpml
        
        if dpml is None :
            dpml = np.int(np.around(0.5*1/self.frequency))
        
        self.PML(dpml)
        self.dpml = dpml
        self.sim_resolution = sim_resolution

        
        self.sim = mp.Simulation(cell_size=self.cell,
                    boundary_layers=self.pml_layers,
                    geometry=self.opt_sys.geometry, #To use the build made with meep shapes
                    sources=self.source,
                    resolution=self.sim_resolution,
                    epsilon_input_file = 'epsilon_map.h5:eps')
                    #material_function = test_func) #To use the custom lenses
        
        
        self.sim.run(until = runtime)
    
    def plot_system(self):
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
        eps_data = self.sim.get_array(center=mp.Vector3(), size=self.cell, component=mp.Dielectric)
        ez_data = self.sim.get_array(center=mp.Vector3(), size=self.cell, component=mp.Ez)
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha = 0.9)
        #plt.axis('off')
        plt.xlabel('x times resolution')
        plt.ylabel('y times resolution')
        plt.show()
        
    def plot_airy_spot(self):
        plt.figure()
        for k in range(np.int(1/self.frequency*0.5)):
            self.sim.run(until = 1)
            ez_data = self.sim.get_array(center=mp.Vector3(self.opt_sys.image_plane_pos-1, 0), 
                                         size=mp.Vector3(0, self.opt_sys.size_y), component=mp.Ez)
            plt.scatter(np.arange(len(ez_data)), ez_data**2, marker = '+')
        plt.title('$E^2$ at image plane')
        plt.xlabel('y times resolution')
        plt.ylabel('$E^2$')
        plt.show()
    
    def plot_beam(self, single_plot = True, colors = ['r'], plot_n = 0, beam_height = 0.):
        
        if single_plot :
            plt.figure()
            
        timestep = .5
        n_iter = np.int(1/(2*self.frequency*timestep)) 
        
        ez_data = self.sim.get_array(center=mp.Vector3(-self.opt_sys.size_x/2, 0), 
                                         size=mp.Vector3(0, self.opt_sys.size_y), component=mp.Ez)**2
        for k in range(n_iter):
            self.sim.run(until = timestep)
            ez_data += self.sim.get_array(center=mp.Vector3(-self.opt_sys.size_x/2, 0), 
                                         size=mp.Vector3(0, self.opt_sys.size_y), component=mp.Ez)**2
        
        E_mean = ez_data/(n_iter+1)
        plt.plot(np.arange(len(ez_data))/self.sim_resolution, 
                    E_mean, 
                    color = colors[plot_n],
                    alpha = .9,
                    label = '{:d} mm'.format(int(beam_height))) 
        plt.title('Average $E^2$ in front of aperture')
        plt.xlabel('y (mm)')
        plt.ylabel('$E^2$')
    
        if single_plot :
            plt.show()
    
class Analysis(object):
    
    def __init__(self, sim):
        self.sim = sim
        
    def image_plane_beams(self, frequency, sourcetype = 'Gaussian beam', y_max = 0., Nb_sources = 1., sim_resolution = 1):
        
        plt.figure()
        colors = plt.cm.viridis(np.linspace(0, 1, Nb_sources))
        for k in range(Nb_sources):
            height = y_max*k/(Nb_sources-1)
            self.sim.define_source(frequency, 
                                   sourcetype = sourcetype,
                                   x=712.704, y = height, 
                                   size_x = 0, size_y = 300, 
                                   beam_width = 10, 
                                   focus_pt_x = 0, focus_pt_y = 0)
            
            self.sim.run_sim(runtime = 1000, sim_resolution = sim_resolution)
            self.sim.plot_beam(single_plot = False, colors= colors, plot_n = k, beam_height = height)
        
        plt.legend(fontsize = 9)        
        plt.show()
            
        


if __name__ == '__main__':
    
    opt_sys = OpticalSystem('test')
    opt_sys.set_size(800,300)
    
    lens1 = AsphericLens(name = 'Lens 1', 
                         r1 = 327.365, 
                         r2 = np.inf, 
                         c1 = -0.66067, 
                         c2 = 0, 
                         thick = 40, 
                         x = 130.+50., 
                         y = 0., 
                         AR_left = 5, AR_right = 5,
                         AR_delamination = 0)
    
    lens2 = AsphericLens(name = 'Lens 2', 
                         r1 = 269.190, 
                         r2 = 6398.02, 
                         c1 = -2.4029, 
                         c2 = 1770.36, 
                         thick = 40, 
                         x = 40.+130.+369.408+50., 
                         y = 0.,
                         AR_left = 5, AR_right = 5,
                         AR_delamination = 0)
    
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
    
    opt_sys.add_component(lens1)
    opt_sys.add_component(lens2)
    opt_sys.add_component(aperture_stop)
    opt_sys.add_component(image_plane)
    print(opt_sys.list_components())
    
    study_freq = 0.05
    dpml = dpml = np.int(np.around(0.5*1/study_freq))
    
    opt_sys.assemble_system(dpml = dpml, resolution = 1)
    # opt_sys.plot_lenses()
    opt_sys.write_h5file()
    
    sim = Sim(opt_sys)
    # sim.define_source(study_freq, sourcetype = 'Gaussian beam', x=712.704, y= 0, beam_width = 5, focus_pt_x= 0, focus_pt_y=0)
    # sim.run_sim(runtime = 1000, sim_resolution = 1)
    # sim.plot_system()
    # sim.plot_efield()
    #sim.plot_airy_spot()
    # sim.plot_beam()
    
    analysis = Analysis(sim)
    analysis.image_plane_beams(study_freq, y_max = 100, Nb_sources = 10)