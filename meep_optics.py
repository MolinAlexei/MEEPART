import meep as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.optimize as sc
from mpi4py import MPI
import h5py
import os
import glob
import csv

mp.verbosity(0)

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

        thick = component.thick*resolution

        #Generate the center of the lumps made by delamination, 
        #different for the left and right surface
        high = np.int(np.around(self.size_y*0.9/2))
        y0_left = np.random.randint(low = -high, high = high)
        y0_right = np.random.randint(low = -high, high = high)

        radius = np.int(component.diameter*resolution/2)

        if component.surf_err_scale!=0 :
            nb_bins = int(component.diameter/component.surf_err_width)
            err_left = np.around(np.random.normal(scale = component.surf_err_scale*resolution, size = nb_bins))
            err_right = np.around(np.random.normal(scale = component.surf_err_scale*resolution, size = nb_bins))

        if component.surf_err_scale == 0:
            nb_bins = int(component.diameter/component.surf_err_width)
            err_left = np.zeros(nb_bins)
            err_right = np.zeros(nb_bins)
        

        for y_res in range(radius) :           
            
            #Left surface sag
            x_left = np.int(np.around((
                        component.left_surface(y_res/resolution) + self.dpml + 
                        component.x - component.thermal_deformation((y_res+mid_y)/resolution))*resolution))
            #Right surface sag       
            x_right = np.int(np.around((
                        component.right_surface(y_res/resolution) + 
                        component.x + self.dpml)*resolution + 
                        thick))
            
            #Above and below the optical axis :
            y_positive = self.dpml*resolution + mid_y + y_res
            y_negative = self.dpml*resolution + mid_y - y_res

            #Get the delamination as a function of y on left surface
            delam_pos_L = np.int(np.around(resolution*
                component.delamination(y_res/resolution, y0_left)))
            delam_neg_L = np.int(np.around(resolution*
                component.delamination(-y_res/resolution, y0_left)))

            #Get the delamination as a function of y on right surface
            delam_pos_R = np.int(np.around(resolution*
                component.delamination(y_res/resolution, y0_right)))
            delam_neg_R = np.int(np.around(resolution*
                component.delamination(-y_res/resolution, y0_right)))
            
            #Gradient in the index
            radial_slope = component.radial_slope/resolution
            axial_slope = component.axial_slope/resolution
            n0 = component.material
            x0 = np.int(np.around(component.x*resolution))
            x_range = range(x_left, x_right+1) 
            material_line = [n0 + y_res*radial_slope + (k-x0)*axial_slope for k in x_range]


            #Surface error
            err_left_pos = int(err_left[int(np.around(y_res/resolution/component.surf_err_width))])
            err_left_neg = int(err_left[- int(np.around(y_res/resolution/component.surf_err_width))])

            err_right_pos = int(err_left[int(np.around(y_res/resolution/component.surf_err_width))]) 
            err_right_neg = int(err_left[- int(np.around(y_res/resolution/component.surf_err_width))])

            x_left_neg = x_left + err_left_neg
            x_left_pos = x_left + err_left_pos

            x_right_neg = x_right + err_right_neg
            x_right_pos = x_right + err_right_pos


            #Write lens between left and right surface below optical axis
            epsilon_map[x_left_neg : x_right_neg+1, y_negative] *= material_line
            
            #So that the center line is not affected twice :
            if y_res != 0 :
                #Write lens between left and right surface above optical axis
                epsilon_map[x_left_pos : x_right_pos+1, y_positive] *= material_line
            
            #Write AR coating on left surface
            if component.AR_left is not None :

                AR_thick = np.int(np.around(component.AR_left*resolution))

                epsilon_map[x_left_neg - AR_thick - delam_neg_L: x_left_neg - delam_neg_L, 
                            y_negative] *= component.AR_material

                if y_res != 0 :
                    epsilon_map[x_left_pos - AR_thick - delam_pos_L: x_left_pos - delam_pos_L, 
                                y_positive] *= component.AR_material
            
            #Write AR coating on right surface                    
            if component.AR_right is not None :
                
                AR_thick = np.int(np.around(component.AR_right*resolution))

                epsilon_map[x_right_neg + 1 + delam_neg_R: AR_thick + x_right_neg + 1 + delam_neg_R, 
                            y_negative] *= component.AR_material

                if y_res != 0 :
                    epsilon_map[x_right_pos + 1 + delam_pos_R: AR_thick + x_right_pos + 1 + delam_pos_R, 
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
                               (self.size_y + 2*dpml)*resolution+1), dtype = 'float32') 
        
                
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
        
                
                self.aper_pos_x = component.x
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
                
            elif component.object_type == 'MetallicTube':

                #The aperture can be done with 2 blocks in 2D, as follows :
                c1 = mp.Block(size=mp.Vector3(self.size_x, component.thick, 0),
                      center=mp.Vector3(0, component.center, 0),
                      material = mp.metal)
                
                c3 = mp.Block(size=mp.Vector3(self.size_x, component.thick, 0),
                      center=mp.Vector3(0, -component.center, 0),
                      material = mp.metal)
                
                if self.geometry is not None :
                    #If there are already objects in geometry, adds the tube
                    #instead of replacing what was there
                    
                    self.geometry.append(c1)
                    self.geometry.append(c3)
                    
                else :
                    self.geometry = [c1, c3]

            elif component.object_type == 'Absorber':
                c2 = mp.Block(size=mp.Vector3(self.size_x, component.thick, 0),
                      center=mp.Vector3(0, component.center, 0),               
                      material = mp.Medium(epsilon=component.epsilon_real, 
                                            D_conductivity=component.conductivity))


                c4 = mp.Block(size=mp.Vector3(self.size_x, component.thick, 0),
                      center=mp.Vector3(0, -component.center, 0),               
                      material = mp.Medium(epsilon=component.epsilon_real, 
                                            D_conductivity=component.conductivity))

                
                if self.geometry is not None :
                    #If there are already objects in geometry, adds the absorber
                    #instead of replacing what was there
                    
                    self.geometry.append(c2)
                    self.geometry.append(c4)
                    
                else :
                    self.geometry = [c2, c4]

            elif component.object_type == 'HWP':
                HWP = mp.Block(center = mp.Vector3(component.x_pos - self.size_x/2, 0, 0),
                                size = mp.Vector3(component.thick, component.size_y, 0),
                                material = mp.Medium(epsilon_diag = component.diag, 
                                    epsilon_offdiag = component.offdiag))

                if self.geometry is not None :
                    #If there are already objects in geometry, adds the HWP
                    #instead of replacing what was there
                    
                    self.geometry.append(HWP)
                    
                else :
                    self.geometry = [HWP]

            elif component.object_type == 'PrismsTube':

                vertices_up = [mp.Vector3(-1.5,0),
                            mp.Vector3(1.5,0),
                            mp.Vector3(0,5)]

                vertices_down = [mp.Vector3(-1.5,0),
                            mp.Vector3(1.5,0),
                            mp.Vector3(0,-5)]

                
                for k in range(250):

                    prism_up = mp.Prism(vertices =vertices_up, 
                                    center = mp.Vector3(-380+k*3, -155), 
                                    material = mp.Medium(epsilon =1.5, D_conductivity = 10), 
                                    height = 1)
                    prism_down = mp.Prism(vertices =vertices_down, 
                                    center = mp.Vector3(-380+k*3, 155), 
                                    material = mp.Medium(epsilon =1.5, D_conductivity = 10), 
                                    height = 1)

                    self.geometry.append(prism_up)
                    self.geometry.append(prism_down)

                
                c1 = mp.Block(size=mp.Vector3(self.size_x, 5, 0),
                      center=mp.Vector3(0, 157.5, 0),
                      material = mp.metal)
                c2 = mp.Block(size=mp.Vector3(self.size_x, 5, 0),
                      center=mp.Vector3(0, -157.5, 0),
                      material = mp.metal)

                self.geometry.append(c1)
                self.geometry.append(c2)

            
        self.permittivity_map = epsilon_map

    def make_lens_bubbles(self, radius, nb_clusters, nb_per_cluster, r_factor = 1):

        """
        Introduces clusters of air bubbles inside the lenses of the system, 
        each cluster has a central bubble and a number of smaller bubble gathered
        around this central bubble
        
        Inputs : 
            - radius : radius of the central bubble
            - nb_clusters : number of clusters per lens
            - nb_per_cluster : number of bubbles surrounding the main one in each
            cluster
            - r_factor : the factor by which the radii are multiplied, allows for the 
            investigation of bubble size only

        Affects the self.permittivity_map object.
        """

        res = self.resolution

        #Function, given a radius, that returns the indices of the points within 
        #the circle centered on (0,0)
        def bubble(rad):
            bubble = []
            for k in range(-rad, rad+1):
                for j in range(-rad, rad+1):
                    if k**2 + j**2 <= rad**2 :
                        bubble.append([k,j])
            return np.array(bubble)

        #List of centers of bubbles
        list_centers = []

        #List of radii of bubbles
        list_radii = []

        #Iterate for all lenses
        for component in self.components:

            if component.object_type == 'Lens':

                #Lens thickness
                thick = component.thick*res

                #So that the bubbles aren't generated on the very edge of the lenses
                low = np.int(np.around(self.size_y*res*0.1))
                high = np.int(np.around(self.size_y*res*0.9))

                #Iterate over cluster numbers
                for i in range(nb_clusters):

                    #The center of the lens can be anywhere on the y axis
                    y0 = np.random.randint(low = low, high = high)

                    #Left surface sag
                    x_left = np.int(np.around((
                        component.left_surface(y0/res - self.size_y/2) + 
                        component.x)*res))
                    #Right surface sag       
                    x_right = np.int(np.around((
                        component.right_surface(y0/res - self.size_y/2) + 
                        component.x)*res + 
                        thick))

                    #The center of the cluster has to be inside the lens
                    x0 = np.random.randint(low = x_left, high = x_right+1)

                    #Radius of the main can vary by 10 percent
                    radius_0 = radius*(0.9 + np.random.random()*0.2)
                
                    #Update lists
                    list_centers.append([x0,y0])
                    list_radii.append(radius_0)

                    #Iterate over the number of surrounding bubbles
                    for k in range(nb_per_cluster):

                        #The center of each surrounding bubble is random, within
                        #a certain distance of the central bubble
                        phi = np.random.random()*2*np.pi
                        r = radius_0*(1 + np.random.random()*3)

                        #change of variables
                        x_k = np.int(np.around(r*np.cos(phi)*res))
                        y_k = np.int(np.around(r*np.sin(phi)*res))

                        #The radius is a function of distance, the farther the 
                        #smaller
                        radius_k = radius_0*np.exp(-r/(3*radius_0))*np.random.random()

                        #Update lists
                        list_centers.append([x0+x_k, y0+y_k])
                        list_radii.append(radius_k)

        list_centers = np.array(list_centers)
        list_radii = np.array(list_radii)
        list_radii *= r_factor
        list_all = []

        #Making bubbles for all centers and radii
        for k in range(len(list_centers)):
            radius_k = np.int(np.around(list_radii[k]*res))
            bubble_k = bubble(radius_k)
            for u in bubble_k : 
                list_all.append(list_centers[k] + u)

        #Update the map
        for index in list_all : 
            self.permittivity_map[index[0], index[1]] = 1



            
    def plot_lenses(self):
        #Only plots the lenses, allows to check their dispostion and shape
        
        plt.figure(figsize = (15,15))
        plt.imshow(self.permittivity_map.transpose())
        plt.savefig('lenses')
        plt.close()
        
    def write_h5file(self, parallel = False):
        #Writes the file that will then be read within the sim function
        
        if parallel : 
            comm = MPI.COMM_WORLD
            rank = comm.rank

            h = h5py.File('epsilon_map.h5', 'w', driver ='mpio', comm=MPI.COMM_WORLD)
        
            size_x = len(self.permittivity_map[:,0])
            size_y = len(self.permittivity_map[0,:])
            dset = h.create_dataset('eps', (size_x, size_y), dtype = 'float32', compression = "gzip")
            with dset.collective :
                dset[:,:] = self.permittivity_map
        
            h.close()

        elif not parallel :
            h = h5py.File('epsilon_map.h5', 'w')
            dset = h.create_dataset('eps', data = self.permittivity_map)
            h.close()

        #self.permittivity_map = 0
        
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

        print('System size = {} x {} wavelengths'.format(self.size_x/wavelength, 
                                                        self.size_y/wavelength))


class TelescopeTube(object):
    def __init__(self, name ='', thick = 5, center = 157.5):

        self.name = name                        #NAME OF OBJECT
        self.object_type = 'MetallicTube'       #OBJECT TYPE
        self.thick = thick                      #TUBE THICKNESS
        self.center = center                    #DISTANCE OF CENTER FROM OPTICAL AXIS


class Absorber(object):
    def __init__(self, name ='', thick = 5, center = 157.5,
                epsilon_real = 3.5, epsilon_imag = 0.11025, freq = 1/3):

        self.name = name                        #NAME OF OBJECT
        self.object_type = 'Absorber'           #OBJECT TYPE
        self.thick = thick                      #TUBE THICKNESS
        self.center = center                    #DISTANCE OF CENTER FROM OPTICAL AXIS
        self.epsilon_real = epsilon_real            #REAL PART OF PERMITTIVITY
        self.epsilon_imag = epsilon_imag            #COMPLEX PART OF PERMITTIVITY
        self.freq = freq                        #OPERATING FREQUENCY
        self.conductivity = epsilon_imag*2*np.pi*freq/epsilon_real #CONDUCTIVITY FOR MEEP


        #ECCOSORB CR110 @ 100 GHz : eps_real = 3.5, eps_imag = 0.11025, freq = 1/3

class HalfWavePlate(object):
    def __init__(self, name = '', fast_ax_index = 3.019, slow_ax_index = 3.336, thick = 5, x_pos = 0, size_y = 300, theta = np.pi/4):
        eps_0 = fast_ax_index**2
        eps_e = slow_ax_index**2

        self.object_type = 'HWP'
        self.diag = mp.Vector3(eps_0, 
                        eps_0*(np.cos(theta)**2) + eps_e*(np.sin(theta)**2), 
                        eps_0*(np.sin(theta)**2) + eps_e*(np.cos(theta)**2))
        self.offdiag = mp.Vector3(0,0, (eps_e-eps_0)*np.cos(theta)*np.sin(theta))
        self.thick = thick
        self.x_pos = x_pos
        self.size_y = size_y
        

class AsphericLens(object):
    """
    This class is used to define an aspheric lens of arbitrary shape and 
    position, and creates the function of sag (curvature) that is used to create 
    the permitttivity map
    """
    
    def __init__(self, name = '',
                diameter = 300, 
                 r1=None, r2=None, 
                 c1=None, c2=None, 
                 thick=None, 
                 x=0., y=0., 
                 n_refr = 1.52, 
                 AR_left = None, AR_right = None,
                 delam_thick = 0,
                 delam_width = 10,
                 radial_slope = 0,
                 axial_slope = 0,
                 surf_err_width = 1,
                 surf_err_scale = 0,
                 therm_def = False):
        
        self.name = name                #NAME OF LENS  
        self.diameter = diameter        #DIAM
        self.r1 = r1                    #LEFT SURFACE RADIUS
        self.r2 = r2                    #RIGHT SURFACE RADIUS
        self.c1 = c1                    #LEFT SURFACE ASPHERIC PARAMETER
        self.c2 = c2                    #RIGHT SURFACE ASPHERIC PARAMETER
        self.thick = thick              #THICKNESS AT CENTER
        self.x = x                      #X POSITION OF LEFT SURFACE CENTER
        self.y = y                      #Y POSITION OF LEFT SURFACE CENTER
        self.material = n_refr**2       #DIELECTRIC PERMITTIVITY
        self.object_type = 'Lens'       #OBJECT TYPE

        self.AR_left = AR_left          #LEFT AR COATING THICKNESS
        self.AR_right = AR_right        #RIGHT AR COATING THICKNESS
        self.AR_material = n_refr       #AR COATING PERMITTIVITY
        self.delam_thick = delam_thick  #AR COATING DELAMINATION THICKNESS
        self.delam_width = delam_width  #DELAMINATION LUMP THICKNESS

        self.radial_slope = radial_slope#RADIAL GRADIENT IN THE INDEX
        self.axial_slope = axial_slope  #AXIAL GRADIENT IN THE INDEX

        self.surf_err_width = surf_err_width    #SURFACE ERROR WIDTH
        self.surf_err_scale = surf_err_scale    #SURFACE ERROR SCALE

        self.therm_def = therm_def      #ENABLES THERMAL DEFORMATION

        deform = []

        #with open('deformedsurface.csv') as csvfile:
        #    reader = csv.reader(csvfile, delimiter=',')
        #    k = 0
        #    for row in reader:
        #        k+= 1 
        #        if k>=11 :
        #            deform.append(np.float(row[2]))

        #deform0 = 2*deform[0]-deform[1]
        #deform.insert(0, deform0)
        self.deform = deform

    
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

    def delamination(self, y, y0):

        #Returns the air layer thickness that makes delamination, it is 
        #zero everywhere excpet where there's the lump, centered on y0, defined by
        #its width and thickness

        thick = self.delam_thick
        width = self.delam_width
        return np.abs(min((((y-y0)/width)**2-1)*thick, 0))


    def thermal_deformation(self, y):
 
        if self.therm_def :
            x = np.linspace(0,300, len(self.deform))

            interp_surf = np.interp(y, x, self.deform)

            return interp_surf

        else :
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
        self.cell_size_x = self.opt_sys.size_x+2*dpml
        self.cell_size_y = self.opt_sys.size_y+2*dpml
        self.cell = mp.Vector3(self.opt_sys.size_x+2*dpml, self.opt_sys.size_y+2*dpml)
        
    def define_source(self, frequency = None,
                      wavelength = None, 
                      sourcetype = 'Plane wave', 
                      x = 0, y = 0, 
                      size_x = 0, size_y = 300, 
                      beam_width = 0, 
                      focus_pt_x = 0, focus_pt_y = 0,
                      fwidth = 0):
        """
        Defines the source to be used by the simulation. Only does one source
        at a time

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
        
        Returns
        -------
        self.source : MEEP source object
            Object that will be used in the sim function.

        """
        
        if wavelength is not None :
           frequency = 1/wavelength

        if frequency is not None:
            wavelength = 1/frequency

        #Its easier for the user to define the system such that x=0 is the 
        #plane on the left and not the center of the cell, this allows for that :
        x_meep = x - self.opt_sys.size_x/2
        y_meep = y
        
        #Defines these objects so that they can be sued outside of the 
        #function later :
        self.wavelength = wavelength
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
                      center = mp.Vector3(self.opt_sys.image_plane_pos, y_meep, 0),
                      beam_x0 = mp.Vector3(focus_pt_x, focus_pt_y),
                      beam_kdir = mp.Vector3(-1, 0),
                      beam_w0 = beam_width,
                      beam_E0 = mp.Vector3(0,0,1),
                      size=mp.Vector3(size_x, self.opt_sys.size_y, 0))]

            self.beam_waist = beam_width
            
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
    
    
    def run_sim(self, runtime = 0., dpml = None, sim_resolution = 1, get_mp4 = False, Nfps = 24):
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


        #n2f_obj = self.sim.add_near2far(self.frequency, 0, 1, mp.Near2FarRegion(center=mp.Vector3(-390), size=mp.Vector3(y=200)))

        #Runs the sim
        

        if get_mp4 :
            self.sim.sim.run(mp.at_every(1, animate), until = runtime)
            animate.to_mp4(Nfps, 'test.mp4')

        if not get_mp4 :
            self.sim.run(until = runtime)
        
        #self.oui = abs(self.sim.get_farfields(n2f_obj, 10, center=mp.Vector3(-2000), size=mp.Vector3(y=800))['Ez'])**2
        
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
        plt.savefig('2Dsystem')
        plt.show()
        plt.close()
    
    def plot_efield(self, path = '.', comp = 'Ez') :

        #Makes a plot of the Ez component of the electric field in the system
        
        eps_data = self.sim.get_array(center=mp.Vector3(), size=self.cell, component=mp.Dielectric)
        if comp == 'Ez' :
            e_data = self.sim.get_array(center=mp.Vector3(), size=self.cell, component=mp.Ez)
        elif comp == 'Ey':
            e_data = self.sim.get_array(center=mp.Vector3(), size=self.cell, component=mp.Ey)

        extent = [0, self.cell_size_x, 0, self.cell_size_y]
        plt.figure(figsize = (20,20))
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary', extent = extent)
        plt.imshow(e_data.transpose(), interpolation='spline36', cmap='RdBu', alpha = 0.9, extent = extent)
        x = np.ones(20)*self.opt_sys.image_plane_pos + self.cell_size_x/2
        y = np.linspace(-self.beam_waist, self.beam_waist, 20) + self.cell_size_y/2
        plt.plot(x, y, c = 'green', linewidth = 3)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('{}/efield'.format(path))
        plt.show()
        plt.close()

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
    
    def plot_beam(self, plot_amp = False, filename = 'test',
                  aper_pos_x = 10, aperture_size = 200):
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
        amplitude*phase : complex ARRAY
            Complex electric field at the aperture

        """       
        #Setting the timestep to a very low value, so that MEEP uses its lowest timestep
        timestep = .3

        #50 steps Is roughly enough to give a few periods for wavelengths from 1 to 10
        n_iter = 150
        
        #Get the real field at aperture
        efield = self.sim.get_array(center=mp.Vector3(-self.opt_sys.size_x/2+aper_pos_x, 0), 
                                         size=mp.Vector3(0, aperture_size), component=mp.Ez)

        res = self.sim_resolution

        #Initializes the list containing the E field evolution
        e_field_evol = np.ones((n_iter, len(efield)))
        e_field_evol[0] = efield

        #List to get the precise timestepping done by meep
        time = np.zeros(n_iter)
        time[0] = self.sim.meep_time()

        ### Stacking the electric field evolution on the aperture
        for k in range(n_iter):
            self.sim.run(until = timestep)
            time[k] = self.sim.meep_time()
            e_field_evol[k] = self.sim.get_array(center=mp.Vector3(-self.opt_sys.size_x/2+aper_pos_x, 0), 
                                         size=mp.Vector3(0, aperture_size), component=mp.Ez)
        
        #Each point on the aperture is fit for a cosine with amplitude and phase
        def f(x, amp, phase):
            return amp*np.cos(x*2*np.pi/self.wavelength + phase)

        #Initialize the lists of amplitude and phase over the aperture
        amplitude = np.zeros(aperture_size*res)
        phase = np.zeros(aperture_size*res)
        #err = np.zeros(aperture_size*res)

        #The field is only taken on the opening of the aperture
        idx = 0 #np.int((self.opt_sys.size_y - aperture_size)/2)

        #Fits amplitude and phase for each point
        for k in range(aperture_size*res):
            popt, pcov = sc.curve_fit(f, time, e_field_evol[:, idx*res+k])
            amplitude[k] = popt[0]
            phase[k] = popt[1]

            #err[k] = np.mean((e_field_evol[:, k] - f(time, popt[0], popt[1]))/e_field_evol[:, k])

        ### Plot
        
    
        if plot_amp :
            #Displays the plot if single_plot is True
            norm = np.max(np.abs(amplitude**2))
            amp = 10*np.log10(np.abs(amplitude**2)/norm)
            y = np.linspace(-aperture_size/2,aperture_size/2,len(amplitude))
            plt.figure()
            plt.plot(y, amp) 
            plt.ylim((-60,0))
            plt.xlim((0, aperture_size/2))
            plt.title('E field amplitude on aperture')
            plt.xlabel('y (mm)')
            plt.ylabel('$Amplitude [dB]$')
            plt.savefig(filename + '.png')
            plt.close()

            h = h5py.File(filename + '.h5', 'w', driver ='mpio', comm=MPI.COMM_WORLD)
            h.create_dataset('y', data=y)
            h.create_dataset('amplitude', data=amplitude, dtype = 'float64')
            h.create_dataset('phase', data=phase, dtype = 'float64')
            h.close()
        
        return amplitude*np.exp(1j*phase)
    
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
                        fwidth = 0,
                        sourcetype = 'Gaussian beam', 
                        y_max = 0., Nb_sources = 1, sim_resolution = 1,
                        linestyle = '-', runtime = 800, aperture_size = 200,
                        beam_w0 = 10, plot_amp = False, plotname = 'test'):
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

        self.wavelength = wavelength
        self.aperture_size = aperture_size

        #For the plot of the electric fields, the cmap is created for the colors
        #to represent the distance from the optical axis
        colors = plt.cm.viridis(np.linspace(0, 1, Nb_sources))

        #Initialize the electric fields list
        self.list_efields = [[] for k in range(Nb_sources)]
        self.FWHM_ap = [0 for k in range(Nb_sources)]

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
                                   x=self.sim.opt_sys.image_plane_pos, y = height, 
                                   size_x = 0, size_y = self.sim.opt_sys.size_y, 
                                   beam_width = beam_w0)
            
            #Runs the sim
            self.sim.run_sim(runtime = runtime, sim_resolution = sim_resolution)

            #Gets the complex electric field and adds it to the plot
            E_field = self.sim.plot_beam(plot_amp = plot_amp,
                filename = plotname,
                aperture_size = aperture_size, aper_pos_x = self.sim.opt_sys.aper_pos_x)

            #Get the FWHM for the field at aperture
            middle_idx = np.int(len(E_field)/2)
            c = max(E_field[middle_idx-5*sim_resolution: middle_idx+5*sim_resolution].real) 

            j = middle_idx
            while E_field[j].real>c/2 : 
                j+=1
                if j == len(E_field):
                    j = middle_idx
                    break
            FWHM_ap = (j-middle_idx)*2/sim_resolution
 
            #FWHM is zero if the field has a max not in the middle
            if np.max(E_field.real)>c :
                FWHM_ap = 0

            self.FWHM_ap[k] = FWHM_ap

            #Updates the list of fields
            self.list_efields[k] = E_field

    def beam_FT(self, precision_factor = 15):

        """
        Gets the Fourier Transforms of the complex electric fields at aperture.
        Parameters
        ----------
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

        self.FWHM_fft = [0 for k in range(len(self.list_efields))]

        #List of frequencies
        freq = np.fft.fftfreq(len(self.list_efields[0])*precision_factor, d = 1/res)

        #Iterate over the number of sources
        for k in range(len(self.list_efields)):

            #FFT over the field
            fft = np.fft.fft(self.list_efields[k], n = precision_factor*len(self.list_efields[k]))

            #Corrective factor for distance to the aperture
            beam = fft*(1+(freq*self.wavelength)**2)

            #FFT is normalized by its max
            FFTs[k] = np.abs(beam) #beam.real
            FFTs[k] = FFTs[k]/np.max(FFTs[k])

            #Get the FWHM for the field at aperture, with a linear interpolation
            j = 0
            while np.abs(FFTs[k][j].real)>0.5 : 
                j+=1

            freq_interp = np.interp(0.5, 
                                        (np.abs(FFTs[k][j].real), np.abs(FFTs[k][j-1].real)), 
                                        (np.arctan(freq[j]*self.wavelength),np.arctan(freq[j-1]*self.wavelength)))
            self.FWHM_fft[k] = freq_interp*2*180/np.pi
        return freq, FFTs

    def plotting(self, freq, FFTs, wavelength,
                deg_range =20,
                ylim = -60, 
                symmetric_beam = True,
                legend = None,
                print_solid_angle = False,
                print_fwhm = False,
                savefig = False,
                path_name = 'plots/meep_guide_plot'):


        deg = np.arctan(freq*wavelength)*180/np.pi
        rads = np.array(deg) * np.pi/180
        rads = np.append(rads, 0)

        plt.figure(figsize = (8,6))
        
        def gaussian(x, stddev, mean):
            return np.exp(-(((x-mean)/4/stddev)**2))
        
        for k in range(len(FFTs)):

            fft_k = FFTs[k]
            fft_dB = 10*np.log10(np.abs(fft_k))


            #BEAM SOLID ANGLE CALCULATION
            if print_solid_angle :
                
                middle = int(len(fft_k)/2)
                integrand = np.append(fft_k, fft_k[0])
                right_part = np.trapz(integrand[:middle], x = rads[:middle])
                left_part = np.trapz(integrand[middle:], x = rads[middle:])
                solid_angle = right_part + left_part
                print('Beam n.{} solid angle : {:.3e} srads'.format(k, solid_angle*2*np.pi))
            
            if legend is not None : 
                plt.plot(deg, fft_dB, label = '{}'.format(legend[k]))

            if legend is None :
                plt.plot(deg, fft_dB)

            #BEST FIT GAUSSIAN FWHM
            if print_fwhm :
                popt, psig = sc.curve_fit(gaussian, deg, fft_k)
                fwhm = popt[1] + 4*popt[0]*np.sqrt(np.log(2))
                fwhm_th = wavelength/self.aperture_size*180/np.pi
                print('Best fit Gaussian FWHM : {:.2f}deg'.format(2*fwhm))
                print('Theoretical FWHM : {:.2f}deg'.format(fwhm_th))
                y = 10*np.log10(gaussian(deg, popt[0], popt[1]))
                #plt.plot(deg, y, linestyle = '--')


        plt.ylim((ylim, 0))
        plt.xlabel('Angle [deg]', fontsize = 14)
        plt.ylabel('Power [dB]', fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)

        if symmetric_beam :
            plt.xlim((0,deg_range))
        if not symmetric_beam : 
            plt.xlim((-deg_range, deg_range))

        if legend is not None :
            plt.legend(loc = 'upper right', fontsize = 12)


        #elif len(fft) == 1:
        #    plt.xlim((0,10))

        #plt.legend(('+2 $\%$ change', '0 $\%$ change', '-2 $\%$ change'))
        #plt.legend(('1mm', '0.5mm', '0.25mm'))

        """
        fwhm = args.wvl*0.28648

        plt.vlines([-fwhm/2, fwhm/2], -100, 0, color = 'grey', linestyle = 'dashdot')
        plt.vlines([fwhm_fft[0]/2], -100, 0, color='grey', linestyle = '--', alpha = 0.7)
        plt.annotate('Expected FWHM : {:.2f}deg'.format(fwhm), 
            xy = (.25, .9), xycoords='figure fraction', color = 'grey')
        plt.annotate('Beam FWHM : {:.2f}deg'.format(fwhm_fft[0]), 
            xy = (.25, .87), xycoords='figure fraction', color = 'grey', alpha = 0.7)
        """

        #plt.annotate('Field FWHM : {:.2f}mm'.format(fwhm_ap[0]), 
        #    xy = (.1, .84), xycoords='figure fraction')
        plt.tight_layout()
        plt.show()
        if savefig :
            plt.savefig('{}.png'.format(path_name))
        plt.close()


if __name__ == '__main__':
    
    opt_sys = OpticalSystem('test')
    opt_sys.set_size(800,300)
    
    lens1 = AsphericLens(name = 'Lens 1', 
                         r1 = 327.365, 
                         r2 = np.inf, 
                         c1 = -0.66067, 
                         c2 = 0, 
                         thick = 40, 
                         x = 130.+10., 
                         y = 0.)
    
    lens2 = AsphericLens(name = 'Lens 2', 
                         r1 = 269.190, 
                         r2 = 6398.02, 
                         c1 = -2.4029, 
                         c2 = 1770.36, 
                         thick = 40, 
                         x = 40.+130.+369.408+10., 
                         y = 0.)
    
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
                             n_refr = 1.2, 
                             conductivity = 0.01)
    


    wavelength = 10
    dpml = 5
    
    tube = TelescopeTube('Tube', thick =10, center = 165)
    absorber = Absorber('Absorber', thick = 10, center = 155, epsilon_real = 3.5, epsilon_imag = 0.05, freq = 1/wavelength)


    opt_sys.add_component(lens1)
    opt_sys.add_component(lens2)
    opt_sys.add_component(aperture_stop)
    opt_sys.add_component(image_plane)
    #opt_sys.add_component(tube)
    #opt_sys.add_component(absorber)    
    

    
    opt_sys.assemble_system(dpml = dpml, resolution = 1)
    opt_sys.plot_lenses()
    opt_sys.write_h5file()

    
    sim = Sim(opt_sys)
    """
    sim.define_source(wavelength = wavelength, sourcetype = 'Gaussian beam', 
                      x=0, y=0, beam_width = 10, 
                      focus_pt_x= 0, focus_pt_y=0, size_x = 0, size_y=10)
    sim.run_sim(runtime = 800, sim_resolution = 1)
    sim.plot_efield()
    """

    analyse = Analysis(sim)
    analyse.image_plane_beams(wavelength = wavelength, sim_resolution = 1)
    freq, ft = analyse.beam_FT()
    analyse.plotting(freq, ft, wavelength, path_name = 'pouet', savefig = True)

    """
    animate = mp.Animate2D(sim.sim,
                       fields=mp.Ez,
                       realtime=True,
                       field_parameters={'alpha':0.8, 'cmap':'RdBu', 'interpolation':'none'},
                       boundary_parameters={'hatch':'o', 'linewidth':1.5, 'facecolor':'y', 'edgecolor':'b', 'alpha':0.3})

    sim.sim.run(mp.at_every(5,animate), until = 800)
    N_fps = 7 #sets the number of frames per second for the movie. Here it has 40 frames, so let's just run at 5fps.
    animate.to_mp4(N_fps, 'surf_err.mp4')
    #sim.plot_efield()
    """

    

