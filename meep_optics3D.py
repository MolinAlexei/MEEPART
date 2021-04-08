import meep as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as sc
from mpi4py import MPI
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import h5py
import os
import glob
import csv

mp.verbosity(1)

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
    
    def set_size(self, size_x, size_y, size_z):
        """
        Defines the size of the system:
            - size_x is the size along the optical axis
            - size_y 
            - size_z 
        """
        
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
    
    def add_component(self, component):
        """
        Adds a component to the list of components, these components can be :
            - Lenses
            - Aparture Stop (x1)
            - Image Plane
            - Telescope Tube
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
        mid_z = np.int(self.size_z*resolution/2)

        thick = component.thick*resolution

        #Generate the center of the lumps made by delamination, 
        #different for the left and right surface
        high_y = np.int(np.around(component.diameter*0.9/2))
        y0_left = np.random.randint(low = -high_y, high = high_y)
        y0_right = np.random.randint(low = -high_y, high = high_y)

        max_z_L = np.sqrt((component.diameter/2)**2 - y0_left**2)
        high_z_L = np.int(np.around(max_z_L))
        z0_left = np.random.randint(low = -high_z_L, high = high_z_L)


        max_z_R = np.sqrt((component.diameter/2)**2 - y0_right**2)
        high_z_R = np.int(np.around(max_z_R))
        z0_right = np.random.randint(low = -high_z_R, high = high_z_R)

        for y_res in range(mid_y) :

            #Above and below the optical axis :
            y_positive = self.dpml*resolution + mid_y + y_res
            y_negative = self.dpml*resolution + mid_y - y_res

            for z_res in range(mid_z) :   

                rho = np.sqrt(y_res**2 + z_res**2)         
            
                #Left surface sag
                x_left = np.int(np.around((
                        component.left_surface(rho/resolution) + 
                        component.x + self.dpml)*resolution))
                #Right surface sag       
                x_right = np.int(np.around((
                        component.right_surface(rho/resolution) + 
                        component.x + self.dpml)*resolution + 
                        thick))
            
                #Above and below the optical axis :
                z_positive = self.dpml*resolution + mid_z + z_res
                z_negative = self.dpml*resolution + mid_z - z_res

                #Get the delamination as a function of y on left surface
                delam_ypos_zpos_L = np.int(np.around(resolution*
                    component.delamination(y_res/resolution, y0_left, z_res/resolution, z0_left)))
                delam_yneg_zpos_L = np.int(np.around(resolution*
                    component.delamination(-y_res/resolution, y0_left, z_res/resolution, z0_left)))
                delam_ypos_zneg_L = np.int(np.around(resolution*
                    component.delamination(y_res/resolution, y0_left, -z_res/resolution, z0_left)))
                delam_yneg_zneg_L = np.int(np.around(resolution*
                    component.delamination(-y_res/resolution, y0_left, -z_res/resolution, z0_left)))

                #Get the delamination as a function of y on right surface
                delam_ypos_zpos_R = np.int(np.around(resolution*
                    component.delamination(y_res/resolution, y0_right, z_res/resolution, z0_right)))
                delam_yneg_zpos_R = np.int(np.around(resolution*
                    component.delamination(-y_res/resolution, y0_right, z_res/resolution, z0_right)))
                delam_ypos_zneg_R = np.int(np.around(resolution*
                    component.delamination(y_res/resolution, y0_right, -z_res/resolution, z0_right)))
                delam_yneg_zneg_R = np.int(np.around(resolution*
                    component.delamination(-y_res/resolution, y0_right, -z_res/resolution, z0_right)))

                material_line = component.material

                #Gradient in the index
                radial_slope = component.radial_slope/resolution
                axial_slope = component.axial_slope/resolution
                n0 = component.material
                x0 = np.int(np.around(component.x*resolution))
                x_range = range(x_left, x_right+1) 

                material_line = [n0 + rho*radial_slope + (k-x0)*axial_slope for k in x_range]

                if rho > component.diameter*resolution/2:
                    material_line = 1

                #Write lens between left and right surface below optical axis
                epsilon_map[x_left:x_right+1, y_negative, z_negative] *= material_line

                if z_res != 0 : 
                    epsilon_map[x_left:x_right+1, y_negative, z_positive] *= material_line
            
                #So that the center line is not affected twice :
                if y_res != 0 :
                    #Write lens between left and right surface above optical axis
                    epsilon_map[x_left:x_right+1, y_positive, z_negative] *= material_line

                    if z_res != 0 : 
                        epsilon_map[x_left:x_right+1, y_positive, z_positive] *= material_line
            
                #Write AR coating on left surface
                if component.AR_left is not None :

                    AR_thick = np.int(np.around(component.AR_left*resolution))

                    epsilon_map[x_left - AR_thick - delam_yneg_zneg_L: x_left - delam_yneg_zneg_L, 
                            y_negative, z_negative] *= component.AR_material

                    if z_res != 0 : 
                        epsilon_map[x_left - AR_thick - delam_yneg_zpos_L: x_left - delam_yneg_zpos_L, 
                            y_negative, z_positive] *= component.AR_material


                    if y_res != 0 :
                        epsilon_map[x_left - AR_thick - delam_ypos_zneg_L: x_left - delam_ypos_zneg_L, 
                                y_positive, z_negative] *= component.AR_material

                        if z_res != 0 :
                            epsilon_map[x_left - AR_thick - delam_ypos_zpos_L: x_left - delam_ypos_zpos_L, 
                                y_positive, z_positive] *= component.AR_material
            
                #Write AR coating on right surface                    
                if component.AR_right is not None :
                
                    AR_thick = np.int(np.around(component.AR_right*resolution))

                    epsilon_map[x_right + 1 + delam_yneg_zneg_R: AR_thick + x_right + 1 + delam_yneg_zneg_R, 
                            y_negative, z_negative] *= component.AR_material

                    if z_res != 0 :
                        epsilon_map[x_right + 1 + delam_yneg_zpos_R: AR_thick + x_right + 1 + delam_yneg_zpos_R, 
                            y_negative, z_positive] *= component.AR_material

                    if y_res != 0 :
                        epsilon_map[x_right + 1 + delam_ypos_zneg_R: AR_thick + x_right + 1 + delam_ypos_zneg_R, 
                                y_positive, z_negative] *= component.AR_material

                        if z_res != 0 :
                            epsilon_map[x_right + 1 + delam_ypos_zpos_R: AR_thick + x_right + 1 + delam_ypos_zpos_R, 
                                y_positive, z_positive] *= component.AR_material
            
            
    
    def assemble_system(self, resolution = 1, dpml = None):
        
        """
        Creates the map that will be read by the simulation later, as well as
        the geometry objects necessary for an absorbing aperture stop and an 
        image plane
        
        Inputs : 
            - resolution : defines how many points/unit of distance, a higher 
            res gives better precision but also longer coomputation
            - dpml : thickness of the outer absorbing layer
        """
        
        self.resolution = resolution
        self.dpml = dpml
        
        # Define the map size, so that the PML is outside of the working system
        shape_x = (self.size_x + 2*dpml)*resolution+1
        shape_y = (self.size_y + 2*dpml)*resolution+1
        shape_z = (self.size_z + 2*dpml)*resolution+1

        epsilon_map = np.ones((shape_x, shape_y, shape_z), dtype = 'float32')
        #epsilon_map = [0,0,0]
                
        #Goes through all the components to add them to the system
        for component in self.components:
            
            #The action is decided with the component type
            
            ### LENSES
            if component.object_type == 'Lens':
                
                #print('baguette')
                self.write_lens(component, epsilon_map, resolution)

            elif component.object_type == 'Tube':
                
                R2 = (component.diameter*resolution/2)**2
                n = component.dielec_perm**2

                for k in range(len(epsilon_map[0,:,0])):
                    for j in range(len(epsilon_map[0,0,:])):
                        y = k - (self.size_y + 2*dpml)*resolution/2
                        z = j - (self.size_z + 2*dpml)*resolution/2
                        if y**2 + z**2 >= R2:
                            epsilon_map[:,k,j] *= n + 10000*1j

                
     
            ### APERTURE STOP
            elif component.object_type == 'AP_stop':
                
                #The aperture can be done with 2 blocks in 2D, as follows :
                              
                c1 = mp.Block(size=mp.Vector3(component.thick, 
                                              self.size_y + 2*dpml, 
                                              self.size_z + 2*dpml),
                              
                      center=mp.Vector3(component.x - self.size_x/2, 0, 0),
                      
                      material = mp.Medium(epsilon=component.permittivity, 
                                           D_conductivity = component.conductivity))
        
                c2 = mp.Cylinder(radius = component.diameter/2,
                    axis = mp.Vector3(1,0,0),                              
                    center=mp.Vector3(component.x - self.size_x/2, 0, 0),
                    material = mp.air, 
                    height = component.thick) 
                
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
                c1 = mp.Block(size=mp.Vector3(component.thick, component.side_size, component.side_size),
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

        #h = h5py.File('epsilon_map3D.h5', 'w', driver ='mpio', comm=MPI.COMM_WORLD)
        #dset = h.create_dataset('eps', (shape_x, shape_y, shape_z), dtype = 'float32')
        #with dset.collective :
        #    dset[:,:,:] = epsilon_map
        #h.close()


    def make_lens_bubbles(self, radius, nb_clusters, nb_per_cluster):

        """
        Introduces clusters of air bubbles inside the lenses of the system, 
        each cluster has a central bubble and a number of smaller bubble gathered
        around this central bubble
        
        Inputs : 
            - radius : radius of the central bubble
            - nb_clusters : number of clusters per lens
            - nb_per_cluster : number of bubbles surrounding the main one in each
            clusters
        Affects the self.permittivity_map object.
        """

        res = self.resolution

        #Function, given a radius, that returns the indices of the points within 
        #the circle centered on (0,0,0)
        def bubble(rad):
            bubble = []
            for k in range(-rad, rad+1):
                for j in range(-rad, rad+1):
                    for i in range(-rad, rad+1):
                        r = k**2 + j**2 +i**2 - rad**2
                        if np.around(r) <= 0 :
                            bubble.append([k,j,i])
            return np.array(bubble) 

        #List of centers of bubbles
        list_centers = []

        #List of radii of bubbles
        list_radii = []


        #Iterate for all lenses
        for component in self.components:

            if component.object_type == 'Lens':

                D = component.diameter
                middle_y = self.dpml + self.size_y/2
                middle_z = self.dpml + self.size_z/2

                #Lens thickness
                thick = component.thick*res

                #So that the bubbles aren't generated on the very edge of the lenses
                low  = np.int(np.around(middle_y*res - D*res*0.8/2))
                high = np.int(np.around(middle_y*res + D*res*0.8/2))

                #Iterate over cluster numbers
                for i in range(nb_clusters):

                    #The center of the bubble can be anywhere on the y axis
                    y0 = np.random.randint(low = low, high = high)

                    #The center of the bubble must be in the radius the lens
                    max_z  = np.sqrt((D*0.8*res/2)**2 - (y0-middle_y*res)**2)
                    high_z = np.int(np.around(middle_z*res + max_z))
                    low_z  = np.int(np.around(middle_z*res - max_z))
                    z0 = np.random.randint(low = low_z, high = high_z+1)

                    rho = np.sqrt((y0-middle_y*res)**2 + (z0-middle_z*res)**2)         
            
                    #Left surface sag
                    x_left = np.int(np.around((
                        component.left_surface(rho/res) + 
                        component.x + self.dpml)*res +
                        0.2*thick))
                    #Right surface sag       
                    x_right = np.int(np.around((
                        component.right_surface(rho/res) + 
                        component.x + self.dpml)*res + 
                        0.8*thick))

                    #The center of the cluster has to be inside the lens
                    x0 = np.random.randint(low = x_left, high = x_right+1)
                    #x_right + 1 so that low != high

                    #Radius of the main bubble can vary by 10 percent
                    radius_0 = radius*(0.9 + np.random.random()*0.2)
                
                    #Update lists
                    list_centers.append([x0,y0,z0])
                    list_radii.append(radius_0)

                    #Iterate over the number of surrounding bubbles
                    for k in range(nb_per_cluster):

                        #The center of each surrounding bubble is random, within
                        #a certain distance of the central bubble
                        theta = np.random.random()*np.pi
                        phi = np.random.random()*2*np.pi
                        r = radius_0*(1 + np.random.random()*3)

                        #Change of variables
                        x_k = np.int(np.around(r*np.cos(phi)*np.sin(theta)*res))
                        y_k = np.int(np.around(r*np.sin(phi)*np.sin(theta)*res))
                        z_k = np.int(np.around(r*np.cos(theta)*res))

                        #The radius is a function of distance, the farther the 
                        #smaller
                        radius_k = radius_0*np.exp(-r/(3*radius_0))*np.random.random()

                        #Update lists
                        list_centers.append([x0+x_k, y0+y_k, z0+z_k])
                        list_radii.append(radius_k)

        list_centers = np.array(list_centers)
        list_radii = np.array(list_radii)
        list_all = []

        #Making bubbles for all centers and radii
        for k in range(len(list_centers)):
            radius_k = np.int(np.around(list_radii[k]*res))
            bubble_k = bubble(radius_k)
            for u in bubble_k : 
                list_all.append(list_centers[k] + u)

        #Update the map
        for index in list_all : 
            self.permittivity_map[index[0], index[1], index[2]] = 1 
            #Permittivity of air is 1

    def add_tube(self, component):
        c1 = mp.Block(size=mp.Vector3(self.size_x, 
                                              self.size_y, 
                                              self.size_z),
                              
                      center=mp.Vector3(0, 0, 0),
                      
                      material = mp.Medium(epsilon=1, 
                                           D_conductivity = component.metal_cond))

        c2 = mp.Block(size=mp.Vector3(self.size_x, 
                                        self.size_y-10, 
                                        self.size_z-10),
                              
                      center=mp.Vector3(0, 0, 0),
                      material = mp.Medium(epsilon=1.2**2, 
                                           D_conductivity = 100))
                #R2 = (component.diameter*resolution/2)**2
                #n = component.dielec_perm**2

                #for k in range(len(epsilon_map[0,:,0])):
                #    for j in range(len(epsilon_map[0,0,:])):
                #        y = k - (self.size_y + 2*dpml)*resolution/2
                #        z = j - (self.size_z + 2*dpml)*resolution/2
                #        if y**2 + z**2 >= R2:
                #            epsilon_map[:,k,j] *= n

        eps = self.permittivity_map - 1 

        mat_array = mp.MaterialGrid(mp.Vector3(eps.shape[0],
                                                eps.shape[1],
                                                eps.shape[2]),
                            mp.air,
                            mp.Medium(epsilon=1.2**2),
                            design_parameters=eps,
                            grid_type='U_SUM')

        c2_bis = mp.Block(center=mp.Vector3(),
                     size=mp.Vector3(self.size_x, 
                                        self.size_y-10, 
                                        self.size_z-10),
                     material=mat_array)

        #c3 = mp.Cylinder(radius = component.diameter/2,
        #            axis = mp.Vector3(1,0,0),                              
        #            center=mp.Vector3(0, 0, 0),
        #            material = mat_array, 
        #            height = self.size_x) 
                
        if self.geometry is not None :
                    #If there are already objects in geometry, adds the aperture
                    #instead of replacing what was there
                    
                self.geometry.append(c1)
                self.geometry.append(c2_bis)
                #self.geometry.append(c3)
                    
        else :
                self.geometry = [c1,c2_bis]
            
    def plot_lenses(self, transverse_plot = True):
        #Only plots the lenses, allows to check their dispostion and shape
        
        plt.figure(figsize = (15,15))
        mid_z = np.int((self.size_z/2+self.dpml)*self.resolution)
        plt.imshow(self.permittivity_map[:,:,mid_z].real.transpose())
        plt.savefig('lenses')
        plt.close()

        
    def write_h5file(self):
        #Writes the file that will then be read within the sim function
        #If not running parallel, remove : 'w', driver ='mpio', comm=MPI.COMM_WORLD) 

        comm = MPI.COMM_WORLD
        rank = comm.rank

        h = h5py.File('epsilon_map3D.h5', 'w', driver ='mpio', comm=MPI.COMM_WORLD)
        #h = h5py.File('epsilon_map3D.h5', 'w')
        size_x = len(self.permittivity_map[:,0,0])
        size_y = len(self.permittivity_map[0,:,0])
        size_z = len(self.permittivity_map[0,0,:])
        dset = h.create_dataset('eps', (size_x, size_y, size_z), dtype = 'float32', compression = "gzip")
        with dset.collective :
            dset[:,:,:] = self.permittivity_map
        
        h.close()
        self.permittivity_map = 0

    def delete_h5file(self):
        #Deletes the h5 file, can be useful when the file is heavy and not to 
        #be kept after simulation
        
        file = glob.glob('epsilon_map3D.h5')
        #os.remove(file)
        
    def sys_info(self, wavelength = None, frequency = None):

        c = 299792458.0
        if wavelength is not None :
           frequency = 1/wavelength

        if frequency is not None:
            wavelength = 1/frequency

        print('System size = {} x {} wavelengths'.format(self.size_x/wavelength, 
                                                        self.size_y/wavelength))

    def material_function(self, x, y, z):

        dpml = self.dpml
        for component in self.components:

            if component.name == 'Lens 1' and x <350:
                
                return component.obj_func(x, y, z, dpml)

            elif component.name == 'Lens 2' and x>=350 :

                return component.obj_func(x, y, z, dpml)

                
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
                 x=0., diameter = None, 
                 n_refr = 1.52, 
                 AR_left = None, AR_right = None,
                 delam_thick = 0,
                 delam_width = 10,
                 radial_slope = 0,
                 axial_slope = 0):
        
        self.name = name                #NAME OF LENS  
        self.r1 = r1                    #LEFT SURFACE RADIUS
        self.r2 = r2                    #RIGHT SURFACE RADIUS
        self.c1 = c1                    #LEFT SURFACE ASPHERIC PARAMETER
        self.c2 = c2                    #RIGHT SURFACE ASPHERIC PARAMETER
        self.thick = thick              #THICKNESS AT CENTER
        self.x = x                      #X POSITION OF LEFT SURFACE CENTER
        self.diameter = diameter        #DIAMETER OF THE LENS
        self.radius = diameter/2        #RADIUS SQUARED
        self.material = n_refr**2       #DIELECTRIC PERMITTIVITY
        self.object_type = 'Lens'

        self.AR_left = AR_left          #LEFT AR COATING THICKNESS
        self.AR_right = AR_right        #RIGHT AR COATING THICKNESS
        self.AR_material = n_refr       #AR COATING PERMITTIVITY
        self.delam_thick = delam_thick  #AR COATING DELAMINATION THICKNESS
        self.delam_width = delam_width  #DELAMINATION LUMP THICKNESS

        self.radial_slope = radial_slope#RADIAL GRADIENT IN THE INDEX
        self.axial_slope = axial_slope  #AXIAL GRADIENT IN THE INDEX


    
    def left_surface(self, rho):
        """
        Aspheric lens equation

        Parameters
        ----------
        rho : STR OR LIST
            Distance from optical axis at which the sag is computed

        Returns
        -------
        STR OR LIST
            Sag at at distance y from optical axis.

        """

        if rho <= self.diameter/2 : 
        
            if self.r1 != np.inf :
                return (rho**2/self.r1) / (1 + np.sqrt(1 - (1+ self.c1)*rho**2/self.r1**2))
            else : 
                #If the radius is infinite, returns a flat surface, i.e. 0 sag
                return 0

        else :
            return 0
    
    def right_surface(self, rho):
        """
        Same as left_surface(self,rho)
        """
        if rho < self.diameter/2 : 
            if self.r2 != np.inf :
                return (rho**2/self.r2) / (1 + np.sqrt(1 - (1+ self.c2)*rho**2/self.r2**2))
            else : 
                #If the radius is infinite, returns a flat surface, i.e. 0 sag
                return 0

        elif rho >= self.diameter/2 :
            return 0


    def delamination(self, y, y0, z, z0):

        #Returns the air layer thickness that makes delamination, it is 
        #zero everywhere excpet where there's the lump, centered on y0, defined by
        #its width and thickness

        thick = self.delam_thick
        width = self.delam_width
        lump = (((y-y0)/width)**2 + ((z-z0)/width)**2 - 2)*thick

        return np.abs(min((lump, 0)))

    def obj_func(self, x, y,z, dpml):

        rho = np.sqrt(y**2 + z**2)
        left = self.left_surface(rho) + self.x + dpml
        right = self.right_surface(rho) + self.x + dpml + self.thick

        if rho < self.radius :
            if x > left and x < right :
                return mp.Medium(epsilon = self.material)

            return mp.air

        return mp.air


        

    
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
                 side_size = None, 
                 pos_x = None, 
                 thickness = None, 
                 n_refr  = None, 
                 conductivity = None):
        
        self.name = name                #NAME OF IMAGE PLANE
        self.thick = thickness          #THICKNESS OF IMAGE PLANE
        self.x = pos_x                  #POSITION ON OPTICAL AXIS
        self.side_size = side_size        #DIAMETER OF IMAGE PLANE
        
        if conductivity != np.inf :
            #Defines the material with given properties
            self.material = mp.Medium(epsilon=n_refr**2, 
                                      D_conductivity = conductivity)
        
        else :
            #If the conductivity is infinite, Meep can define a perfect conductor
            self.material = mp.perfect_electric_conductor
        
        self.object_type = 'ImagePlane'

class TelescopeTube(object):
    """
    Defines the tube of the telescope
    """

    def __init__(self, name = '',
                 diameter = None, 
                 dielec_perm  = None, 
                 metal_cond = None):
        
        self.name = name                  #NAME OF IMAGE PLANE
        self.diameter = diameter          #DIAMETER OF TUBE
        self.dielec_perm = dielec_perm    #PERMITTIVITY OF DIELEC LAYER
        self.metal_cond = metal_cond      #CONDUCTIVITY OF METAL ENVELOPE
        self.object_type = 'Tube'


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
        self.cell = mp.Vector3(self.opt_sys.size_x+2*dpml, self.opt_sys.size_y+2*dpml, self.opt_sys.size_z+2*dpml)

    

    def define_source(self, frequency = None,
                      wavelength = None, 
                      sourcetype = 'Plane wave', 
                      x = 0, y = 0, z =0, 
                      size_x = 0, size_y = 300,  size_z = 300,
                      beam_width = 0, 
                      focus_pt_x = 0, focus_pt_y = 0, focus_pt_z = 0,
                      fwidth = 0):

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
        size_z : FLOAT, optional
            z-size of the source. The default is 300.
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
                           center=mp.Vector3(x_meep, y, z),
                           size=mp.Vector3(size_x, size_y, size_z))]
        
        elif sourcetype == 'Gaussian beam':
            self.source = [mp.GaussianBeamSource(mp.ContinuousSource(frequency),
                      component = mp.Ez,
                      center = mp.Vector3(self.opt_sys.image_plane_pos-2, y, z),
                      beam_x0 = mp.Vector3(focus_pt_x, focus_pt_y, focus_pt_z),
                      beam_kdir = mp.Vector3(-1, 0, 0),
                      beam_w0 = beam_width,
                      beam_E0 = mp.Vector3(0,0,1),
                      size=mp.Vector3(size_x, size_y, size_z))]
        """
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
        """
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
            dpml = np.int(np.around(0.5*self.wavelength))
        
        
        self.PML(dpml)
        self.dpml = dpml
        self.sim_resolution = sim_resolution


        ###Function of material
        cell_middle = (self.opt_sys.size_x/2 + dpml)
        def matfunc(P):
            x = P.x + cell_middle
            y = P.y
            z = P.z 
            return self.opt_sys.material_function(x, y, z)
            #return mp.air

        matfunc.do_averaging = True    
        

        #Defines the simulation environment, using the various objects defined
        #previously
        self.sim = mp.Simulation(cell_size=self.cell,
                    boundary_layers=self.pml_layers,
                    geometry=self.opt_sys.geometry, 
                    sources=self.source,
                    resolution=self.sim_resolution,
                    #subpixel_tol = 0.1,
                    #subpixel_maxeval = 10,
                    #eps_averaging = False,
                    #material_function = matfunc)
                    epsilon_input_file = 'epsilon_map3D.h5:eps')     
        #Runs the sim
        self.sim.run(until = runtime)

    

    def plot_system(self):
        
        #Makes a plot of the various objects in the computational cell, with 
        #the objects in grey and the PML in purple.
        
        mid_z = np.int(self.opt_sys.size_z*self.opt_sys.resolution/2)
        eps_data = self.sim.get_array(center=mp.Vector3(0,0,0), 
            size=mp.Vector3(self.opt_sys.size_x+ 2*self.dpml, self.opt_sys.size_y+2*self.dpml,  0), 
            component=mp.Dielectric)

        dpml_res = np.int(np.around(self.dpml*self.sim_resolution))
        pml = np.zeros((eps_data.transpose().shape))
        pml[0: dpml_res, :] = 1
        pml[:, 0: dpml_res] = 1
        pml[:, -dpml_res : ] = 1
        pml[-dpml_res: , :] = 1
        plt.figure(figsize = (12,12))
        plt.imshow(eps_data.transpose(), 
            extent = (0,self.opt_sys.size_x+2*self.dpml,0,self.opt_sys.size_y+2*self.dpml))

        #plt.imshow(pml, cmap = 'Purples', alpha = 0.4)
        plt.xlabel('x times resolution')
        plt.ylabel('y times resolution')
        plt.savefig('system_plot')
        plt.close()
        
    def plot_efield(self, path = '.'):

        #Makes a plot of the Ez component of the electric field in the system
        
        eps_data = self.sim.get_array(center=mp.Vector3(0,0,0), 
            size=mp.Vector3(self.opt_sys.size_x, self.opt_sys.size_y,  0), 
            component=mp.Dielectric)
        ez_data = self.sim.get_array(center=mp.Vector3(0,0,0), 
            size=mp.Vector3(self.opt_sys.size_x, self.opt_sys.size_y,  0), 
            component=mp.Ez)
        ey_data = self.sim.get_array(center=mp.Vector3(0,0,0), 
            size=mp.Vector3(self.opt_sys.size_x, self.opt_sys.size_y,  0), 
            component=mp.Ey)
        plt.figure(figsize = (20,20))
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha = 0.9)
        plt.xlabel('x times resolution')
        plt.ylabel('y times resolution')
        plt.savefig('{}/ez_field'.format(path))
        plt.show()

        plt.figure(figsize = (20,20))
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.imshow(ey_data.transpose(), interpolation='spline36', cmap='RdBu', alpha = 0.9)
        plt.xlabel('x times resolution')
        plt.ylabel('y times resolution')
        plt.savefig('{}/ey_field'.format(path))
        plt.show()


    def plot_beam(self, single_plot = True, colors = ['r'], plot_n = 0, linestyle = '-',
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

        if single_plot :
            #Displays the plot if single_plot is True
            plt.figure()
        
        #Setting the timestep to a very low value, so that MEEP uses its lowest timestep
        timestep = .3

        #50 steps Is roughly enough to give a few periods for wavelengths from 1 to 10
        n_iter = 40
        
        #Get the real field at aperture
        efield = self.sim.get_array(center=mp.Vector3(-self.opt_sys.size_x/2+aper_pos_x, 0, 0), 
                                         size=mp.Vector3(0, self.opt_sys.size_y, self.opt_sys.size_z), component=mp.Ez)

        res = self.sim_resolution

        #Initializes the list containing the E field evolution
        e_field_evol = np.ones((n_iter, len(efield), len(efield[0])))
        e_field_evol[0] = efield

        #List to get the precise timestepping done by meep
        time = np.zeros(n_iter)
        time[0] = self.sim.meep_time()

        ### Stacking the electric field evolution on the aperture
        for k in range(n_iter):
            self.sim.run(until = timestep)
            time[k] = self.sim.meep_time()
            e_field_evol[k] = self.sim.get_array(center=mp.Vector3(-self.opt_sys.size_x/2+aper_pos_x, 0, 0), 
                                         size=mp.Vector3(0, self.opt_sys.size_y, self.opt_sys.size_z), component=mp.Ez)
        
        #Each point on the aperture is fit for a cosine with amplitude and phase
        def f(x, amp, phase):
            return amp*np.cos(x*2*np.pi/self.wavelength + phase)

        #Initialize the lists of amplitude and phase over the aperture

        n_zeros = np.int(np.around(aperture_size*res))
        amplitude = np.zeros((n_zeros, n_zeros))
        phase = np.zeros((n_zeros, n_zeros))

        #The field is only taken on the opening of the aperture
        idx = np.int(res*(self.opt_sys.size_y - aperture_size)/2)

        middle_y = np.int(self.opt_sys.size_y*res/2)
        middle_z = np.int(self.opt_sys.size_y*res/2)

        R2 = (aperture_size*res/2)**2
        #Fits amplitude and phase for each point
        for k in range(n_zeros):
            for i in range(n_zeros):

                y = idx+k-middle_y
                z = idx+i-middle_z
                if y**2 + z**2 <= R2 :
                    popt, pcov = sc.curve_fit(f, time, e_field_evol[:, idx+k, idx + i])
                    amplitude[k,i] = popt[0]
                    phase[k,i] = popt[1]

        ### Plot
        plt.figure()
        for k in range(len(amplitude[0])) :
            plt.plot(np.arange(len(amplitude[:,k]))/self.sim_resolution, 
                    amplitude[:,k]) 
                    #color = colors[plot_n],
                    #alpha = .9,
                    #linestyle = linestyle) 
        plt.title('E field amplitude on aperture, $\lambda = ${:2.1} mm'.format(self.wavelength))
        plt.xlabel('y (mm)')
        plt.ylabel('$E^2$')
        plt.savefig('amplitude3D')

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
                        y_max = 0., Nb_sources = 1., sim_resolution = 0.5,
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
                                   x=self.sim.opt_sys.image_plane_pos, y = height, z= 0, 
                                   size_x = 0, size_y = 10, size_z = 10, 
                                   beam_width = 10)
            
            #Runs the sim
            self.sim.run_sim(runtime = runtime, sim_resolution = sim_resolution)

            #Gets the complex electric field and adds it to the plot
            E_field = self.sim.plot_beam(single_plot = False, colors= colors, plot_n = k, linestyle = linestyle)

            #Updates the list of fields
            self.list_efields[k] = E_field
        
    
    def beam_FT(self, aperture_size = 200, precision_factor = 5):

        """
        Gets the Fourier Transforms of the complex electric fields at aperture.
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

        self.FWHM_fft = [0 for k in range(len(self.list_efields))]

        #List of frequencies
        freq = np.fft.fftfreq(len(self.list_efields[0][0])*precision_factor, d = 1/res)

        corr_matrix = [np.ones(len(freq)) + 
                        (self.wavelength**2)*(freq**2+np.ones(len(freq))*freq[k]**2) for k in range(len(freq))]

        #Iterate over the number of sources
        for k in range(len(self.list_efields)):

            #FFT over the field
            fft = np.fft.fft2(self.list_efields[k], s = [len(freq),len(freq)])

            #Corrective factor for distance to the aperture
            beam = fft*np.conj(fft)*corr_matrix

            #FFT is normalized by its max
            FFTs[k] = beam.real
            FFTs[k] = FFTs[k]/np.max(FFTs[k])

        return freq, FFTs

if __name__ == '__main__':
    
    opt_sys = OpticalSystem('test')
    opt_sys.set_size(800,300,300)
    
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
                         delam_width = 10,
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
                         delam_width = 10,
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
                             n_refr = 1.2, 
                             conductivity = 0)
    
    tube = TelescopeTube(name = 'Tube', 
                        diameter = 300,
                        dielec_perm = 1.7,
                        metal_cond = 1e7)


    

    opt_sys.add_component(lens1)
    opt_sys.add_component(lens2)
    #opt_sys.add_component(tube)

    opt_sys.add_component(aperture_stop)
    opt_sys.add_component(image_plane)
    print(opt_sys.list_components())
    #opt_sys.sys_info(frequency = 0.5)
    
    wavelength = 10
    dpml = 5

    
    opt_sys.assemble_system(dpml = dpml, resolution = 1)
    #opt_sys.make_lens_bubbles(4,5,15)
    opt_sys.plot_lenses()
    opt_sys.write_h5file()

    #opt_sys.add_tube(tube)

    eps_0 = 1.2**2
    eps_e = 1.7**2

    diag = mp.Vector3(eps_0, (eps_0+eps_e)/2, (eps_0+eps_e)/2)
    offdiag = mp.Vector3(0,0, (eps_e-eps_0)/2)


    HWP = mp.Block(center = mp.Vector3(-320, 0, 0),
              size = mp.Vector3(10, 300, 300),
              material = mp.Medium(epsilon_diag = diag, 
                                    epsilon_offdiag = offdiag))
    opt_sys.geometry.append(HWP)
  
    sim = Sim(opt_sys)
    sim.define_source(wavelength = wavelength, sourcetype = 'Gaussian beam', 
                     x = 720, y = 0, size_x = 0, size_y = 300, size_z = 300, beam_width = 10)
    sim.run_sim(runtime = 850, sim_resolution = 1)
    sim.plot_system()
    sim.plot_efield()

    #sim.plot_beam()

    """
    analysis = Analysis(sim)  
    print(mp.count_processors())

    analysis.image_plane_beams(wavelength = wavelength, fwidth = 0, sourcetype='Gaussian beam',
                                    y_max = 100, Nb_sources = 1, sim_resolution = 0.5) 

    freq, fft = analysis.beam_FT(aperture_size = 200, precision_factor = 5)
    degrees = np.arctan(freq*wavelength)*180/np.pi

    print(np.shape(degrees), np.shape(fft))

    fft_dB = 10*np.log10(np.abs(fft[0]))
    """


    """ 
    #### 3D PLOTTTING #####

    #fig = plt.figure(figsize = (10,10))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax = fig.add_subplot(111, projection='3d')

    
    X = [degrees for k in range(len(degrees))]
    Y = [np.ones(len(degrees))*degrees[k] for k in range(len(degrees))]

    surf = ax.plot_surface(X, Y, fft_dB, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    #ax.set_xlim((-15,15))
    #ax.set_ylim((-15,15))

    # Customize the z axis.
    #ax.set_zlim(-50, 0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    #plt.xlim((-10,10))
    #plt.ylim((-10,10))
    plt.savefig('beam3D')
    plt.close()
    """

    """
    X = [degrees for k in range(len(degrees))]
    Y = [np.ones(len(degrees))*degrees[k] for k in range(len(degrees))]

    middle_idx = np.int(len(fft_dB)/2)
    fft_dB2 = np.concatenate((fft_dB[middle_idx:,:], fft_dB[:middle_idx,:]), axis = 0)
    fft_reshaped = np.concatenate((fft_dB2[:, middle_idx:], fft_dB2[:, :middle_idx]), axis = 1)

    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    im = ax.imshow(fft_reshaped, extent = (-90,90,-90,90))
    cbar = plt.colorbar(im)
    cbar.ax.set_xlabel('Power')
    c = ax.contour(X,Y,fft_dB, cmap=plt.get_cmap('Greys'),levels=[-20, -13, -10, -3], vmin = -20, vmax = -3)
    ax.set_xlim((-10,10))
    ax.set_ylim((-10,10))
    plt.savefig('Beam_contour')
    plt.close()
    """


    """
    ### PLOT CROSS SECTION OF BEAM AT CENTER
    plt.figure()
    plt.plot(degrees, fft_dB[:,0])
    plt.plot(degrees, fft_dB[0,:])
    fwhm = wavelength*180/(200*np.pi)

    plt.vlines([-fwhm/2, fwhm/2], -100, 0, color = 'grey', linestyle = 'dashdot')
    plt.hlines(-3, -40,40)
    plt.xlim((-40,40))
    plt.ylim((-40,0))
    plt.savefig('3Dbeamslice')
    """
