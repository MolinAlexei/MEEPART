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


class OpticalSystem(object):
    '''
    This class is used to define the optical system, by creating the dielectric
    map associated to the system that can the be used within the simulation 
    class.
    '''
    
    def __init__(self, name=''):
        
        '''
        Sets the size of the optical system,
        the list of components and list of
        geometry objects.

        Arguments
        ---------
        name : str, optional
            Name of the system (default : '')
        '''
        
        self.name = name
        self.geometry = []
        self.components = []
        
    def __str__(self):
        '''
        Returns the name of the system.
        '''
        return self.name
    
    def set_size(self, size_x, size_y):
        '''
        Sets the size of the optical system
        ---------
        size_x : int
            Size of system on x-axis
        size_y : int
            Size of system on y-axis
        Notes
        -----
        The system is always defined to have 
        the optical axis along the x axis.
        '''
        
        self.size_x = size_x
        self.size_y = size_y
    
    def add_component(self, component):
        '''
        Adds the component to the list of objects in the optical system
        ---------
        component : component
            Component object
        '''
        
        self.components.append(component)
        
    def list_components(self):
        '''
        Returns the list of components by their names.
        '''
        print('---Components---')
        for component in self.components:
            print(component.name)
        print('----------------')
    
    def write_lens(self, comp, eps_map, res):
        '''
        The lens equation returns a sag (distance from plane orth. to
        optical axis) as a function of distance from optical axis y,
        so the code cycles through the different y to change the 
        dielectric map between left surface and right surface
        ---------
        comp : component
            Lens component object
        eps_map : 2D or 3D array
            Dielectric map on which the lens will be written
        res : float
            Resolution of map
        '''

        # The y axis has its zero in the middle of the cell, the offset
        # is mid_y
        mid_y = np.int(self.size_y*res/2)

        #Thickness of the lens on optical axis
        thick = comp.thick*res

        #Generate the center of the lumps made by delamination, 
        #different for the left and right surface
        high = np.int(np.around(self.size_y*0.9/2))
        y0_left = np.random.randint(low = -high, high = high)
        y0_right = np.random.randint(low = -high, high = high)

        radius = np.int(comp.diameter*res/2)

        #Generates the bins of random surface errors.
        if comp.surf_err_scale!=0 :
            nb_bins = int(comp.diameter/comp.surf_err_width)
            err_left = np.around(np.random.normal(scale = comp.surf_err_scale*res,
                                                  size = nb_bins))
            err_right = np.around(np.random.normal(scale = comp.surf_err_scale*res, 
                                                   size = nb_bins))

        if comp.surf_err_scale == 0:
            nb_bins = int(comp.diameter/comp.surf_err_width)
            err_left = np.zeros(nb_bins)
            err_right = np.zeros(nb_bins)
        
        #Iterates y over the radius, as the lenses are symmetric
        #abov and below the optical axis
        for y_res in range(radius) :           
            
            #Left surface sag
            x_left = np.int(np.around((
                        comp.left_surface(y_res/res) + self.dpml + 
                        comp.x - comp.cust_def((y_res+mid_y)/res))*res))
            #Right surface sag       
            x_right = np.int(np.around((
                        comp.right_surface(y_res/res) + 
                        comp.x + self.dpml -
                        comp.cust_def((y_res+mid_y)/res))*res + 
                        thick))
            
            #Above and below the optical axis :
            y_positive = self.dpml*res + mid_y + y_res
            y_negative = self.dpml*res + mid_y - y_res

            #Get the delamination as a function of y on left surface
            delam_pos_L = np.int(np.around(res*
                comp.delamination(y_res/res, y0_left)))
            delam_neg_L = np.int(np.around(res*
                comp.delamination(-y_res/res, y0_left)))

            #Get the delamination as a function of y on right surface
            delam_pos_R = np.int(np.around(res*
                comp.delamination(y_res/res, y0_right)))
            delam_neg_R = np.int(np.around(res*
                comp.delamination(-y_res/res, y0_right)))
            
            #Gradient in the index
            radial_slope = comp.radial_slope/res
            axial_slope = comp.axial_slope/res
            eps0 = comp.eps
            x0 = np.int(np.around(comp.x*res))
            x_range = range(x_left, x_right+1) 
            #The value is squared as the permittivity is index squared
            eps_line = [eps0 + 
                            (y_res*radial_slope)**2 + 
                            ((k-x0)*axial_slope)**2 for k in x_range]


            #Surface error
            err_bin_idx = int(np.around(y_res/res/comp.surf_err_width))
            err_left_pos = int(err_left[err_bin_idx])
            err_left_neg = int(err_left[- np.around(err_bin_idx)])

            err_right_pos = int(err_left[err_bin_idx]) 
            err_right_neg = int(err_left[- err_bin_idx])

            x_left_neg = x_left + err_left_neg
            x_left_pos = x_left + err_left_pos

            x_right_neg = x_right + err_right_neg
            x_right_pos = x_right + err_right_pos


            #Write lens between left and right surface below optical axis
            eps_map[x_left_neg : x_right_neg+1, y_negative] *= eps_line
            
            #So that the center line is not affected twice :
            if y_res != 0 :
                #Write lens between left and right surface above optical axis
                eps_map[x_left_pos : x_right_pos+1, y_positive] *= eps_line
            
            #Write AR coating on left surface
            if comp.AR_left is not None :

                AR_thick = np.int(np.around(comp.AR_left*res))

                eps_map[x_left_neg - AR_thick - delam_neg_L: x_left_neg - 
                        delam_neg_L, y_negative] *= comp.AR_material

                if y_res != 0 :
                    eps_map[x_left_pos - AR_thick - delam_pos_L: x_left_pos - 
                            delam_pos_L, y_positive] *= comp.AR_material
            
            #Write AR coating on right surface                    
            if comp.AR_right is not None :
                
                AR_thick = np.int(np.around(comp.AR_right*res))

                eps_map[x_right_neg + 1 + delam_neg_R: AR_thick + x_right_neg + 
                        1 + delam_neg_R, y_negative] *= comp.AR_material

                if y_res != 0 :
                    eps_map[x_right_pos + 1 + delam_pos_R: AR_thick + 
                            x_right_pos + 1 + delam_pos_R, 
                            y_positive] *= comp.AR_material
            
    def assemble_system(self, res, dpml):
        '''
        Creates the map that will be read by the simulation later, as well as
        the geometry objects necessary for an absorbing aperture stop and an 
        image plane

        Arguments
        -----------------
        res : float
            Defines how many points/unit of distance, a higher 
            res gives better precision but also longer coomputation
        dpml : float
            Perfectly Matched Layer thickness
        
        Notes
        -----
        The dpml is necessary to make a map that is bigger than the system,
        so that the PML is outside of the system size provided earlier.
        '''
        
        self.res = res
        self.dpml = dpml
        
        # Define the map size, so that the PML is outside of the working system
        epsilon_map = np.ones(((self.size_x + 2*dpml)*res+1, 
                               (self.size_y + 2*dpml)*res+1), dtype = 'float32') 
        
                
        #Goes through all the components to add them to the system
        for comp in self.components:
            
            #The action is decided with the comp type
            
            ### LENSES
            if comp.object_type == 'Lens':
                
                self.write_lens(comp, epsilon_map, res)
     
            ### APERTURE STOP
            elif comp.object_type == 'AP_stop':
                
                #The aperture can be done with 2 blocks in 2D, as follows :
                size = mp.Vector3(comp.thick, 
                                  (self.size_y - comp.diameter)/2 + dpml, 
                                  0)
                center_up = mp.Vector3(comp.x - self.size_x/2, 
                                (comp.diameter + self.size_y + 2*dpml)/4, 0)
                center_down = mp.Vector3(comp.x - self.size_x/2, 
                                -(comp.diameter + self.size_y + 2*dpml)/4, 0)

                up_part = mp.Block(size=size,
                                   center=center_up,
                                   material = mp.Medium(epsilon=comp.permittivity, 
                                           D_conductivity = comp.conductivity))
                
                down_part = mp.Block(size=size,
                                     center=center_down,
                                     material = mp.Medium(epsilon=comp.permittivity, 
                                           D_conductivity = comp.conductivity))

                self.aper_pos_x = comp.x
                self.aper_size = comp.diameter
                self.geometry.append(up_part)
                self.geometry.append(down_part)
            
            ### IMAGE PLANE
            elif comp.object_type == 'ImagePlane':
                
                #The image plane is just a single plane, made with a block :
                block = mp.Block(size=mp.Vector3(comp.thick, comp.diameter, 0),
                      center=mp.Vector3(comp.x - self.size_x/2, 0, 0),
                      material = comp.material)
                
                self.IP_pos = comp.x
                self.geometry.append(block)
                
            elif comp.object_type == 'MetallicTube':

                #The telescope tube is 2 blocks :
                up_part = mp.Block(size=mp.Vector3(self.size_x, comp.thick, 0),
                      center=mp.Vector3(0, comp.center, 0),
                      material = mp.metal)
                
                down_part = mp.Block(size=mp.Vector3(self.size_x, comp.thick, 0),
                      center=mp.Vector3(0, -comp.center, 0),
                      material = mp.metal)
                
                self.geometry.append(up_part)
                self.geometry.append(down_part)
                

            elif comp.object_type == 'Absorber':
                #The absorber is 2 blocks :
                up_part = mp.Block(size=mp.Vector3(self.size_x, comp.thick, 0),
                      center=mp.Vector3(0, comp.center, 0),               
                      material = mp.Medium(epsilon=comp.epsilon_real, 
                                            D_conductivity=comp.conductivity))


                down_part = mp.Block(size=mp.Vector3(self.size_x, comp.thick, 0),
                      center=mp.Vector3(0, -comp.center, 0),               
                      material = mp.Medium(epsilon=comp.epsilon_real, 
                                            D_conductivity=comp.conductivity))

                
                self.geometry.append(up_part)
                self.geometry.append(down_part)
                
            elif comp.object_type == 'HWP':
                HWP = mp.Block(center = mp.Vector3(comp.x_pos- self.size_x/2, 0, 0),
                                size = mp.Vector3(comp.thick, comp.size_y, 0),
                                material = mp.Medium(epsilon_diag = comp.diag, 
                                    epsilon_offdiag = comp.offdiag))

                self.geometry.append(HWP)

            """
            #THIS IS A TEMPORARY TEST
            elif comp.object_type == 'PrismsTube':

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
            """
            
        self.permittivity_map = epsilon_map

    def make_lens_bubbles(self, radius, nb_clusters, nb_per_cluster):
        '''
        Introduces clusters of air bubbles inside the lenses of the system, 
        each cluster has a central bubble and a number of smaller bubble gathered
        around this central bubble
        
        Arguments
        -----------------
        radius : float
            Radius of the central bubble
        nb_clusters : float
            Number of clusters per lens
        nb_per_cluster : 
            Number of bubbles surrounding the main one in each
            cluster
        Notes
        -----
        This function alters the permittivity map. 
        '''

        res = self.res

        #Function which, given a radius, that 
        #returns the indices of the points within 
        #the circle centered on (0,0)
        def bubble(rad):
            '''
            Introduces clusters of air bubbles inside the lenses of the system, 
            each cluster has a central bubble and a number of smaller bubble gathered
            around this central bubble
        
            Arguments
            -----------------
            rad : float
                Radius of the bubble

            Returns
            -------
            bubble : array
                Array of indexes within radius
            '''
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

                #So that the bubbles aren't generated 
                #on the very edge of the lenses
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
        '''
        Plots the permittivity map, where we can see only the lenses,
        allows to check their dispostion and shape
        '''
        extent = (0, 
                  len(self.permittivity_map[:])/self.res,
                  0,
                  len(self.permittivity_map[:][0])/self.res)
        plt.figure(figsize = (15,15))
        plt.title('Permittivity map')
        plt.imshow(self.permittivity_map.transpose(), extent = extent)
        plt.savefig('Lenses.png')
        plt.close()
        
    def write_h5file(self, parallel = False, filename = 'epsilon_map'):
        '''
        Writes the file that will then be 
        read within the MEEP simulation

        Arguments
        ---------
        parallel : bool, optional
            If the computation is run in parallel (default : False)
        filename : str, optional
            Name of the permittivity map file written. 
            Needs to be the same name given to the MEEP simulation
            (default : 'epsilon_map')
        '''
        self.mapname = filename
        if parallel : 
            comm = MPI.COMM_WORLD
            rank = comm.rank

            h = h5py.File(filename + '.h5', 'w', 
                          driver ='mpio', 
                          comm=MPI.COMM_WORLD)
        
            size_x = len(self.permittivity_map[:,0])
            size_y = len(self.permittivity_map[0,:])
            dset = h.create_dataset('eps', (size_x, size_y), 
                                    dtype = 'float32', 
                                    compression = "gzip")
            with dset.collective :
                dset[:,:] = self.permittivity_map
        
            h.close()

        elif not parallel :
            h = h5py.File(filename + '.h5', 'w')
            dset = h.create_dataset('eps', data = self.permittivity_map, 
                                           compression = "gzip")
            h.close()
                
    def sys_info(self, dist_unit, wvl = None, meep_freq = None, real_freq = None):
        '''
        Writes the file that will then be 
        read within the MEEP simulation

        Arguments
        ---------
        dist_unit : float
            Chosen ratio between MEEP distances and real distance 
        wvl : float, optional
            Wavelength in MEEP units (default : None)
        meep_freq : float, optional
            Frequency in MEEP units (default : None)
        real_freq : float, optional
            Frequency in Hz (default : None)
        '''

        c = 299792458.0
        if wvl is not None :
           meep_freq = 1/wvl
           real_freq = c*meep_freq/dist_unit

        if real_freq is not None:
            wvl = c/real_freq/dist_unit
            meep_freq = 1/wvl

        if meep_freq is not None: 
            wvl= 1/meep_freq
            real_freq = c*meep_freq/dist_unit

        print('--- System Info ---')
        print('Real Wavelength = {:.1e}m'.format(wvl*dist_unit))
        print('MEEP Wavelength = {:.1e}'.format(wvl))
        print('System size = {:.0f} x {:.0f} wavelengths'.format(self.size_x/wvl, 
                                                        self.size_y/wvl))
        print('System size = {:.2e} x {:.2e} m'.format(self.size_x*dist_unit, 
                                                        self.size_y*dist_unit))
        print('Real frequency = {:.2e} Hz'.format(real_freq))
        print('MEEP frequency = {:.2e}'.format(meep_freq))
        print('------------------')

class TelescopeTube(object):
    '''
    Class defining a telescope tube.
    '''
    def __init__(self, thick, center,name = None):      
        '''
        Defines the attributes of the telescope tube object

        Arguments
        ---------
        name : str, optional
            Name of object (default : None)
        thick : float
            Thickness of the tube walls
        center : float
            Center of the tube walls along the y-axis
            Needs to account for half a thickness
        '''
        self.name = name                        #NAME OF OBJECT
        self.object_type = 'MetallicTube'       #OBJECT TYPE
        self.thick = thick                      #TUBE THICKNESS
        self.center = center                    #DISTANCE OF CENTER FROM OPTICAL AXIS

    def __str__(self):
        if self.name is not None :
            return self.name + ', thickness ' + str(self.thick)  
        else :
            return 'Metallic tube, thickness ' + str(self.thick) 

class Absorber(object):
    '''
    Class defining a wall made of absorbing material.
    '''
    def __init__(self, thick, center, name = None, 
                epsilon_real = 3.5, epsilon_imag = 0.11025, freq = 1/3):
        '''
        Defines the attributes of the absorber object

        Arguments
        ---------
        name : str, optional
            Name of object (default : None)
        thick : float
            Thickness of the tube walls 
        center : float
            Center of the tube walls along the y-axis
            Needs to account for half a thickness
        epsilon_real : float, optional 
            Real part of the permittivity of the material (default : 3.5)
        epsilon_imag : float, optional
            Imaginary part, i.e. conductivity, of the material 
            (default : 0.11025)
        freq : float, optional
            Frequency at which the sim will be run.
            Needed to set the material property accordingly (default : 0.33)

        Notes
        -----
        The values given are set by default to be ECCOSORB CR110 @ 100 GHz
        -> eps_real = 3.5, eps_imag = 0.11025, freq = 1/3
        When meep units distance is mm.
        '''

        self.name = name                        
        self.object_type = 'Absorber'           
        self.thick = thick                      
        self.center = center                    
        self.epsilon_real = epsilon_real            
        self.epsilon_imag = epsilon_imag            
        self.freq = freq                        
        self.conductivity = epsilon_imag*2*np.pi*freq/epsilon_real

    def __str__(self):
        if self.name is not None :
            return self.name + ', thickness ' + str(self.thick)
        else :
            return 'Absorber walls, thickness ' + str(self.thick)

class HalfWavePlate(object):
    '''
    Class defining a Half Wave Plate.
    '''
    def __init__(self, name = '', fast_ax_index = 3.019, slow_ax_index = 3.336, 
                thick = 5, x_pos = 0, size_y = 300, theta = 45):
        '''
        Defines the attributes of the HWP object

        Arguments
        ---------
        name : str, optional
            Name of object
        fast_ax_index : float, optional
            Index of fast axis (default : 3.019)
        slow_ax_index : float, optional
            Index of slow axis (default : 3.336)
        thick : float, optional
            Thickness of the plate (default : 5)
        x_pos : float, optional
            Position along x of the center of the plate (default : 0)
        size_y : float, optional
            Size of the plate along y axis (default : 300)
        theta : float, optional
            Angle at which the fast axis is rotated in degrees
            around x-axis, 0 deg being // to y axis.
            (default : 45)

        Notes
        -----
        The values given are set by default to be a saphire 
        plate with fast axis is 45 degrees from y axis. 
        '''

        #Permittivity is index squared
        eps_0 = fast_ax_index**2
        eps_e = slow_ax_index**2

        self.object_type = 'HWP'
        #Permittivity matrix rotated by theta
        theta = np.radians(theta)
        self.diag = mp.Vector3(eps_0, 
                        eps_0*(np.cos(theta)**2) + eps_e*(np.sin(theta)**2), 
                        eps_0*(np.sin(theta)**2) + eps_e*(np.cos(theta)**2))
        self.offdiag = mp.Vector3(0,0, (eps_e-eps_0)*np.cos(theta)*np.sin(theta))
        self.thick = thick
        self.x_pos = x_pos
        self.size_y = size_y

    def __str__(self):
        if self.name is not None :
            return self.name + ', thickness ' + str(self.thick) + ', rotated by ' + str(self.theta) + 'degrees'
        else :
            return 'Absorber walls, thickness ' + str(self.thick) + ', rotated by ' + str(self.theta) + 'degrees'
        
class AsphericLens(object):
    '''
    Class defining an aspheric lens of arbitrary shape and 
    position, and creating the function of sag (curvature) 
    used to create the permitttivity map
    '''
    
    def __init__(self, diameter, r1, r2, thick,
                 c1 = 0, c2 = 0, 
                 name = None, 
                 x = 0., y = 0., 
                 n_refr = 1.52, 
                 AR_left = None, AR_right = None,
                 delam_thick = 0,
                 delam_width = 10,
                 radial_slope = 0,
                 axial_slope = 0,
                 surf_err_width = 1,
                 surf_err_scale = 0,
                 custom_def = False):
        '''
        Defines the attributes of the Lens object

        Arguments
        ---------       
        diameter : float 
            Diameter of the lens
        r1 : float
            Left surface curvature radius
        r2 : float  
            Right surface cruvature radius
        thick : float
            Thickness of lens on the optical axis
        name : str, optional
            Name of object (default : None)
        c1 : float, optional
            Left surface aspheric parameter (default : 0)
        c2 : float, optional
            Right surface aspheric parameter (default : 0)
        x : float, optional
            Position of center of left surface along x axis (default : 0)
        y : float, optional
            Position of center of left surface along y axis (default : 0)
        n_refr : float, optional
            Index of refraction of the lens. 
            Set to HDPE by default.
            (default : 1.52) 
        AR_left : float, optional
            Anti Reflection coating thickness of left surface of the lens
            (default : None) 
        AR_right : float, optional
            Anti Reflection coating thickness of right surface of the lens
            (default : None) 
        delam_thick : float, optional
            Thickness of delaminated lumps at their center
            (default : 0)
        delam_width : float, optional
            Width of delaminated lumps along y-axis
            Used in a division, hence default is not 0.
            (default : 10)
        radial_slope : float, optional
            Derivative of the index of refraction w.r.t y-axis (default : 0)
        axial_slope : float, optional
            Derivative of the index of refraction w.r.t x-axis (default : 0)
        surf_err_scale : float, optional
            Width of the gaussian of the distribution of surface errors
            (default : 0)
        surf_err_width : float, optional
            Size of the bins of same surface error (default : 1)
        custom_def : bool, optional 
            Enables custom deformation function (default : False) 
        '''
        self.name = name                  
        self.diameter = diameter        
        self.r1 = r1                    
        self.r2 = r2                    
        self.c1 = c1                    
        self.c2 = c2                    
        self.thick = thick              
        self.x = x                      
        self.y = y                      
        self.eps = n_refr**2            
        self.object_type = 'Lens'       
        self.AR_left = AR_left          
        self.AR_right = AR_right        
        self.AR_material = n_refr       
        self.delam_thick = delam_thick  
        self.delam_width = delam_width  
        self.radial_slope = radial_slope
        self.axial_slope = axial_slope  
        self.surf_err_width = surf_err_width    
        self.surf_err_scale = surf_err_scale    
        self.custom_def = custom_def      

        #TESTING IMPORTED DEFORMED PROFILE AS CSV
        #deform = []
        #with open('deformedsurface.csv') as csvfile:
        #    reader = csv.reader(csvfile, delimiter=',')
        #    k = 0
        #    for row in reader:
        #        k+= 1 
        #        if k>=11 :
        #            deform.append(np.float(row[2]))
        #deform0 = 2*deform[0]-deform[1]
        #deform.insert(0, deform0)
        #self.deform = deform

    
    def left_surface(self, y):
        '''
        Aspheric lens equation for left surface

        Arguments
        ---------
        y : float
            Distance from optical axis at which the sag is computed

        Returns
        -------
        sag : float
            Sag at at distance y from optical axis.
        '''
        
        if self.r1 != np.inf :
            return (y**2/self.r1) / (1 + np.sqrt(1 - (1+ self.c1)*y**2/self.r1**2))
        else : 
            #If the radius is infinite, returns a flat surface, i.e. 0 sag
            return 0
    
    def right_surface(self, y):
        '''
        Aspheric lens equation for right surface

        Arguments
        ---------
        y : float
            Distance from optical axis at which the sag is computed

        Returns
        -------
        sag : float
            Sag at at distance y from optical axis.
        '''
        
        if self.r2 != np.inf :
            return (y**2/self.r2) / (1 + np.sqrt(1 - (1+ self.c2)*y**2/self.r2**2))
        else : 
            #If the radius is infinite, returns a flat surface, i.e. 0 sag
            return 0

    def delamination(self, y, y0):       
        '''
        Returns the air layer thickness that makes delamination, it is 
        zero everywhere excpet where there's the lump, centered on y0, defined by
        its width and thickness

        Arguments
        ---------
        y : float
            Distance from optical axis at which 
            the delamination is evaluated
        y0 : float
            Center of the delaminated lump
        Returns
        -------
        delam : float
            Delamination layer thickness along x-axis at y
        '''

        thick = self.delam_thick
        width = self.delam_width
        return np.abs(min((((y-y0)/width)**2-1)*thick, 0))

    def cust_def(self, y):
        '''
        Returns custom deformation function

        Arguments
        ---------
        y : float
            Distance from optical axis at which 
            the deformation is evaluated
        Returns
        -------
        deform : float
            Deformation of surface along x-axis at y
        '''
        if self.custom_def :
            #Insert here the custom function
            return 0

        else :
            return 0

    def __str__(self):
        if self.name is not None : 
            return self.name + ' at position ' + str(self.x)
        else :
            return 'Lens at position ' + str(self.x)
          
class ApertureStop(object):
    '''
    Class defining an aperture stop
    '''

    def __init__(self,
                 diameter, 
                 pos_x, 
                 thickness, 
                 n_refr = 1, 
                 conductivity = 1e7,
                 name = None):
        '''
        Defines the attributes of the aperture stop object

        Arguments
        ---------       
        diameter : float 
            Diameter of the aperture stop opening
        pos_x : float
            Position of the aperture stop along x-axis
        thickness : float
            Thickness of aperture stop slab
        n_refr : float, optional 
            Index of refraction of the material 
            if the stop is dielectric
            (default = 1)
        conductivity : float, optional
            Conductivity of the material (default = 1e7)
        name : str, optional
            Name of object (default : None)
        '''

        self.name = name                
        self.thick = thickness          
        self.x = pos_x                  
        self.diameter = diameter        
        self.permittivity = n_refr**2   
        self.conductivity = conductivity
        self.object_type = 'AP_stop'

    def __str__(self):
        if self.name is not None :
            return self.name + ' at position ' + str(self.x)
        else : 
            return 'Aperture Stop at position ' + str(self.x)

class ImagePlane(object):
    '''
    Class defining an image plane
    '''

    def __init__(self,  
                 diameter, 
                 pos_x, 
                 thickness, 
                 n_refr = 1, 
                 conductivity = np.inf,
                 name = ''):
        '''
        Defines the attributes of the aperture stop object

        Arguments
        ---------       
        diameter : float 
            Diameter of the image plane slab
        pos_x : float
            Position of the image plane along x-axis
        thickness : float
            Thickness of image plane slab
        n_refr : float, optional
            Index of refraction of the material 
            if the stop is dielectric
            (default = 1)
        conductivity : float, optional
            Conductivity of the material (default = np.inf)
        name : str, optional
            Name of object (default : None)
        '''

        self.name = name                
        self.thick = thickness          
        self.x = pos_x                  
        self.diameter = diameter        
        
        if conductivity != np.inf :
            #Defines the material with given properties
            self.material = mp.Medium(epsilon=n_refr**2, 
                                      D_conductivity = conductivity)
        
        else :
            #If the conductivity is infinite, Meep can define a perfect conductor
            self.material = mp.perfect_electric_conductor
        
        self.object_type = 'ImagePlane'

    def __str__(self):
        if self.name is not None :
            return self.name + ' at position ' + str(self.x)
        else : 
            return 'Image Plane at position ' + str(self.x)
        
class Sim(object):
    '''
    Runs the sim object from MEEP with the dielectric map 
    created for the system and a source that can be 
    specified as Gaussian or plane wave.
    Custom sources not implemented yet.
    '''
    
    def __init__(self, optical_system):
        '''
        Initializes the main simulation properties

        Arguments
        ---------
        optical_system : OpticalSystem object
            Optical system to be used for the simulation
        '''

        self.OS = optical_system

        # Defines the boundary layer of Perfectly Matched Layer as well as the 
        # computational cell so that the PML doesn't overlap on the materials.
        dpml = self.OS.dpml
        self.pml_layers = [mp.PML(thickness = dpml)]
        self.cell_size_x = self.OS.size_x+2*dpml
        self.cell_size_y = self.OS.size_y+2*dpml
        self.cell = mp.Vector3(self.OS.size_x+2*dpml, 
                               self.OS.size_y+2*dpml)

    def __str__(self):
        return 'Simulation object of Optical System named ' + self.OS.name

    def help_gaussian_beam(self, taper_angle, wvl,
                                beam_waist = None,
                                taper = None):
        '''
        For a gaussian beam source
        Provides taper when given beam waist and provides beam waist
        when given taper, at a given taper angle and wavelength in meep units.
        Arguments
        ---------
        taper_angle : float
            Angle in degrees at which the taper is given
        wvl : float
            Wavelength of the source
        beam_waist : float, optional
            Size of the beam waist in MEEP units
        taper : float, optional
            Taper in dB
        '''                        

        a = 20*np.log10((1 + np.cos(np.radians(taper_angle)))/2)
        b = 10*(2*np.pi)**2 * (1-np.cos(np.radians(taper_angle)))*np.log10(np.exp(1)) 

        if beam_waist is None :
            w0 = np.sqrt(- wvl**2 * (taper - a)/b)
            print('The beam waist is {:.2e} MEEP units'.format(w0))

        if taper is None :
            A = a - b* beam_waist**2 / wvl**2
            print('The taper at angle {:.1f} deg is {:.2f} dB'.format(taper_angle, A))

    def set_verbosity(self, verbosity = 0):
        '''
        Sets MEEP's verbosity
        
        Arguments
        ---------
        verbosity : int
            Value to give to the verbosity
            0 is no prints
            1 is standard prints (enough for debugging)
            2 is all prints
        '''

        mp.verbosity(verbosity)
                   
    def define_source(self, 
                      f = None,
                      wvl = None, 
                      sourcetype = 'Plane wave', 
                      x = 0, 
                      y = 0, 
                      size_x = 0, 
                      size_y = 0, 
                      beam_width = 0,
                      rot_angle = 0):
        '''
        Defines the source to be used by the simulation. Only does one source
        at a time

        Arguments
        ---------
        f : float, optional
            Frequency of the source (default : None)
        wvl : float, optional
            Wavelength of the source (default : None)
        sourcetype : str, optional
            A source can be a plane wave coming on the aperture
            or a gaussian beam on the image plane. 
            (default : 'Plane wave')
        x : float, optional
            x-Position of the source center. (default : 0.)
        y : float, optional
            y-Position of the source center. (default : 0.)
        size_x : float, optional
            x-size of the source. (default : 0.)
        size_y : float, optional
            y-size of the source. (default : 0.)
        beam_width : float, optional
            For a gaussian beam, defines its width. (default : 0.)
        rot_angle : float, optional
            Angle by which the plane wave is rotated w.r.t vertical

        Returns
        -------
        self.source : MEEP source object
            Object that will be used in the sim function.

        Notes
        -----
        Gaussian beam is located by default on the image plane.
        '''
        
        if wvl is not None :
           f = 1/wvl

        if f is not None:
            wvl = 1/f
        
        self.wvl = wvl
        self.f = f
        
        #Rotation of plane wave
        rot_angle *= np.pi/180
        def amp_func(P):
            '''
            Returns amplitude of source with added phase to 
            emulate source rotation

            Arguments
            ---------
            P : mp.Vector3
                Meep position object at which the source is evaluated.

            Returns
            -------
            amp : complex
                Complex amplitude of source at P.
            '''

            k = mp.Vector3(2*np.pi*np.cos(rot_angle)/self.wvl,
                           2*np.pi*np.sin(rot_angle)/self.wvl,
                           0)
            return np.exp(1j* k.dot(P))

        
        #Different sources definitions
        if sourcetype == 'Plane wave':
            self.source = [mp.Source(mp.ContinuousSource(f, is_integrated=True),
                           component=mp.Ez,
                           center=mp.Vector3(x-self.OS.size_x/2, y, 0),
                           size=mp.Vector3(size_x, size_y, 0),
                           amp_func = amp_func)]
        

        elif sourcetype == 'Gaussian beam':
            self.source = [mp.GaussianBeamSource(mp.ContinuousSource(f),
                      component = mp.Ez,
                      center = mp.Vector3(self.OS.IP_pos-self.OS.size_x/2, y, 0),
                      beam_x0 = mp.Vector3(0, 0),
                      beam_kdir = mp.Vector3(-1, 0),
                      beam_w0 = beam_width,
                      beam_E0 = mp.Vector3(0,0,1),
                      size=mp.Vector3(size_x, size_y, 0))]

            self.beam_waist = beam_width
        
        """ 
        #NOT CURRENTLY USED   
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
                      center = mp.Vector3(self.OS.IP_pos-1, y_meep, 0),
                      beam_x0 = mp.Vector3(focus_pt_x, focus_pt_y),
                      beam_kdir = mp.Vector3(-1, 0),
                      beam_w0 = beam_width,
                      beam_E0 = mp.Vector3(0,0,1),
                      size=mp.Vector3(size_x, size_y, 0))]
        """
        return self.source
    
    
    def run_sim(self, 
                runtime, 
                simres = 1, 
                get_mp4 = False, 
                Nfps = 24, 
                movie_name = 'movie', 
                image_every = 5,
                ff_angle = 45,
                ff_npts = 500):
        '''
        Creates the sim environment as defined by MEEP and then runs it.
        Also defines the near field region to get the far field beam.
        Also can save a video of the sim.

        Arguments
        ----------
        runtime : float
            Meep time for which the sim should be run. 
            Usually equal to system size along x.
        sim_resolution : float, optional
            Resolution of the grid created by Meep. 
            Recall that wavelength*resolution should 
            be at least 8 in the highest index material. 
            (default : 1)
        get_mp4 : bool, optional
            Whether to make a video of the sim (default : False)
        Nfps : int, optional 
            Number of fps at which to save the video. (default : 24)
        movie_name : str, optional
            Name of the file of the video. (default : 'movie')
        ff_angle : float, optional
            Max angle in degrees at which the 
            far field is retrieved (default : 45)
        ff_npts : int, optional
            Number of far field points (default : 500)
        '''

        self.simres = simres

        #Parameters defining the far fied properties
        self.ff_distance = 1e8      
        self.ff_angle = ff_angle       
        self.ff_npts = ff_npts        
        

        
        #Defines the simulation environment, using the various objects defined
        #previously
        self.sim = mp.Simulation(cell_size=self.cell,
                    #geometry_center = mp.Vector3(-self.OS.size_x/2,0,0),
                    boundary_layers=self.pml_layers,
                    geometry=self.OS.geometry, 
                    sources=self.source,
                    resolution=self.simres,
                    epsilon_input_file = self.OS.mapname + '.h5:eps')

        #Defines the near field region that can then be used to retrieve
        #the far field beam.
        nfreq = 1
        fcen = self.f
        df = 0

        n2f_pt = mp.Vector3(self.OS.aper_pos_x-self.OS.size_x/2, 0)
        self.n2f_obj = self.sim.add_near2far(fcen, 
                            df, 
                            nfreq, 
                            mp.Near2FarRegion(center=n2f_pt, 
                                size = (0,self.OS.aper_size), weight = -1))     

        
        #Runs sim
        if get_mp4 :
            #Animate object
            animate = mp.Animate2D(self.sim,
                       fields=mp.Ez,
                       realtime=True,
                       field_parameters={'alpha':0.8, 
                                        'cmap':'RdBu', 
                                        'interpolation':'none'},
                       boundary_parameters={'hatch':'o', 
                                        'linewidth':1.5, 
                                        'facecolor':'y', 
                                        'edgecolor':'b', 
                                        'alpha':0.3})

            self.sim.run(mp.at_every(image_every, animate), until = runtime)
            animate.to_mp4(Nfps, movie_name + '.mp4')

        if not get_mp4 :
            self.sim.run(until = runtime)
        
    def get_MEEP_ff(self, 
                    saveplot = False,
                    parallel = False,
                    saveh5 = False,
                    filename = None,
                    ylim = -60):
        '''
        Gets the far field using MEEP near2far function.

        Arguments
        ---------
        saveplot : bool, optional
            Whether to save the plot (default : False)
        parallel : bool, optional
            Whether the code is running in parallel MPI (default : False)
        saveh5 : bool, optional
            Whether to save the far field beam amplitude (default : False)
        filename : str, optional
            Name of the files saved (default : None)
        ylim : float, optional
            Limit of plot in negative dB (default : -60)
        '''
        ff_length = self.ff_distance*np.tan(np.radians(self.ff_angle))
        ff_res = self.ff_npts/ff_length

        ff = self.sim.get_farfields(self.n2f_obj, 
            ff_res, 
            center=mp.Vector3(- self.ff_distance,0.5*ff_length), 
            size=mp.Vector3(y=ff_length))

        ff_lengths = np.linspace(0,ff_length,self.ff_npts)
        angles = [np.degrees(np.arctan(f)) for f in ff_lengths/self.ff_distance]

        norm = np.absolute(ff['Ez'])/np.max(np.absolute(ff['Ez'])) / (np.cos(np.radians(angles)))**2
        ff_dB = 10*np.log10(norm)  

        self.ffmeep = ff_dB
        self.angles = angles
        if saveplot : 
            plt.figure(figsize = (8,6))
            plt.plot(angles,ff_dB,'bo-')
            plt.xlim(0,self.ff_angle)
            plt.ylim((ylim,0))
            plt.xticks([t for t in range(0,self.ff_angle+1,10)])
            plt.xlabel("Angle [deg]")
            plt.ylabel("Amplitude [dB]")
            plt.grid(axis='x',linewidth=0.5,linestyle='--')
            plt.title("f.-f. spectra @  λ = 10 mm")
            plt.savefig(filename + '.png')
            plt.close()

        if saveh5 :
            if parallel : 
                h = h5py.File(filename + '.h5', 'w', driver ='mpio', comm=MPI.COMM_WORLD)
            else : 
                h = h5py.File(filename + '.h5', 'w')
            h.create_dataset('deg', data=angles)
            h.create_dataset('amplitudedB', data=ff_dB, dtype = 'float64')
            h.close()
        
    def plot_system(self):
        
        '''
        Plots the various objects in the computational cell, with 
        the objects in grey and the PML in purple.

        Notes
        -----
        Doesn't work ideally when some objects are perfect 
        conductors, as the permittivity goes to infinity, the objects
        with finite permittivity are not shown.
        '''
        
        eps_data = self.sim.get_array(center=mp.Vector3(), 
                                      size=self.cell, 
                                      component=mp.Dielectric)
        pml = np.zeros((eps_data.transpose().shape))
        pml[0: self.dpml*self.simres, :] = 1
        pml[:, 0: self.dpml*self.simres] = 1
        pml[:, -self.dpml*self.simres : ] = 1
        pml[-self.dpml*self.simres : , :] = 1
        plt.figure()
        plt.contourf(eps_data.transpose(), interpolation='spline36', cmap='Greys')
        plt.imshow(pml, cmap = 'Purples', alpha = 0.4)
        plt.xlabel('x times resolution')
        plt.ylabel('y times resolution')
        plt.savefig('2Dsystem')
        plt.show()
        plt.close()
    
    def plot_efield(self, name = 'efield', comp = 'Ez', amp_dB = False) :

        '''
        Plots the electric field in the system.

        Arguments
        ---------
        name : str, optional
            Name of plot to be saved
        comp : str, optional
            Component of field to plot. 'Ez' or 'Ey'. (default : 'Ez')
        amp_dB : bool, optional
            Whether to plot the amplitude in dB 
        '''
        if comp == 'Ez':
            field = mp.Ez
        elif comp == 'Ey':
            field = mp.Ey

        def func(x):
            if amp_dB :
                return 10*np.log10(np.abs(x))
            else : 
                return x

        plt.figure(figsize = (15,15))
        self.sim.plot2D(fields = field, 
            labels = True,
            eps_parameters = {'interpolation':'spline36', 
                                'cmap':'binary'},
            field_parameters={'alpha':0.8, 
                                'cmap':'RdBu', 
                                'interpolation':'none',
                                'postprocess':func},
            boundary_parameters={'hatch':'o', 
                                'linewidth':1.5, 
                                'facecolor':'y', 
                                'edgecolor':'b', 
                                'alpha':0.3},
            plot_monitors_flag = False)

        x_pos = [k - self.OS.size_x/2 for k in range(0, int(self.OS.size_x), 50)]
        plt.xticks(x_pos, [str(k) for k in range(0, int(self.OS.size_x), 50)] )
        plt.savefig(name + '.png')

    """
    NOT USED, CAN BE USEFUL
    def plot_airy_spot(self):

        #Makes a plot of the evolution of the instantaneous Ez squared over the 
        #image plane, if the system is properly focused, for an incoming plane
        #wave, should give an Airy pattern

        plt.figure()
        for k in range(np.int(5/self.frequency*0.5)):
            self.sim.run(until = 1)
            #Gets the e-field just 1 unit before the image plane
            ez_data = self.sim.get_array(center=mp.Vector3(self.OS.IP_pos-1, 0), 
                                         size=mp.Vector3(0, self.OS.size_y), component=mp.Ez)
            plt.scatter(np.arange(len(ez_data)/self.simres), ez_data**2, marker = '+')
        plt.title('$E^2$ at image plane')
        plt.xlabel('y (mm)')
        plt.ylabel('$E^2$')
        plt.show()
    """

    def get_complex_field(self, 
                        plot_amp = False, 
                        saveh5 = False, 
                        filename = 'test',
                        parallel = False):
        '''
        Gets the electric field in its complex form at the aperture.
        To that end, fits the time evolution of the field there.

        Arguments
        ---------
        plot_amp : bool, optional
            Whether to plot the amplitude of field at aperture. 
            (default : False)
        saveh5 : bool, optional
            Whether to save the amplitude in an h5 file. (default : False)
        filename : str, optional
            Name of the plot to be saved
        parallel : bool, optional
            Whether the code is running in parallel

        Returns
        -------
        amplitude*phase : complex array
            Complex electric field at the aperture

        '''

        #Setting the timestep to a very low value, 
        #so that MEEP uses its lowest timestep
        timestep = .3

        #120 steps Is roughly enough to give a 
        #few periods for wavelengths from 1 to 10
        #Can be tweaked to save on sim time.
        n_iter = 120

        res = self.simres
        AP_size = self.OS.aper_size
        aper_pos_x = self.OS.aper_pos_x - self.OS.size_x/2
        
        #Get the real field at aperture
        efield = self.sim.get_array(center=mp.Vector3(aper_pos_x, 0), 
                                    size=mp.Vector3(0, AP_size), 
                                    component=mp.Ez)

        

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
            e_field_evol[k] = self.sim.get_array(center=mp.Vector3(aper_pos_x, 0), 
                                         size=mp.Vector3(0, AP_size), 
                                         component=mp.Ez)
        
        #Each point on the aperture is fit for a cosine with amplitude and phase
        def f(x, amp, phase):
            return amp*np.cos(x*2*np.pi/self.wvl + phase)

        #Initialize the lists of amplitude and phase over the aperture
        amplitude = np.zeros(AP_size*res)
        phase = np.zeros(AP_size*res)

        #The field is only taken on the opening of the aperture

        #Fits amplitude and phase for each point
        for k in range(AP_size*res):
            popt, pcov = sc.curve_fit(f, time, e_field_evol[:,k])
            amplitude[k] = popt[0]
            phase[k] = popt[1]
        
        ### Plot
        if plot_amp :
            norm = np.max(np.abs(amplitude))
            amp = 10*np.log10(np.abs(amplitude)/norm)
            y = np.linspace(-AP_size/2,AP_size/2,len(amplitude))
            plt.figure()
            plt.plot(y, amp) 
            plt.ylim((-60,0))
            plt.xlim((0, AP_size/2))
            plt.title('E field amplitude on aperture')
            plt.xlabel('y (mm)')
            plt.ylabel('$Amplitude [dB]$')
            plt.savefig(filename + '.png')
            plt.close()

        if saveh5 : 
            if parallel :
                h = h5py.File(filename + '.h5', 'w', 
                            driver ='mpio', 
                            comm=MPI.COMM_WORLD)
            else: 
                h = h5py.File(filename + '.h5', 'w')
            h.create_dataset('y', data=y)
            h.create_dataset('amplitude', data=amplitude, dtype = 'float64')
            h.create_dataset('phase', data=phase, dtype = 'float64')
            h.close()
        
        return amplitude*np.exp(1j*phase)
    
class Analysis(object):
    '''
    Class definining analysis tools.
    '''
    
    def __init__(self, sim):
        '''
        Defines the used sim environment : the objects and their properties as 
        specified in the sim will be the same all throughout the analysis.

        Arguments
        ---------
        sim : sim object
            Sim to be used
        '''
        self.sim = sim
        
    def image_plane_beams(self, 
                        f = None, 
                        wvl = None, 
                        y_source = 0., 
                        simres = 1,
                        runtime = 750, 
                        beam_w0 = 10,
                        plot_amp = False, 
                        saveh5 = False, 
                        filename = 'test',
                        parallel = False):
        '''
        Sends gaussian beams from the image plane and recovers 
        the amplitudes at the aperture. 

        Arguments
        ---------
        f : float or list, optional
            Frequency of the source (default : None)
        wvl : float or list, optional
            Wavelength of the source (default : None)
        y_source : float or list, optional
            Position of source on image plane along y axis. (default : 0)
        simres : float or list, optional
            Resolution of simulation (default : 1)
        runtime : float, optional
            Runtime of simulation. Should roughly be 
            system size along optical axis. (default : 750)
        beam_w0 : float or list, optional
            Size of beam waist of gaussian source (default : 10)
        plot_amp : bool, optional
            Whether to plot the amplitude of field at aperture. 
            (default : False)
        saveh5 : bool, optional
            Whether to save the amplitudes in an h5 file. (default : False)
        filename : list of str, optional
            Names of the files to be saved
        parallel : bool, optional
            Whether the code is running in parallel
        Notes
        -----
        If the arguments above are given as lists, they should all be 
        the same length, and a sim will be run for each element, taking
        the property i of each list.
        '''

        if not isinstance(wvl, list) :
            wvl = [wvl]
            y_source = [y_source]
            beam_w0 = [beam_w0]
            filename = [filename]

        # Adaptation to specify either in wavelength or frequency :
        if wvl is not None :
           f = [1/wvl[k] for k in range(len(wvl))]

        if f is not None:
            wvl = [1/x for x in f]

        self.wvl = wvl

        Nb_src = len(wvl)

        #Initialize the electric fields list
        self.list_efields = [[] for k in range(Nb_src)]

        #Iterates over the number of sources
        for k in range(Nb_src):

            #Defines the source at the appropriate height on the image plane
            self.sim.define_source(wvl = wvl[k], 
                                   sourcetype = 'Gaussian beam', 
                                   y = y_source[k], 
                                   size_x = 0, 
                                   size_y = self.sim.OS.size_y, 
                                   beam_width = beam_w0[k])

            
            #Runs the sim
            self.sim.run_sim(runtime, simres = simres, ff_angle = 80, ff_npts = 800)

            #self.sim.plot_efield()

            #self.sim.get_MEEP_ff(saveplot = True,
            #        parallel = False,
            #        saveh5 = False,
            #        filename = 'testFF',
            #        ylim = -40)

            #Gets the complex electric field and adds it to the plot
            E_field = self.sim.get_complex_field(plot_amp = plot_amp,
                                    saveh5 = saveh5, 
                                    filename = filename[k],
                                    parallel = parallel)

            #Updates the list of fields
            self.list_efields[k] = E_field

    def beam_FT(self, 
                zero_pad = 15):

        '''
        Gets the Fourier Transforms of the complex electric fields at aperture.

        Arguments
        ---------
        zero_pad : float, optional
            Multiplicative factor to the length of the field list,
            which is padded with zeros in the added length.

        Returns
        -------
        freq : array
            List of the frequencies at which the FFT has been done
        FFTs : list of arrays
            Each array contains the FFT for the k-th source.
        '''

        #Initialize the list
        FFTs = [[] for k in range(len(self.list_efields))]

        res = self.sim.simres

        #List of frequencies
        freq = np.fft.fftfreq(len(self.list_efields[0])*zero_pad, d = 1/res)

        #Iterate over the number of sources
        for k in range(len(self.list_efields)):

            #FFT over the field
            fft = np.fft.fft(self.list_efields[k], 
                n = zero_pad*len(self.list_efields[k]))

            #FFT is normalized by its max
            FFTs[k] = np.abs(fft) 
            FFTs[k] = FFTs[k]/np.max(FFTs[k])

        return freq, FFTs

    def plotting(self, fftfreq, FFTs, wvl,
                deg_range = 20,
                ylim = -60, 
                symmetric_beam = True,
                legend = None,
                print_solid_angle = False,
                print_fwhm = False,
                savefig = False,
                path_name = 'plots/meep_guide_plot'):
        '''
        Plots far field beam

        Arguments
        ---------
        fftfreq : float
            Array of the frequencies of the FFT
        FFTS : float
            List of the normalized beams of the FFT
        wvl : float or list of floats
            Wavelengths of the beams 
        deg_range : float, optional
            Range in degrees of the plot (default : 20)
        ylim : float, optional
            Min amplitude of the plot, in dB (default : -60)
        symmetric_beam : bool, optional
            If the beam is symmetric, if true only plots half of the beam
            (default : True)
        legend : list of str, optional
            Legend of the various far fields plotted (default : None)
        print_solid_angle : bool, optional
            Whether to print the solid angle (default : False)
        print_fwhm : bool, optional
            Whether to print the best fit gaussian FWHM (default : False)
        savefig : bool, optional
            Whether to save the figure (default : False)
        path_name : str, optional
            Path and name of the plot to be saved 
            (default : 'plots/meep_guide_plot')
        '''

        deg = np.arctan(fftfreq*wvl)*180/np.pi
        rads = np.array(deg) * np.pi/180
        

        plt.figure(figsize = (8,6))
        
        def gaussian(x, stddev, mean):
            return np.exp(-(((x-mean)/4/stddev)**2))
        
        for k in range(len(FFTs)):

            fft_k = FFTs[k] / (np.cos(rads)**2)
            fft_dB = 10*np.log10(fft_k / np.max(fft_k))
            middle = int(len(fft_k)/2)

            #BEAM SOLID ANGLE CALCULATION
            if print_solid_angle :
                
                
                x_span = np.append(rads, 0)
                integrand = np.append(fft_k, fft_k[0])
                right_part = np.trapz(integrand[:middle], x = x_span[:middle])
                left_part = np.trapz(integrand[middle:], x = x_span[middle:])
                solid_angle = right_part + left_part
                print('Beam n.{} solid angle : {:.3e} srads'.format(k, 
                    solid_angle*2*np.pi))
            
            if legend is not None : 
                plt.plot(deg[:middle], fft_dB[:middle], 
                    label = '{}'.format(legend[k]))

            if legend is None :
                plt.plot(deg[:middle], fft_dB[:middle])

                #TESTING, ignore this
                #plt.plot(self.sim.angles, self.sim.ffmeep)

            #BEST FIT GAUSSIAN FWHM
            if print_fwhm :
                popt, psig = sc.curve_fit(gaussian, deg, fft_k)
                fwhm = popt[1] + 4*popt[0]*np.sqrt(np.log(2))
                fwhm_th = wvl/self.sim.OS.aper_size*180/np.pi
                print('Best fit Gaussian FWHM : {:.2f}deg'.format(2*fwhm))
                print('Theoretical FWHM : {:.2f}deg'.format(fwhm_th))
                y = 10*np.log10(gaussian(deg, popt[0], popt[1]))

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

        #Additional plotting tools
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

    #Testing code here
    
    lens1 = AsphericLens(name = 'Lens 1', 
                         r1 = 327.365, 
                         r2 = np.inf, 
                         c1 = -0.66067, 
                         c2 = 0, 
                         thick = 40,
                         diameter = 300, 
                         x = 130.+10., 
                         y = 0.)
    
    lens2 = AsphericLens(name = 'Lens 2', 
                         r1 = 269.190, 
                         r2 = 6398.02, 
                         c1 = -2.4029, 
                         c2 = 1770.36, 
                         diameter = 300,
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
                             n_refr = 1., 
                             conductivity = 0)



    opt_sys = OpticalSystem('OptSysTest')
    opt_sys.set_size(750,340)

    #print(opt_sys)
    #opt_sys.sys_info(1e-3, meep_freq = 0.3333333)
    #print(lens1)
    #print(lens2)
    #print(image_plane)
    #print(aperture_stop)

    res = 5
    wavelength = 10
    dpml = 5

    tube = TelescopeTube(name = 'Tube', thick = 10, center = 165)
    absorber = Absorber(name = 'Absorber', thick = 10, center = 155)

    #print(tube)
    #print(absorber)
    
    
    opt_sys.add_component(lens1)
    opt_sys.add_component(lens2)
    opt_sys.add_component(aperture_stop)
    opt_sys.add_component(image_plane)
    opt_sys.add_component(tube)
    opt_sys.add_component(absorber)    
    

    opt_sys.assemble_system(dpml = dpml, res = res)
    opt_sys.plot_lenses()
    opt_sys.write_h5file(parallel = False)

    opt_sys.plot_lenses()
    sim = Sim(opt_sys)

    sim.help_gaussian_beam(5, 10, beam_waist = 30)

    print(sim)
    
    """
    sim.define_source(wvl = wavelength, 
                      sourcetype = 'Gaussian beam',
                      y=0, 
                      beam_width = 30, 
                      size_x = 0, 
                      size_y = 300)
    
    sim.run_sim(runtime = 800, simres = 1)

    
    sim.plot_efield()

    sim.get_MEEP_ff(saveplot = True,
                    parallel = False,
                    saveh5 = False,
                    filename = 'testFF',
                    ylim = -40)
    """
    
    analyse = Analysis(sim)

    analyse.image_plane_beams(wvl = 3., 
                        y_source = 100., 
                        simres = res,
                        runtime = 800, 
                        beam_w0 = 3.,
                        parallel = False)

    fftfreq, fft = analyse.beam_FT()
    analyse.plotting(fftfreq, fft, wavelength,
                deg_range = 60,
                ylim = -40, 
                symmetric_beam = True,
                legend = None,
                print_solid_angle = True,
                print_fwhm = True,
                savefig = True,
                path_name = 'withabs')
    
    

