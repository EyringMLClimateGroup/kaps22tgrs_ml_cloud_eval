import glob
import numpy as np
import os
import random
import warnings
import itertools
from matplotlib import pyplot as plt
import time
import sys
import traceback
import netCDF4 as nc4
import torch
from torch.utils.data import Dataset
import warnings
from copy import copy
import timeit
import torchvision.transforms.functional as T

radiances = ['ev_250_aggr1km_refsb_1', 'ev_250_aggr1km_refsb_2', 'ev_1km_emissive_29', 'ev_1km_emissive_33', 'ev_1km_emissive_34', 'ev_1km_emissive_35', 'ev_1km_emissive_36', 'ev_1km_refsb_26', 'ev_1km_emissive_27', 'ev_1km_emissive_20', 'ev_1km_emissive_21', 'ev_1km_emissive_22', 'ev_1km_emissive_23']
coordinates = ['latitude', 'longitude']
properties = ['cloud_water_path', 'cloud_optical_thickness', 'cloud_effective_radius', 'cloud_phase_optical_properties', 'cloud_top_pressure', 'cloud_top_height', 'cloud_top_temperature', 'cloud_emissivity', 'surface_temperature']
rois = 'cloud_mask'
labels = 'cloud_layer_type'




class LoadingError(Exception):
    pass


def window_nd(a, window, steps = None, axis = None, gen_data = False):
        """
        Create a windowed view over `n`-dimensional input that uses an 
        `m`-dimensional window, with `m <= n`
        
        Parameters
        -------------
        a : Array-like
            The array to create the view on
            
        window : tuple or int
            If int, the size of the window in `axis`, or in all dimensions if 
            `axis == None`
            
            If tuple, the shape of the desired window.  `window.size` must be:
                equal to `len(axis)` if `axis != None`, else 
                equal to `len(a.shape)`, or 
                1
            
        steps : tuple, int or None
            The offset between consecutive windows in desired dimension
            If None, offset is one in all dimensions
            If int, the offset for all windows over `axis`
            If tuple, the steps along each `axis`.  
                `len(steps)` must me equal to `len(axis)`
    
        axis : tuple, int or None
            The axes over which to apply the window
            If None, apply over all dimensions
            if tuple or int, the dimensions over which to apply the window

        gen_data : boolean
            returns data needed for a generator
    
        Returns
        -------
        
        a_view : ndarray
            A windowed view on the input array `a`, or `a, wshp`, where `whsp` is the window shape needed for creating the generator
            
        """
        ashp = np.array(a.shape)
        
        if axis != None:
            axs = np.array(axis, ndmin = 1)
            assert np.all(np.in1d(axs, np.arange(ashp.size))), "Axes out of range"
        else:
            axs = np.arange(ashp.size)
            
        window = np.array(window, ndmin = 1)
        assert (window.size == axs.size) | (window.size == 1), "Window dims and axes don't match"
        wshp = ashp.copy()
        wshp[axs] = window
        assert np.all(wshp <= ashp), "Window is bigger than input array in axes"
        
        stp = np.ones_like(ashp)
        if steps:
            steps = np.array(steps, ndmin = 1)
            assert np.all(steps > 0), "Only positive steps allowed"
            assert (steps.size == axs.size) | (steps.size == 1), "Steps and axes don't match"
            stp[axs] = steps
    
        astr = np.array(a.strides)
        
        shape = tuple((ashp - wshp) // stp + 1) + tuple(wshp)
        strides = tuple(astr * stp) + tuple(astr)
        as_strided = np.lib.stride_tricks.as_strided
        a_view = np.squeeze(as_strided(a, 
                                     shape = shape, 
                                     strides = strides))
        if gen_data :
            return a_view, shape[:-wshp.size]
        else:
            return a_view


def get_class_occurrences(labels):
    """ 
    Takes in a numpy.ndarray of size (nb_instances, W, H, nb_layers=10) describing for each pixel the types of clouds identified at each of the 10 heights and returns a numpy.ndarray of size (nb_points, 8) counting the number of times one of the 8 type of clouds was spotted vertically over a whole instance.
    The height information is then lost. 
    """
    
    occurrences = np.zeros((labels.shape[0], 8))
    
    for occ, lab in zip(occurrences, labels):

        values, counts = np.unique(lab, return_counts=True)

        for v, c in zip(values, counts):
            
            if v > -1: # unlabeled pixels are marked with -1, ignore them
                occ[v] = c
    
    return occurrences  

def get_most_frequent_label(labels):
    """ labels should be of size (nb_instances, ...).
        Returns the most frequent label for each whole instance.
    """

    label_occurrences = get_class_occurrences(labels)

    labels = np.argmax(label_occurrences, 1).astype(float)
    
    # set label of pixels with no occurences of clouds to NaN
    labels[np.sum(label_occurrences, 1) == 0] = np.NaN

    return labels.squeeze()

def get_coords(nc_file):
    """gets latitude and longitude in  that order"""
    file = nc4.Dataset(nc_file, "r", format="NETCDF4")
    coords = np.vstack([file.variables[name][:] for name in coordinates])
    return coords

def read_npz(npz_file, choice = None, liquid=False):
    """
    Reads the npz file, gets the variables, can convert based on phase flag
    Also gets the integer labels per pixel
    Parameters
    ----------
    npz_file : str
        path to file
    choice : np.ndarray, optional
        1-d array, indicating which variables to take. Takes all 8 if None.
        [0:cwp,1:lwp, 2:iwp,  3:reffclw, 4:reffcli, 5:cod,
         6:ptop, 7:htop, 8:ttop, 9:emiss, 10:tsurf]
        The default is None.
    liquid : Bool, optional
        If true, converts cwp to lwp and twp. The default is False.

    Returns
    -------
    np.ndarray (13,1354,2030)
        radiances
    np.ndarray (len(choice),1354,2030)
        chosen pyhsical variables
    np.ndarray (1,1354,2030)
        cloud mask
    np.ndarray (1354,2030)
        labels from 0 to 7, including nan´

    """
    file = np.load(npz_file)
    fileprops= np.ma.masked_invalid(file["properties"])
    assert not np.any(fileprops[4]>1200), np.max(fileprops[4])
    #check the phase flag
    isliquid = fileprops[3,...] == 2
    isclear = fileprops[3] == 1
    isice = fileprops[3] == 3

    for i in [1,4,5,6]:
        fileprops[i,isclear]=np.nan
    newprops = []
    cwp = np.copy(fileprops[0])
    cwp=np.where(cwp>0,cwp,np.nan)
    newprops.append(cwp)
    if liquid:        
        #water cloud cwp
        temp = np.copy(cwp)
        temp[isice] =np.nan
        newprops.append(temp)
        
        #ice cloud cwp
        temp = np.copy(cwp)
        temp[isliquid] =np.nan
        newprops.append(temp)
        
        #water cloud reff
        temp = np.copy(fileprops[2])
        temp[isice]=np.nan
        newprops.append(temp)
        
        #ice cloud reff
        temp = np.copy(fileprops[2])
        temp[isliquid] =np.nan
        newprops.append(temp)
        
        for i in [1,4,5,6,7,8]:
            newprops.append(fileprops[i])
        fileprops = np.stack(newprops)
    else:
        for i in [2,1,4,5,6,7,8]:
            newprops.append(fileprops[i])
        fileprops=np.stack(newprops)
    if choice is None:
        return file['radiances'], fileprops, file['cloud_mask'], file['labels']
    else:            
        fileprops = np.ma.masked_invalid(fileprops)    
        return (file['radiances'], fileprops[choice,...],
                file['cloud_mask'], file['labels'])


def label_stats_wc(arr, classes = 9):
    """
    average amount of each class (including undetermined)

    Parameters
    ----------
    arr : np.ndarray, 
        probably a tile, like (3x3)
    classes : int, optional
        how many classes. The default is 9.

    Raises
    ------
    ValueError
        If nans magically appear after being removed

    Returns
    -------
    relative amount of each class

    """
    x=arr[~np.isnan(arr)].astype(int).flatten()
    if np.any(np.isnan(x)):
        raise ValueError("got nan in statsfct")
    l=len(x)
    if l==0:
        l+=1
    return(np.bincount(x+1, minlength=classes)/l)

def label_abs_wc(arr, classes = 9):
    #total amount of each class
    x=arr.astype(int).flatten()+1
    return(np.bincount(x, minlength=classes))

try: 
    import numba
    @numba.jit(nopython=True, parallel=True)
    def set_count(arr):
        """
        
    
        Parameters
        ----------
        arr : np.ndarray, shape x,y
            array containing integers -1:7
    
        Returns
        -------
        counted np.ndarray
            now 3d, where each 2d point now is an array with zeros, except one at
            the index of the input integer+1
    
        """
        counted = np.zeros((9,arr.shape[0]*arr.shape[1]))
        arr2 =arr+1 
        for i,x in enumerate(arr2.flatten()):
            if not np.isnan(x):
                counted[x,i] =1
        return counted.reshape(9,arr.shape[0],arr.shape[1])
except Exception:
    def set_count(arr):
        """
        
    
        Parameters
        ----------
        arr : np.ndarray, shape x,y
            array containing integers -1:7
    
        Returns
        -------
        counted np.ndarray
            now 3d, where each 2d point now is an array with zeros, except one at
            the index of the input integer+1
    
        """
        counted = np.zeros((9,arr.shape[0]*arr.shape[1]))
        arr2 =arr+1 
        for i,x in enumerate(arr2.flatten()):
            if not np.isnan(x):
                counted[x,i] =1
        return counted.reshape(9,arr.shape[0],arr.shape[1])

#@profile
def transform_bincount(arr):  
    """
    counts the classes in each submatrix
    nans are not counted
    Parameters
      ---------- 
    arr : (x,y) array

    Returns
    -------
    out : (x,y,9) arra
    out[x_i,y_j]=bincount(arr[:x_i,:y_j])
    """
    out = set_count(arr)
    assert out.shape[1]==arr.shape[0], (out.shape, arr.shape)
    out =np.nancumsum(out,1)
    out = np.nancumsum(out,2)
    return out


    
    
class CumuloDataset(Dataset):
    """ Should get a number of mean values for all images with the
        respective label distributions. Pretty sure this is obsolete"""
    def __init__(self, root_dir, label_dir,ext="npz", normalizer=None, 
                  indices=None, check = False, variables = None,
                 num_chunks = 20, chunksize_x = 100, chunksize_y =100,
                 plotting = False, label_fct="rel", filt = None,
                 transform = None):
        """Dataset creator

        Args:
            root_dir (string): where the files with the features are stored
            label_dir (String): where the files with the labels are stored
            ext (str, optional): datatype of input files. Defaults to "npz".
            normalizer (function, optional): Function that normalizes the inputs. I never used this. Defaults to None.
            indices (iterable, optional): will chose only specific files. Defaults to None.
            check (bool, optional): If filepaths should be checked for consistency. avoids errors downstream. Defaults to False.
            variables (iterable, optional): indices of variables to use. Defaults to None.
            num_chunks (int, optional): how many grid cells should be extracted from each file. Low values to avoid memory overflow. Defaults to 20.
            chunksize_x (int, optional): x dimension of grid cell. Defaults to 100.
            chunksize_y (int, optional): y dimension of grid cell. Defaults to 100.
            plotting (bool, optional): Probably bugged. Handles data a bit differently so the output can be plotted. Defaults to False.
            label_fct (str, optional): key for label handling. "rel" is really the only thing that makes sense. Defaults to "rel".
            filt (int, optional): gets only files froma specific month. Never used. Defaults to None.
            transform (function, optional): could use this to for example log-transform the input. Defaults to None.

        Raises:
            NotImplementedError: _description_
            FileNotFoundError: _description_
        """
        self.transform = transform
        self.root_dir = root_dir
        self.ext = ext
        self.label_dir = label_dir
        self.plotting = plotting
        
        if label_fct == "rel":#relative cloud type amount
            self.label_fct = label_stats_wc
        elif label_fct == "abs":#absolute cloud type amount
            self.label_fct = label_abs_wc
            
        
        self.label_paths = glob.glob(os.path.join(label_dir, "*.npy"))
        
        if len(self.label_paths) == 0:
            raise FileNotFoundError("no label files in {}".format( label_dir))
        
        self.label_paths.sort()
        if indices is not None:
            if filt is None:
                self.label_paths = [self.label_paths[i] for i in indices]
            else:
                months = [1, -2, 1, 0, 1,0,1,1,0,1,0,1]
                comp_min = int(filt)*30
                comp_max = comp_min + 31 + np.sum(months[:int(filt)])
                label_paths = []
                for p in self.label_paths:
                    day = int(p[-12:-9])
                    if day>=comp_min and day<comp_max:
                        label_paths.append(p)
                self.label_paths=label_paths
                
        # in theory this can deal with nc files. would not bet on that being bug-free        
        if ext == "nc":
            self.file_paths = list(map(lambda x: os.path.join(root_dir, 
                                            os.path.basename(x).replace(".npy", 
                                                                        ".nc")),
                                         self.label_paths))
            self.read = read_nc
        elif ext == "npz":
            self.file_paths = list(map(lambda x: os.path.join(root_dir, 
                                            os.path.basename(x).replace(".npy", 
                                                                        ".npz")),
                                         self.label_paths))
           
            self.read = read_npz
            
        if check:
            copy_paths = np.copy(self.file_paths)
            label_remove = []
            for i,file in enumerate( copy_paths):
                if not os.path.exists(file):
                    self.file_paths.remove(file)
                    warnings.warn("removing {}".format(file), Warning)
                    if self.ext =="npz":
                        label_remove.append(self.label_paths[i])
                    elif ext =="nc":
                        label_remove.append(self.label_paths[i])
                                            
            for i in label_remove:
                self.label_paths.remove(i)
            
        self.normalizer = normalizer
        
        self.chunksize_x = chunksize_x
        self.chunksize_y = chunksize_y
        self.num_chunks = num_chunks
        self.variables = variables
        # for some reason I established the convention that I only compute cwp if lwp and iwp are loaded
        if self.variables is not None:
            if (0 in self.variables) and (1 in self.variables):
                self.channels = self.variables+1
                self.channels = np.hstack(([0],self.channels))
            else:
                self.channels=self.variables
        else:
            self.channels = np.arange(10)

        
        if self.plotting:
            filename = self.file_paths[0]
            print(filename)
            _, properties, _, _ = self.read(filename, self.variables)
            
            allowed_x = np.arange(self.chunksize_x//2, 
                                       properties.shape[1] - self.chunksize_x//2,
                                       self.chunksize_x)
            allowed_y = np.arange(self.chunksize_y//2, 
                                       properties.shape[2] - self.chunksize_y//2,
                                       self.chunksize_y)
            locations= list(itertools.product(
                    allowed_x, allowed_y))
            self.num_chunks = len(locations)
        
        
    def __len__(self):

        return len(self.file_paths)

    def __getitem__(self, info):
        """loads and processes a file

        Args:
            info (int): index of file to load. tuple handling no longer supported

        Returns:
            tuple: return filename, grid box averages of inputs, relative cloud amount per cell,
                    locations in degrees(?) and cloud mask
        """
        if isinstance(info, tuple):
            # load single tile
            idx, tile_idx = info
        else:
            idx, tile_idx = info, None

        filename = self.file_paths[idx]
        self.labelname = self.label_paths[idx]
        filebasename = filename[filename.find("A200"):filename.find("A200")+5]
        labelbasename = self.labelname[self.labelname.find("A200"):
                                       self.labelname.find("A200")+5]
        assert filebasename == labelbasename
        
        _, self.properties, cloud_mask, _ = self.read(filename, self.variables, liquid=True)
        
        
        labels = np.load(self.labelname).transpose()
        labels = np.where(np.isnan(labels), -1, labels)
        #only pixels that are labelled or cloudy are relevant (unlabelled cloudy is 0, labelled not cloudy is >=0)
        labels = np.where(labels+cloud_mask.squeeze()==-1, np.nan, labels)

        self.properties = np.where(cloud_mask+labels==-1,np.nan, self.properties)
        assert (self.properties.shape[1]==labels.shape[0] and 
                self.properties.shape[2]==labels.shape[1]), (self.properties.shape,
                                                        labels.shape)
        if not self.plotting:
            self.allowed_x = np.arange(self.chunksize_x//2, 
                                       self.properties.shape[1] - self.chunksize_x//2)
            self.allowed_y = np.arange(self.chunksize_y//2, 
                                       self.properties.shape[2] - self.chunksize_y//2)
            self.locations= []
            for _ in range(self.num_chunks):
                self.generate_sample_locations_wc(labels)
        else:
            self.allowed_x = np.arange(self.chunksize_x//2, 
                                       self.properties.shape[1] - self.chunksize_x//2,
                                       self.chunksize_x)
            self.allowed_y = np.arange(self.chunksize_y//2, 
                                       self.properties.shape[2] - self.chunksize_y//2,
                                       self.chunksize_y)
            self.locations= list(itertools.product(
                    self.allowed_x, self.allowed_y))     
            
        if self.normalizer is not None:
            self.properties = self.normalizer(self.properties)
        self.locations = np.array(self.locations)

        tiles_mean, labels_stats = self.sample_wc(self.properties, labels)
        cs_x, cs_y = self.chunksize_x//2, self.chunksize_y//2
        
        cloud_mask = np.stack([np.nanmean(cloud_mask[0,x[0]-cs_x : x[0]+cs_x,
                                x[1]-cs_y : x[1]+cs_y])  
                           for x in self.locations ])
        
        ignore = np.any(np.isnan(tiles_mean),1)
        tiles_mean[ignore]=np.zeros(tiles_mean.shape[1])
        
        #cloud_mask[ignore]=0
        #labels_stats[ignore]=np.zeros(labels_stats.shape[1])
        if self.ext == "nc":
            self.coords = get_coords(filename)
            self.locations  = self.coords[:,self.locations[:,0],
                                          self.locations[:,1]]       
            self.locations = self.locations.data.transpose()
            print(filename, np.min(self.locations), np.max(self.locations))
        #tiles_mean=np.where(np.isnan(tiles_mean), 0,tiles_mean)
        
        if self.transform == "log":
            tiles_mean=np.log(tiles_mean+1)
                    
        
        if tile_idx is not None:
            return (filename, tiles_mean[tile_idx], self.locations[tile_idx],
                labels_stats[tile_idx],cloud_mask[tile_idx])

        else:
            return filename, tiles_mean, self.locations, labels_stats,cloud_mask


    def __str__(self):
        return 'CUMULO'
    

        
        
    def generate_sample_locations_wc(self, labels):
        """ samples chunk centers that are reasonably far apart"""

        s_x = random.sample(list(self.allowed_x), 1)[0]
        s_y = random.sample(list(self.allowed_y), 1)[0]
        #chunk = labels[s_x-self.chunksize_x//2:s_x+self.chunksize_x//2,
        #           s_y-self.chunksize_y//2:s_y+self.chunksize_y//2]
            
        self.locations.append([int(s_x),int(s_y)])
        self.allowed_x = np.setdiff1d(self.allowed_x, np.arange(s_x-10,s_x+10))
        self.allowed_y = np.setdiff1d(self.allowed_y, np.arange(s_y-10, s_y+10))
        

    def sample_wc(self, props, labels):
        """ means of the physical properties in the chunk and the distribution
            of labels in the chunk"""
        cs_x, cs_y = self.chunksize_x//2, self.chunksize_y//2
        chunks = np.stack([list(map(np.nanmean, props[:,x[0]-cs_x : x[0]+cs_x,
                                x[1]-cs_y : x[1]+cs_y]))  
                           for x in self.locations ])
        labelchunks = np.stack([self.label_fct(labels[x[0]-cs_x : x[0]+cs_x,
                                          x[1]-cs_y : x[1]+cs_y]) 
                                          for x in self.locations])
        if np.any(np.isnan(labelchunks)):
            raise ValueError("got nan")
        
        return chunks, labelchunks



    

class SummedDataset(Dataset):
    def __init__(self, root_dir, label_dir, normalizer=None, 
                  indices=None,  variables = None,
                  chunksize = 100, output_tiles=True,
                 plotting = None, label_fct="rel", filt = None,
                 transform = None, tilesize=3, classmap=None,
                 subsample=None, rs=False, combine_strat="",
                 load=read_npz):
        """
        Dataset that loads files individually and can return the
        input/output pairs in different configurations

        Parameters
        ----------
        root_dir : str
            directory with input files.
        label_dir : str
            directory with label files
        normalizer : function or method, optional
            function that is applied to the inputs The default is None.
        indices : np.ndarray (n,), optional
            loads n files indexed this way. The default is None.
        variables : np.ndarray (v,), optional
            indices of variables to be loaded. The default is None.
        chunksize : int, optional
            side length of cells to average over. 
            The default is 100.
        output_tiles : Bool, optional
            if tiles of cell averages are to be returned The default is True.
        plotting : str, optional
             path to files where info about the geographical location of the
             inputs are. If not None does some interesting stuff so that the 
             output can the plotted. The default is None.
        label_fct : str, optional
            key for which function to use, the relative amount of labels per
            tile is always recommended. The default is "rel".
        filt : str, optional
            filter files by name, specifically month which is encoded in the
            name by day. The default is None.
        transform : str, optional
            apply transform to input. "log" is the only option.
            The default is None.
        tilesize : int, optional
            preferrably odd. The default is 3.
        classmap : dict, optional
            if input variables are in different order than usual.
            the dict maps the ints to the right ints. The default is None.
        subsample : int, optional
            amount of pixels to be subsampled. The default is None.
        rs : Bool, optional
            random s**t, shuffles the labels in relation to the inputs.
            This way the regression should stop working (turns out thats true)
            The default is False.
        combine_strat : str, optional
            if contains "nophase" load the files without making phase distinction
            if contains "cer" makes incloud averages for effective radius
            if contains "comb", combines the st and sc class. some other things could 
            be added. The default is "".
        load : function, optional
            function to use to load the files. The default is read_npz.

        Raises
        ------
        FileNotFoundError
            well...

        Returns
        -------
        the getitem method return the filename, the inputs, labels, and useless
        cloud mask

        """
        #whether to combine stratus and stratocumulus
        self.combine_strat = combine_strat
        #whether to populate a gridbox with pixels from random locations
        self.random_nonsense = rs
        #whether to subsample only a fraction of grid boxes
        self.subsample = subsample
        #possible to change order of labels
        self.classmap=classmap
        #logtransform the inputs
        self.transform = transform
        
        self.root_dir = root_dir
        self.label_dir = label_dir
        lq=True
        if "nophase" in combine_strat:
            lq=False
        self.load = lambda x : load(x,choice=variables,liquid=lq)
        self.plotting = plotting
        self.output_tiles = output_tiles
        if self.output_tiles:
            self.tilesize=tilesize
        else:
            self.tilesize=None
        
        if label_fct == "rel":#relative cloud type amount
            self.label_fct = label_stats_wc
        elif label_fct == "abs":#absolute cloud type amount
            self.label_fct = label_abs_wc
            
        
        self.file_paths = glob.glob(os.path.join(root_dir, "*.npz"))
        
        if len(self.file_paths) == 0:
            raise FileNotFoundError("no files in {}".format( root_dir))
        
        self.file_paths.sort()
        if indices is not None:
            if filt is None:
                self.file_paths = [self.file_paths[i] for i in indices]
            else:
                months = [1, -2, 1, 0, 1,0,1,1,0,1,0,1]
                comp_min = int(filt)*30
                comp_max = comp_min + 31 + np.sum(months[:int(filt)])
                file_paths = []
                for p in self.label_paths:
                    day = int(p[-12:-9])
                    if day>=comp_min and day<comp_max:
                        file_paths.append(p)
                self.file_paths=file_paths
       
                
        assert len(self.file_paths)>0, root_dir 
        for bn in copy(self.file_paths):
            if not os.path.exists(os.path.join(self.label_dir,
                                               os.path.basename(bn).replace("npz","npy"))):
                self.file_paths.remove(bn)
            
        
        self.normalizer = normalizer
        self.chunksize = chunksize
        self.variables = variables
        self.channels = variables
        if "cer" in self.combine_strat:
            self.incloud_vars = np.stack([np.argwhere(self.variables==x) 
                                          for x in [3,4,6,7,8]if x in self.variables]).reshape(-1)
            self.gba_vars = np.stack([np.argwhere(self.variables==x) 
                                      for x in [0,1,2,5,9,10] if x in self.variables]).reshape(-1)
        elif "nophase" in self.combine_strat:
            self.incloud_vars = np.stack([np.argwhere(self.variables==x) 
                                          for x in [3,4,5] if x in self.variables]).reshape(-1)
            self.gba_vars = np.stack([np.argwhere(self.variables==x) 
                                      for x in [0,1,2,6,7] if x in self.variables]).reshape(-1)
        else:
            tostack = [np.argwhere(self.variables==x) 
                                          for x in [6,7,8]if x in self.variables]
            if len(tostack)>0:
                self.incloud_vars = np.stack(tostack).reshape(-1)
            else:
                self.incloud_vars = np.array([])

            tostack = [np.argwhere(self.variables==x) 
                                      for x in [0,1,2,3,4,5,9,10] if x in self.variables]
            if len(tostack)>0:
                self.gba_vars = np.stack(tostack).reshape(-1)
            else:
                self.gba_vars = np.array([])
       
        
    def __len__(self):

        return len(self.file_paths)
    #@profile
    def __getitem__(self, info):    
        if isinstance(info, tuple):
            # load single tile
            idx, *tile_idx = info
        else:
            idx, tile_idx = info, None

        filename = self.file_paths[idx]
        failsafe =0
        while True:
            try:
                
                _,properties,cm,_ = self.load(filename)
                labels = np.load(os.path.join(self.label_dir, 
                                              os.path.basename(filename
                                                               ).replace("npz",
                                                                         "npy")))
                
                break
            
            except Exception as err:
                traceback.print_exc()
                time.sleep(5)
                failsafe+=1
                if failsafe>120:
                    raise LoadingError("dataloader timed out")
            
        
        if "comb" in self.combine_strat:
            newlabels = np.zeros((labels.shape[0]-1, labels.shape[1],
                                      labels.shape[2]))
            newlabels[:5] += labels[:5]
            newlabels[4:] += labels[5:]
            labels = newlabels.copy()
            del newlabels
        
        """ 
        if "nophase" in self.combine_strat:
            num_variables = len(self.variables)
            
            if 3 in self.variables and 4 in self.variables:
                num_variables-=1
                cer = properties[3]+properties[4]
            new_properties =np.zeros((num_variables, properties.shape[1], properties.shape[2]))
            new_properties[:3] = properties[:3]
            new_properties[3] = cer
            new_properties[4:] = properties[5:]
            properties=new_properties
            assert np.nanmax(properties)<1e6,np.where(properties>=1e6)
        """   
            
        if self.classmap is not None:
            labels = self.classmap(labels)
        
         
        labels = np.where(np.isnan(labels),-1,labels).astype(np.int64)
        if len(self.incloud_vars)>0:
            properties[self.incloud_vars] = np.where(labels[np.newaxis,...]==-1, 0,
                                                 properties[self.incloud_vars])
        #properties = np.vstack((properties,cm))
        properties = np.where(np.isnan(properties),0,properties).astype(np.float64)
        labels = transform_bincount(labels)
        
        properties = np.nancumsum(properties,1)
        properties = np.nancumsum(properties,2)
        
        cm=np.nancumsum(cm,1)
        cm=np.nancumsum(cm,2)
        
        combined = self.extract(np.vstack((properties, labels,cm)))
        properties = combined[:len(properties)]
        
        labels = combined[len(properties):len(properties)+len(labels)]
        #mask = combined[-len(mask):]
        #mask= mask[self.channels]
        cloud_mask = combined[-1][np.newaxis,...]
        cloud_amount_gb = np.sum(labels[1:], 0)
        if len(self.gba_vars)>0:
            properties[self.gba_vars] = properties[self.gba_vars]/(self.chunksize**2)#np.where(mask!=0,properties[:-1]/ mask, 0
        if len(self.incloud_vars)>0:
            properties[self.incloud_vars] = properties[self.incloud_vars]/cloud_amount_gb
        
        assert np.nanmax(properties)<1e6,np.where(properties>=1e6)
        tiles_mean = properties
        if self.normalizer is not None:
            tiles_mean = self.normalizer(tiles_mean)
        normsum = np.sum(labels, (0)).reshape(1,labels.shape[1],-1)
        labels_stats = np.where(normsum!=0,labels/normsum, 0)
        
        assert (tiles_mean.shape[1]==labels_stats.shape[1] and 
                tiles_mean.shape[2]==labels_stats.shape[2]), (tiles_mean.shape,
                                                        labels_stats.shape)
                                                              

        del properties, labels,  combined
        
        
        if self.output_tiles:
            temp = window_nd(tiles_mean, self.tilesize,steps=self.chunksize,
                             axis=(1,2)).copy()
            tiles_mean= np.copy(temp)
            temp = window_nd(labels_stats, self.tilesize,steps=self.chunksize,
                             axis=(1,2)).copy()
            labels_stats=np.copy(temp)
            temp = window_nd(cloud_mask, self.tilesize,steps=self.chunksize,
                             axis=(1,2)).copy()
            cloud_mask = np.copy(temp) 
            cloud_mask = cloud_mask[:,:,np.newaxis,...]
        else:
                
            ignore = np.any(np.isnan(tiles_mean) | np.isinf(tiles_mean), 0)
            tiles_mean[:, ignore]=0
            assert not (np.any(np.isinf(tiles_mean)) | np.any(np.isnan(tiles_mean)))
            #cloud_mask[ignore]=0
            labels_stats[:, ignore]=0
            #tiles_mean=np.where(np.isnan(tiles_mean), 0,tiles_mean)
        
        if self.transform == "log":
            tiles_mean=np.log(tiles_mean+1)
            
        if self.plotting is not None:
            nc_path = os.path.join(self.plotting, os.path.basename(filename).replace("npy", "nc"))
            self.coords = get_coords(nc_path)
            centers_1 = np.arange(int(self.chunksize//2),
                                                      int(self.coords.shape[1]
                                                          -self.chunksize//2),
                                                      self.chunksize)
            centers_2 = np.arange(int(self.chunksize//2),
                                                      int(self.coords.shape[2]
                                                          -self.chunksize//2),
                                                      self.chunksize)
            
            self.locations  = self.coords[:,centers_1][...,centers_2]       
            self.locations = self.locations.data
            assert self.locations.shape[1:]==labels_stats.shape[1:3], (self.locations.shape, labels_stats.shape)
        else:
            self.locations = None
        
        
        tiles_mean = tiles_mean.astype(np.float16)
        cloud_mask = cloud_mask.astype(np.float16)
        labels_stats = labels_stats.astype(np.float16)
        #assert np.all(not (np.isnan(tiles_mean) | np.isinf(tiles_mean)))
        #assert np.all(not (np.isnan(cloud_mask) | np.isinf(cloud_mask)))
        #assert np.all(not (np.isnan(labels_stats) | np.isinf(labels_stats)))
        
        if tile_idx is not None:
            x,y=tile_idx
            if self.output_tiles:
                o_s = self.tilesize//2
                return (filename, tiles_mean[x,y], 
                    labels_stats[x,y,:,o_s,o_s], cloud_mask[x,y])
            else:
                return(filename, tiles_mean[:,x,y], labels_stats[:,x,y], cloud_mask[:,x,y], self.locations[:,x,y])
        else:
            if self.output_tiles:
                o_s = self.tilesize//2
                return filename, tiles_mean, labels_stats[...,o_s,o_s], cloud_mask, self.locations
            return filename, tiles_mean, labels_stats, cloud_mask, self.locations


    def __str__(self):
        return 'SumSet'
    
    #@profile  
    def extract(self,arr):
        """
        gets the sum of each gridbox. as each element in the array is
        the sum of all the elements up until that (in dim 1 and 2), the
        elments outside of the grid box need to be subtracted to get only the
        grid sum. because the top left is subtracted twice, we add it again

        Parameters
        ----------
        arr : array, (n,x,y)
            recursive sum array

        Returns
        -------
        array, (n, (x-d)/d,(y-d)/d)
            sums of non-overlapping grid boxes

        """
        arr = np.pad(arr, ((0,0),(1,0), (1,0)),constant_values=0)
        d=self.chunksize
        if self.random_nonsense:
            d=1
            out = arr[:,d::,d::]+arr[:,:-d:,:-d:]-arr[:,:-d:,d::]-arr[:,d::,:-d:]
            out=out.transpose(1,2,0) #to x,y,n
            np.random.shuffle(out)
            out=out.transpose(1,0,2) # to y,x ,n
            np.random.shuffle(out)
            out=out.transpose(2,1,0) # to n,x,y
            arr = np.cumsum(np.cumsum(out,1),2)
            d=self.chunksize
            return arr[:,d::d,d::d]+arr[:,:-d:d,:-d:d]-arr[:,:-d:d,d::d]-arr[:,d::d,:-d:d]
        
        # can sumsample the data to save memory
        if self.subsample is not None:
            out = arr[:,d::,d::]+arr[:,:-d:,:-d:]-arr[:,:-d:,d::]-arr[:,d::,:-d:]
            x = np.random.randint(low=0, high=out.shape[1], size=self.subsample)
            y = np.random.randint(low=0, high=out.shape[2], size=self.subsample)
            return out[:,x,y, np.newaxis]
        else:
            return arr[:,d::,d::]+arr[:,:-d:,:-d:]-arr[:,:-d:,d::]-arr[:,d::,:-d:]
        
        
        
class ESACCIDataset(Dataset):
    def __init__(self, root_dir, normalizer=None, 
                  indices=None, variables = None,
                  chunksize = 100, output_tiles=True,
                 plotting = None, filt = None,tilesize=3,
                 transform=None, subsample=None, time_type=None):
        """
        

        Parameters
        ----------
        root_dir : str
            ESACCI Files
        normalizer : function or method, optional
            function that is applied to the inputs The default is None.
        indices : np.ndarray (n,), optional
            loads n files indexed this way. The default is None.
        variables : np.ndarray (v,), optional
            indices of variables to be loaded. The default is None.
        chunksize : int, optional
            side length of cells to average over. 
            The default is 100.
        output_tiles : Bool, optional
            if tiles of cell averages are to be returned The default is True.
        plotting : str, optional
             path to files where info about the geographical location of the
             inputs are. If not None does some interesting stuff so that the 
             output can the plotted. The default is None.
         filt : str, optional
           filter files by name, specifically month which is encoded in the
           name by day. The default is None.
        tilesize : int, optional
            preferrably odd. The default is 3.
        transform : str, optional
            something like "log". The default is None.
         subsample : int, optional
             amount of pixels to be subsampled. The default is None.
        time_type : str, optional
            "dn" if daily mean, "de" if daily instantaneous, "mn" if 
            monthly mean. The default is None, which will probably crash.
            
        Returns
        -------
        getitme return filename, variables, geolocations

        """
        self.subsample = subsample
        self.transform=transform
        self.root_dir = root_dir
        self.plotting = plotting
        self.output_tiles = output_tiles
        if self.output_tiles:
            self.tilesize=tilesize
        else:
            self.tilesize=None
        
        self.file_paths = glob.glob(os.path.join(root_dir, "*.npy"))
        self.file_paths.sort()
        if indices is not None:
            self.file_paths=[self.file_paths[x] for x in indices]
            
        self.time=time_type
        if self.time == "dn":
            #get tuples of the ascending and descending orbits
            #very unelegant but works
            self.file_paths=list(np.unique([os.path.basename(x)[2:]
                                            for x in self.file_paths]))
            
            self.file_paths =[(os.path.join(root_dir,"0_"+x),
                               os.path.join(root_dir,"1_"+x))
                              for x in self.file_paths
                              if (os.path.exists((os.path.join(root_dir,"0_"+x))) 
                                  and os.path.exists((os.path.join(root_dir,"1_"+x)))) ]
            
        assert len(self.file_paths)>0, os.listdir(root_dir)
        
        self.normalizer = normalizer
        self.chunksize = chunksize
        self.variables = variables
        
        tostack = [np.argwhere(self.variables==x) 
                                      for x in [6,7,8]if x in self.variables]
        if len(tostack)>0:
            self.incloud_vars = np.stack(tostack).reshape(-1)
        else:
            self.incloud_vars =np.array([],dtype=bool)
        
        tostack = [np.argwhere(self.variables==x) 
                                  for x in [0,1,2,3,4,5,9,10] if x in self.variables]
        if len(tostack)>0:
            self.gba_vars = np.stack(tostack).reshape(-1)
        else:
            self.gba_vars = np.array([],dtype=bool)

        
        
    def __len__(self):
        return len(self.file_paths)
    
    #@profile
    def __getitem__(self, info):
        if isinstance(info, tuple):
            # load single tile
            idx, *tile_idx = info
        else:
            idx, tile_idx = info, None

        if self.time=="de" or self.time=="mn":
            filenames = [self.file_paths[idx]]
        elif self.time=="dn":
            filenames =[ self.file_paths[idx][0], self.file_paths[idx][1]]
            stack=[]
            if isinstance(self.subsample,int):
                self.subsample=(self.subsample,int(time.process_time()))    
            elif isinstance(self.subsample,tuple):
                self.subsample=(self.subsample[0],int(time.process_time()))    
            elif self.subsample is not None:
                print(type(self.subsample))
                raise Exception("wtf")
                
            
        else:
            raise ValueError("wtf")
        for filename in filenames:
            failsafe =0
            while True:
                try:
                    properties = np.load(filename)
                    break
                
                except Exception as err:
                    traceback.print_exc()
                    time.sleep(5)
                    failsafe+=1
                    if failsafe>120:
                        raise LoadingError("loading timed out")
                        
            locations = properties[-3:]
            tiles_mean = properties[ self.variables]
            if self.normalizer is not None:
                tiles_mean = self.normalizer(tiles_mean)
    
            del properties
            
            if self.chunksize>1:
                if len(self.incloud_vars)>0:
                    cloud_amount_gb = ~np.isnan(tiles_mean)[self.incloud_vars].reshape(-1, 
                                                                tiles_mean.shape[1],
                                                                tiles_mean.shape[2])
                else:
                    cloud_amount_gb = np.zeros((1,tiles_mean.shape[1],tiles_mean.shape[2]))
                cloud_amount_gb = np.cumsum(cloud_amount_gb, axis=1)
                cloud_amount_gb = np.cumsum(cloud_amount_gb, axis=2)
                
                temp = np.nancumsum(tiles_mean,axis=1, dtype=np.float64)
                temp = np.nancumsum(temp,axis=2, dtype=np.float64)
           
                locstemp =np.nancumsum(locations,axis=1,dtype=np.float64)
                locstemp = np.nancumsum(locstemp, axis=2,dtype=np.float64)
                temp = np.vstack((temp,locstemp,cloud_amount_gb))
                
                temp = self.extract(temp)
                
                
                cloud_amount_gb = temp[-1]
                cloud_mask = cloud_amount_gb>0
                locations = temp[-4:-1]/(self.chunksize**2)
                tiles_mean= temp[:-4]
                del temp,locstemp
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",category=RuntimeWarning)
                    tiles_mean[self.incloud_vars] = tiles_mean[self.incloud_vars]/cloud_amount_gb
            if not self.output_tiles:
                tiles_mean[self.gba_vars] = tiles_mean[self.gba_vars]/(self.chunksize**2)
                tiles_mean = tiles_mean.reshape(len(self.variables),-1)
                locations=locations.reshape(3,-1)
                assert tiles_mean.shape[1]==locations.shape[1],(tiles_mean.shape,locations.shape)  
                ignore = np.any(np.isnan(tiles_mean) | np.isinf(tiles_mean), 0)
                tiles_mean[:, ignore]=0
                
            
            elif self.output_tiles:
                #this gets non-opverlapping patches of non-overlapping grid cell averages
                temp = window_nd(tiles_mean[:,::self.chunksize,::self.chunksize],
                                 self.tilesize,self.tilesize, axis=(1,2)).copy()
                tiles_mean= np.copy(temp)
                locations = np.copy(window_nd(locations[:,::self.chunksize,::self.chunksize]

                           ,self.tilesize, self.tilesize,axis=(1,2)))

                locations = locations.reshape(-1, locations.shape[2],locations.shape[3],

                                           locations.shape[4])

                tiles_mean = tiles_mean.reshape(-1, tiles_mean.shape[2],

                                              tiles_mean.shape[3],

                                              tiles_mean.shape[4])



                del temp 
                cloud_mask=~np.isnan(tiles_mean)[self.incloud_vars].reshape(-1, 
                                                                tiles_mean.shape[1],
                                                                tiles_mean.shape[2])
            
            if self.transform == "log":
                tiles_mean=np.log(tiles_mean+1)
                
            if self.plotting is not None:
                nc_path = os.path.join(self.plotting, os.path.basename(filename).replace("npy", "nc"))
                self.coords = get_coords(nc_path)
                centers_1 = np.arange(int(self.chunksize//2),
                                                          int(self.coords.shape[1]
                                                              -self.chunksize//2),
                                                          self.chunksize)
                centers_2 = np.arange(int(self.chunksize//2),
                                                          int(self.coords.shape[2]
                                                              -self.chunksize//2),
                                                          self.chunksize)
                
                self.locations  = self.coords[:,centers_1][...,centers_2]       
                self.locations = self.locations.data
                #assert self.locations.shape[1:]==labels_stats.shape[1:3], (self.locations.shape, labels_stats.shape)
            else:
                self.locations = None
                
            tiles_mean = tiles_mean.astype(np.float32)
            if self.time=="dn":
                stack.append((tiles_mean,locations,cloud_mask))
            
        if self.time=="dn":
            stack=np.stack(stack)
            tiles_mean = np.nanmean(np.stack(stack[:,0]),0)
            locations = np.nanmean(np.stack(stack[:,1]),0)
            cloud_mask = np.nanmean(np.stack(stack[:,2]),0)
            
        if tile_idx is not None:
            if self.output_tiles:
                o_s = self.tilesize//2
                return filename, tiles_mean[:,tile_idx], locations[:,tile_idx],cloud_mask[:,tile_idx]
            else:
                return(filename, tiles_mean[:,tile_idx], locations[:,tile_idx])
        else:
            if self.output_tiles:
                return filename, tiles_mean, locations, cloud_mask
            return filename, tiles_mean, locations


    def __str__(self):
        return 'ESACCISet'
    
    def extract(self,arr):
        """
        gets the sum of each gridbox. as each element in the array is
        the sum of all the elements up until that (in dim 1 and 2), the
        elments outside of the grid box need to be subtracted to get only the
        grid sum. because the top left is subtracted twice, we add it again

        Parameters
        ----------
        arr : array, (n,x,y)
            recursive sum array

        Returns
        -------
        array, (n, (x-d+1),(y-d+1))
            sums of overlapping grid boxes

        """
        arr = np.pad(arr, ((0,0),(1,0), (1,0)),constant_values=0)
        d=self.chunksize
        
        # can sumsample the data to save memory
        if isinstance(self.subsample,tuple):
            out = arr[:,d::,d::]+arr[:,:-d:,:-d:]-arr[:,:-d:,d::]-arr[:,d::,:-d:]
            rng = np.random.default_rng(self.subsample[1])
            x = rng.integers(low=0,high=out.shape[1],size=self.subsample[0])
            y = rng.integers(low=0,high=out.shape[2],size=self.subsample[0])
            
            return out[:,x,y, np.newaxis]            
        elif self.subsample is not None:
            out = arr[:,d::,d::]+arr[:,:-d:,:-d:]-arr[:,:-d:,d::]-arr[:,d::,:-d:]
            x = np.random.randint(low=0, high=out.shape[1], size=self.subsample)
            y = np.random.randint(low=0, high=out.shape[2], size=self.subsample)
            return out[:,x,y, np.newaxis]
        else:
            return arr[:,d::,d::]+arr[:,:-d:,:-d:]-arr[:,:-d:,d::]-arr[:,d::,:-d:]    
        
def summed_collate(arr):
    """
    shuffles the grid boxes

    Parameters
    ----------
    arr : array, should be [Batch,x/cx-pad_x, y/cy-pad_y,prop,tile_x,tile_y]
         or [Batch, prop,x/cx, y/cy]
        DESCRIPTION.

    Returns
    -------
    dem grid boxes

    """
    arr=np.stack(arr)
    fn, mean, labels, mask,*_ = arr.transpose()
    fn=np.stack(fn)
    mean=np.stack(mean)
    labels=np.stack(labels)
    mask=np.stack(mask)
    if len(mean.shape)==4:
        mean = mean.transpose(2,3,0,1)
        labels = labels.transpose(2,3,0,1)
        mask = mask.transpose(2,3,0,1)
        mean=mean.reshape(-1,mean.shape[3])
        labels=labels.reshape(-1,labels.shape[3])
        mask=mask.reshape(-1,mask.shape[3])
        
    elif len(mean.shape)==6:
        mean=mean.reshape(-1,mean.shape[3],mean.shape[4], mean.shape[5])    
        labels=labels.reshape(-1,labels.shape[3])    
        mask=mask.reshape(-1, mask.shape[3],mask.shape[4], mask.shape[5])   
    else:
        raise ValueError("dont understand input shape {}".format(arr.shape))
    
    interesting = np.array(np.any(labels[:,1:]>0, 1))
    reduce_shape = tuple(range(len(mean[:].shape)))[1:]
    interesting *= np.array(np.all(mean[:]>0,axis=reduce_shape))
    mean = mean[interesting]
    labels=labels[interesting]
    mask = mask[interesting]
    shuffle_idx = np.arange(mean.shape[0])
    np.random.shuffle(shuffle_idx)
    mean = torch.tensor(mean[shuffle_idx])
    labels = torch.tensor(labels[shuffle_idx])
    mask = torch.tensor(mask[ shuffle_idx])
    arr = (fn, mean, labels, mask)
    return arr 

def transform_collate(arr):
    """
    shuffles the grid boxes and transform (like flip and rotate)

    Parameters
    ----------
    arr : array, should be [Batch,x/cx-pad_x, y/cy-pad_y,prop,tile_x,tile_y]
         or [Batch, prop,x/cx, y/cy]
        DESCRIPTION.

    Returns
    -------
    dem grid boxes

    """
    arr=np.stack(arr)
    fn, mean, labels, mask,*_ = arr.transpose()
    fn=np.stack(fn)
    mean=np.stack(mean)
    labels=np.stack(labels)
    mask=np.stack(mask)
    if len(mean.shape)==4:
        mean = mean.transpose(2,3,0,1)
        labels = labels.transpose(2,3,0,1)
        mask = mask.transpose(2,3,0,1)
        mean=mean.reshape(-1,mean.shape[3])
        labels=labels.reshape(-1,labels.shape[3])
        mask=mask.reshape(-1,mask.shape[3])
        
    elif len(mean.shape)==6:
        mean=mean.reshape(-1,mean.shape[3],mean.shape[4], mean.shape[5])    
        labels=labels.reshape(-1,labels.shape[3])    
        mask=mask.reshape(-1, mask.shape[3],mask.shape[4], mask.shape[5])   
    else:
        raise ValueError("dont understand input shape {}".format(arr.shape))
    
    interesting = np.array(np.any(labels[:,1:]>0, 1))
    reduce_shape = tuple(range(len(mean[:].shape)))[1:]
    interesting *= np.array(np.all(mean[:]>0,axis=reduce_shape))
    mean = mean[interesting]
    labels=labels[interesting]
    mask = mask[interesting]
    shuffle_idx = np.arange(mean.shape[0])
    np.random.shuffle(shuffle_idx)
    
    mean = torch.tensor(mean[shuffle_idx])
    labels = torch.tensor(labels[shuffle_idx])
    mask = torch.tensor(mask[ shuffle_idx])
    mean_h = T.hflip(mean)
    mean_v = T.vflip(mean)
    mean_r = T.hflp(mean_v)
    mean = torch.cat([mean,mean_v,mean_h,mean_r])
    mean = torch.cat([labels,labels,labels,labels])
    mean = torch.cat([mask,mask,mask,mask])
    arr = (fn, mean, labels, mask)
    return arr 


def ESACCI_collate(arr):
    arr=np.stack(arr)
    name, arr, locs=arr.transpose()    
    assert len(arr)==len(locs),(arr.shape,locs.shape)
    arr = np.hstack(arr)
    arr = arr.transpose()
    name = np.hstack(name)
    locs=np.hstack(locs).transpose()
    
    return  name, torch.tensor(arr), torch.tensor(locs)


def ICON_collate(arr):
    #stack both filenames and output
    arr=np.stack(arr)
    #ignore the filename move batch size back
    filename, arr, locations = arr.transpose()
    #now batch size is in front
    ##stack along some axis that is not the first
    arr = np.hstack(arr)
    filename = np.hstack(filename)
    locations = np.hstack(locations)
    vmany = arr.shape[0]
    #pixelwise and for every variable
     
    arr = arr.reshape(vmany,-1)
    locations = locations.reshape(3, -1)
    print("collate", torch.unique(torch.tensor(locations)[2]), locations.shape)
    return filename,torch.tensor(arr), torch.tensor(locations)


class ICONDataset(Dataset):
    def __init__(self, root_dir,grid, normalizer=None, 
                  indices=None, variables = None,
                   output_tiles=True,
                 plotting = None, filt = None,tilesize=3,
                 transform=None):
        """
        

        Parameters
        ----------
        root_dir : str
            location of transformed ICON npy files
        grid : str
            grid the files are regridded to
        normalizer : function or method, optional
            function that is applied to the inputs The default is None.
        indices : np.ndarray (n,), optional
            loads n files indexed this way. The default is None.
        variables : np.ndarray (v,), optional
            indices of variables to be loaded. The default is None.
        output_tiles : Bool, optional
            if tiles of cell averages are to be returned The default is True.
        plotting : str, optional
             path to files where info about the geographical location of the
             inputs are. If not None does some interesting stuff so that the 
             output can the plotted. The default is None.
         filt : str, optional
           filter files by name, specifically month which is encoded in the
           name by day. The default is None.
        transform : str, optional
            something like "log". The default is None.
        Returns
        -------
        None.

        """
        
        self.transform=transform
        self.root_dir = root_dir
        self.plotting = plotting
        self.output_tiles = output_tiles
        if self.output_tiles:
            self.tilesize=tilesize
        else:
            self.tilesize=None
        
        self.file_paths = glob.glob(os.path.join(root_dir, "{}*.npz".format(grid)))
        
            
        assert len(self.file_paths)>0      
        
        self.normalizer = normalizer
        # this allows the same variable indexing for ICON as for CUMULO without bothering with the source files
        variable_translator = {0:0, 1:2,2:3, 6:4, 10:5}
        self.variables = np.array([variable_translator[i] for i in variables])
        

        
        
    def __len__(self):

        return len(self.file_paths)

    def __getitem__(self, info):

        if isinstance(info, tuple):
            # load single tile
            idx, *tile_idx = info
        else:
            idx, tile_idx = info, None

        
        filename = self.file_paths[idx]
        
        
        failsafe =0
        while True:
            try:
                #this should be [5,96,192]
                #wehre 5=[ps, cllvi, clivi, topmax, ts]
                properties = np.load(filename)
                locations = properties["locations"] 
                properties = properties["properties"]
                
                break
            
            except Exception as err:
                traceback.print_exc()
                time.sleep(5)
                failsafe+=1
                if failsafe>120:
                    raise LoadingError("loading timed out")
                    
        properties = np.vstack((properties[np.newaxis,1]+properties[np.newaxis,2],
                                properties))
        tiles_mean = properties[ self.variables]
        tiles_mean[-2] /= 100 #from Pa to hPa
        tiles_mean[:3] *= 1000 # form kg/m^2 to g/m^2
        if self.normalizer is not None:
            tiles_mean = self.normalizer(tiles_mean)
        
                                

        del properties
        ignore = np.any(np.isnan(tiles_mean) | np.isinf(tiles_mean), 0)
        tiles_mean[:, ignore]=0
        
 
        if self.output_tiles:
            
            temp = window_nd(tiles_mean, self.tilesize,self.tilesize, axis=(1,2)).copy()

            tiles_mean= np.copy(temp)

            locations = np.copy(window_nd(locations, self.tilesize, self.tilesize, axis=(1,2)))

            del temp

            locations = locations.reshape(-1, locations.shape[2],locations.shape[3],

                                           locations.shape[4])

            tiles_mean = tiles_mean.reshape(-1, tiles_mean.shape[2],

                                              tiles_mean.shape[3],

                                              tiles_mean.shape[4])


            tiles_mean= np.copy(temp)
            del temp
        
        if self.transform == "log":
            tiles_mean=np.log(tiles_mean+1)
            
        if self.plotting is not None:
            nc_path = os.path.join(self.plotting, os.path.basename(filename).replace("npy", "nc"))
            self.coords = get_coords(nc_path)
            centers_1 = np.arange(int(self.chunksize//2),
                                                      int(self.coords.shape[1]
                                                          -self.chunksize//2),
                                                      self.chunksize)
            centers_2 = np.arange(int(self.chunksize//2),
                                                      int(self.coords.shape[2]
                                                          -self.chunksize//2),
                                                      self.chunksize)
            
            self.locations  = self.coords[:,centers_1][...,centers_2]       
            self.locations = self.locations.data
            assert self.locations.shape[1:]==labels_stats.shape[1:3], (self.locations.shape, labels_stats.shape)
        else:
            self.locations = None
            
        tiles_mean = tiles_mean.astype(np.float32)
        
        if tile_idx is not None:
            x,y=tile_idx
            if self.output_tiles:
                o_s = self.tilesize//2
                return (filename, tiles_mean[x,y], locations[x,y])
            else:
                return(filename, tiles_mean[:,x,y], locations[:,x,y])
        else:
            if self.output_tiles:
                o_s = self.tilesize//2
                return filename, tiles_mean, locations
            return filename, tiles_mean, locations


    def __str__(self):
        return 'ICONSet'

if __name__ == "__main__":  

    # try loading precomputed 3x3 tiles
    load_path = "../DATA/npz/label/"

    dataset = CumuloDataset(load_path, ext="npz")

    for instance in dataset:

        filename, radiances, properties, rois, labels = instance
