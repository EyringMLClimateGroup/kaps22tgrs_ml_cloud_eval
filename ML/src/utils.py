"""large parts from CUMULO, but few are used"""
import glob
import numpy as np
import os
import pickle
import time

import multiprocessing as mp
import matplotlib.pyplot as plt
from itertools import product
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm

def get_path_basename(filename, rep=".png"):
    return os.path.basename(filename).replace(rep, "")

def load_dir_npys(dir_path):
    paths = glob.glob(os.path.join(dir_path, "*.npy"))
    
    if len(paths) == 0:
        print("no .npy in", dir_path)
        
    for array in paths:
        yield get_path_basename(array, ".npy"), np.load(array)

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

def save_instance(instance, save_path):

    with open(save_path,"wb") as pickle_out:
        pickle.dump(instance, pickle_out)

def load_instance(instance, load_path):

    with open(load_path,"rb") as pickle_in:
        instance = pickle.load(pickle_in)

    return instance

def get_dataset_statistics(dataset,sampler,
                           inp_idx, lab_idx, use_cuda=True, hist=0, collate=None):
    """
    Computes mean and standard deviation of dataset.
    Can also save a histogram but that is slow

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        dataset
    sampler : torch.utils.data.sampler
        sampler to use for dataloader
    inp_idx : int
        which index of the dataloader output corresponds to the input
    lab_idx : int
        which index of the dataloader output corresponds to the labels
    use_cuda : Bool, optional
        If cuda can be used. The default is True.
    hist : boold, optional
        if a hsitogram shsould be computed. The default is 0.
    collate : fct, optional
        collate function to use for dataloader The default is None.

    Returns
    -------
    mean and std per vairable

    """
    names = ["lwp", "iwp",  "reffclw", "reffcli", "cod", "ptop", "htop", 
             "ttop", "emiss", "tsurf"]
    batch_size=min(20, len(dataset))
    varstring=""
    for i in dataset.variables:
        varstring+=str(i)
    if not os.path.exists(os.path.join("minmax"+varstring,
                                       "min{}.npy".format(len(dataset)))):
        hist=0
    if hist:
        filenames=[]
        ma = np.load(os.path.join("minmax"+varstring,
                                  "max{}.npy".format(len(dataset))),
                                  allow_pickle=True)
        mi = np.load(os.path.join("minmax"+varstring,
                                  "min{}.npy".format(len(dataset)))
                                 , allow_pickle=True)
        try:
            mi=mi.cpu()
            ma=ma.cpu()+1
        except Exception:
            pass
        bins = []
        histithingy=[]
        print(mi,ma)
        for i in range(len(ma)):
            bins.append(np.linspace(mi[i],ma[i], num=min(1000,int(batch_size))))
            histithingy.append(np.zeros(len(bins[i])))
    train_sampler = sampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,sampler=train_sampler, collate_fn=collate,
                                             batch_size=batch_size, shuffle=False, 
                                             num_workers=16)

    if not os.path.exists("minmax"+varstring):
        os.mkdir("minmax"+varstring)
        
    len_c = len(dataset.channels)
    sum_x = torch.zeros(1,len_c)
    sum_y = torch.zeros(1,9)
    std = torch.zeros(1,len_c )
    std_y = torch.zeros(1,9 )
    
    if use_cuda:
        sum_x = sum_x.cuda()
        sum_y = sum_y.cuda()
        std = std.cuda()
        std_y = std_y.cuda()
        
    nb_tiles = 0
    pbar = tqdm(enumerate(dataloader))
    for now,x in pbar:
        pbar.set_description("{}/{}".format(now,len(dataloader)))
        
        tiles = x[inp_idx].float()
        labels = x[lab_idx].float()
        assert not np.any(np.isnan(tiles.numpy()))
        
        nb_tiles += len(tiles)
        
        if use_cuda:
            tiles = tiles.cuda()
            labels = labels.cuda()
        """    
        if not os.path.exists(os.path.join("minmax"+varstring, 
                                           "min{}.npy".format(len(dataset)))):
            try:
                ma = torch.max(torch.cat((tiles, ma[0].view(1,-1))),axis=0)
                mi = torch.min(torch.cat((tiles, mi[0].view(1,-1))),axis=0)
            except Exception as err:
                print(err)
                ma=torch.max(tiles,axis=0)
                mi = torch.min(tiles,axis=0)
        if hist:
            fn = x[0]
            if fn not in filenames:
                filenames.append(fn)
                print(fn, file=open("filenames{}".format(len(dataset)), "w"))
                
            for i in range(tiles.shape[1]):
                histi=torch.histc( tiles[:,i], bins=len(bins[i]), min =min(bins[i]),
max=max(bins[i]))
                histithingy[i]+=histi.cpu().numpy()
        """
        tiles = tiles.transpose(0,1).reshape(len_c,-1).transpose(0,1)   
        sum_x += torch.sum(tiles, (0))
        sum_y += torch.sum(labels,0)
        
    m = (sum_x / nb_tiles).reshape(1, len_c)
    ml =(sum_y/nb_tiles).reshape(1,9)
    
    try:
        
        np.save(os.path.join("minmax"+varstring,"max{}.npy".format(len(dataset))),ma)
        np.save(os.path.join("minmax"+varstring,"min{}.npy".format(len(dataset))),mi)
    except Exception:
        print("nothing new saved")
    pbar=tqdm(enumerate(dataloader))  
    for now,x in pbar:
        pbar.set_description("{}/{}".format(now,len(dataloader)))
        tiles = x[inp_idx].float()
        
        tiles = tiles.transpose(0,1).reshape(len_c,-1).transpose(0,1)   
        assert not np.any(np.isnan(tiles.numpy()))
        if use_cuda:
            tiles = tiles.cuda()

        std += torch.sum((tiles - m).pow(2), (0), keepdim=True)
        std_y += torch.sum((labels - ml).pow(2), (0), keepdim=True)
        
    s = ((std / nb_tiles)**0.5)
    sl = ((std_y / nb_tiles)**0.5)
    
    if use_cuda:
        m = m.cpu()
        s = s.cpu()
        sl = sl.cpu()
        ml = ml.cpu()
        
    if hist:
        fig, ax = plt.subplots(1,tiles.shape[1], sharey=True)
        ax=ax.flatten()
            
        for i in range(tiles.shape[1]):
            print(bins[i].shape)
            ax[i].bar(bins[i],histithingy[i], log=True)    
            ax[i].set_title(names[dataset.variables[i]])
      
        fig.tight_layout()
        fig.savefig("dataset_hist{}.png".format(len(dataset)))
    print("label stats", ml, sl)
    return  m.reshape(len_c,1,1).numpy(), s.reshape(len_c,1,1).numpy()

class Normalizer(object):

    def __init__(self, mean, std, choice=None):
        if choice is None:
            self.mean = mean
            self.std = std
        else:
            print(mean.shape, std.shape)
            self.mean = mean[choice]
            self.std = std[choice]

    def __call__(self, image):

        return (image - self.mean) / self.std

class TileExtractor(object):

    def __init__(self, t_width=3, t_height=3):

        self.t_width = t_width
        self.t_height = t_height

    def __call__(self, image):

        img_width = image.shape[1]
        img_height = image.shape[2]

        nb_tiles_row = img_width // self.t_width
        nb_tiles_col = img_height // self.t_height

        tiles = []
        locations = []

        for i in range(nb_tiles_row):
            for j in range(nb_tiles_col):

                tile = image[:, i * self.t_width: (i+1) * self.t_width, j * self.t_height: (j+1) * self.t_height]

                tiles.append(tile)

                locations.append(((i * self.t_width, (i+1) * self.t_width), (j * self.t_height, (j+1) * self.t_height)))

        tiles = np.stack(tiles)
        locations = np.stack(locations)

        return tiles, locations

# ------------------------------------------------------------ MODIS HELPERS

def get_idx_cloudy_tiles(tiles, cm_idx=24):

    return np.sum(tiles[:, cm_idx], axis=(1, 2)) > 6

def get_most_frequent_label_tile(labels, single_tile=True):

    if single_tile:

        # if no occurences
        if np.sum(labels) == 0:
            labels = -np.ones(1)

        else:
            class_dist = np.sum(labels, axis=(1, 2))
            labels = [np.argmax(class_dist)]

    else:

        occurrences = np.sum(labels, axis=(2, 3))
        labels = np.argmax(occurrences, axis=1)
        labels[np.sum(occurrences, axis=1) == 0] = -1

    return np.asarray(labels, dtype=np.int8)

def get_avg_tiles(dataset, allowed_idx=None, ext="npy", random=True):

    paths = np.arange(len(dataset.file_paths))

    if allowed_idx is not None:
        paths = [paths[i] for i in allowed_idx]
    
    indices = list(product(paths,range(dataset.x_size),
                            range(dataset.y_size)))
    if random:    
        return SubsetRandomSampler(indices)
    else:
        return indices

def get_avg_all(dataset, allowed_idx=None, ext="npy", random=True):

    paths = np.arange(len(dataset.file_paths))

    if allowed_idx is not None:
        paths = [paths[i] for i in allowed_idx]
    
    #indices = list(product(range(len(paths)),range(dataset.x_size),
    #                        range(dataset.y_size)))
    if random:
        return SubsetRandomSampler(paths)
    else:
        return paths

def get_chunky_sampler(dataset, allowed_idx=None, ext="npz"):

    indices = []
    paths = dataset.file_paths.copy()

    if allowed_idx is not None:
        paths = [paths[i] for i in allowed_idx]
    
    for i in range(len(paths)):
        indices += [(i, j) for j in range(dataset.num_chunks)]
        
    return SubsetRandomSampler(indices)

def get_sequential_chunks(dataset, allowed_idx=None, ext="npz"):

    indices = []
    paths = dataset.file_paths.copy()

    if allowed_idx is not None:
        paths = [paths[i] for i in allowed_idx]
    
    for i in range(len(paths)):
        indices += [(i, j) for j in range(dataset.num_chunks)]
        
    return indices

def tile_collate(swath_tiles):
    
    data = np.vstack([s["tiles"] for s in swath_tiles])
    target = np.hstack([s["labels"] for s in swath_tiles])

    return {"tiles": torch.from_numpy(data), "labels": torch.from_numpy(target)}

class TileDataset(Dataset):
    """ load Cumulo a tile at a time using npy files containing multiple tiles (grouped by swath). 
        Tiles are picked randomly from the entire dataset, not swath-wise"""

    def __init__(self, root_dir, normalizer=None, indices=None, channels = np.arange(13)):
        """
        Args:
            root_dir (string): directory containing the tiles (collected by swath = a file per swath and multiple tiles)
        """
        
        self.root_dir = root_dir
        self.swath_paths = os.listdir(self.root_dir)
        self.channels = channels
        if len(self.swath_paths) == 0:
            print("no npy files in", self.root_dir)
        
        if indices is not None:
            self.swath_paths = [self.swath_paths[i] for i in indices]
        
        self.normalizer = normalizer

    def __len__(self):

        return len(self.swath_paths)

    def __getitem__(self, info):

        if isinstance(info, tuple):

            # load single tile
            swath_idx, tile_idx = info

            swath_path = os.path.join(self.root_dir, self.swath_paths[swath_idx])
            while True:
                try:    
                    data = np.load(swath_path)[tile_idx]
                except IOError:
                    time.sleep(5)
                    continue
                break

        else:

            # load all swath's tiles
            swath_path = os.path.join(self.root_dir, self.swath_paths[info])
            data = np.load(swath_path)

        if len(data.shape) == 3:

            x = data[self.channels]
            labels = get_most_frequent_label_tile(data[-8:], single_tile=True)

        elif len(data.shape) == 4:

            x = data[:, self.channels]
            labels = get_most_frequent_label_tile(data[:, -8:], single_tile=False)

        if self.normalizer is not None:
            x = self.normalizer(x)

        return {"tiles": x, "labels": labels}

class CumuloDataset(Dataset):
    """ load Cumulo a swath at a time."""

    def __init__(self, root_dir="./datasets/cumulo/", tile_extractor=None, normalizer=None, channels = np.arange(13)):
        """
        Args:
            root_dir (string): directory containing CUMULO npys
        """
        
        if isinstance(root_dir, str):
            self.swath_paths = glob.glob(os.path.join(root_dir, "*.npy"))
        else:
            self.swath_paths = root_dir

        if len(self.swath_paths) == 0:
            print("no npy files loaded")

        self.tile_extractor = tile_extractor
        self.normalizer = normalizer
        self.channels = channels

    def __len__(self):

        return len(self.swath_paths)

    def __getitem__(self, idx):

        """ return multiple tiles"""

        swath_name = self.swath_paths[idx]
        swath = np.load(swath_name)

        if self.tile_extractor is not None: 
            tiles, locations = self.tile_extractor(swath)

            cloudy_idx = get_idx_cloudy_tiles(tiles)

            if self.normalizer is not None:
                cloudy_tiles = self.normalizer(tiles[cloudy_idx][:, self.channels])
            else:
                cloudy_tiles = tiles[cloudy_idx]

            return {"path": swath_name, "shape": swath.shape, "tiles": cloudy_tiles, "locations": locations[cloudy_idx]}

        else:

            labels = get_most_frequent_label_swath(swath[-8:])

            return {"path": swath_name, "swath": swath, "labels": labels}


def get_ds_labels_stats(ds, sampler, brake):
    dataloader = torch.utils.data.DataLoader(ds,sampler=sampler,
                                             batch_size=100, shuffle=False, 
                                             num_workers=8)
    labels = np.zeros(9)
    for i,x in enumerate(dataloader):
        labels += np.mean(x[3].numpy(),0)
        if i>brake:
            break
    labels=labels[1:]
    print(labels/np.sum(labels))
