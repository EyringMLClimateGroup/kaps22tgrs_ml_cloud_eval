#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 15:03:31 2021

@author: arndt
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 400
import os
import glob
import cartopy.crs as ccrs
import netCDF4 as nc4
from matplotlib.colors import Normalize,ListedColormap
from matplotlib.cm import ScalarMappable
import traceback 

def extract(d,arr):
    
    arr = np.pad(arr, ((0,0),(1,0), (1,0)),constant_values=0)
    
    return arr[:,d::d,d::d]+arr[:,:-d:d,:-d:d]-arr[:,:-d:d,d::d]-arr[:,d::d,:-d:d]

if __name__=="__main__":
    plt.close("all")
    
    
    cMap= ListedColormap(['gold', 'green', 'blue','red', "cyan",
                          "lime", "black", "magenta"])#aquamarine
    conf = np.zeros(2)
    #work =os.path.join("/mnt/lustre02/work/bd1083/b309177")
    #work = os.path.join("/mnt/work/")
    work = os.path.join("/work/bd1179/b309177")
    files =      glob.glob(os.path.join(work, "lrz_data/oob/npz/train/*.npz"))
    #files.extend(glob.glob(os.path.join(work, "lrz_data/oob/npz/test/*.npz")))
    #files.extend(glob.glob(os.path.join(work, "lrz_data/npz/*.npz")))
    np.random.shuffle(files)
    label_tiles = os.path.join(work,"lrz_data/oob/numpy/label/tiles")
    label_locs = os.path.join(work,"lrz_data/oob/numpy/label/metadata")
    predicted = os.path.join(work,"lrz_data/oob/results/2021-10-15phy",
                             "best/out/predicted-label-random/")
    all_f=glob.glob(os.path.join(work,"lrz_data/oob/*.nc"))
    all_f.extend(glob.glob(os.path.join(work,"lrz_data/*.nc")))
    all_f=[os.path.basename(x) for x in all_f]
    assert len(all_f)
    days,counts=np.unique([int(x[6:9]) for x in all_f],return_counts=True)
    argmaxes = np.argwhere(counts==np.max(counts))
    
    days=days[argmaxes]
    
    bestday=0
    for day in days:
        day=day[0]
        if len(glob.glob(os.path.join(work,
                                          "lrz_data/oob/npz/train",
                                          "*.{:03d}.*".format(day))))>bestday:
            bestday = len(glob.glob(os.path.join(work,"lrz_data/oob/npz/train",
                "*.{:03d}.*".format(day))))
            loadday=day
    print(loadday)
    
    ctnames = [ "Ci", "As", "Ac", "St", "Sc", "Cu", "Ns", "Dc"]
    
    fig3 = plt.figure(figsize=(7, 5))
    ax3 = fig3.add_subplot(1, 1, 1, projection=ccrs.Geostationary(-25))
    ax3.set_global()
    fig2 = plt.figure(figsize=(7, 5))
    ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Geostationary(-25))
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(-25))
    ax.set_global()
    ax2.set_global()
    
    fig4 = [plt.figure(figsize=(7, 5)) for _  in range(9)]
    ax4 = [fig4[i].add_subplot(1, 1, 1, projection=ccrs.Geostationary(-25)) for i in range(9)]
    [ax4[i].set_global() for i in range(9)]
    maxima=[0 for _ in range(9)]
    for load in files:
        if (".{:03d}.".format(loadday) not in load):
            continue
        name = os.path.basename(load).replace("npz","npy")
    
        try:
            try:
                labs = np.load(os.path.join(label_tiles,name))[:,-8:].reshape(-1,8,9)
                locs = np.load(os.path.join(label_locs,name))
                ds = nc4.Dataset(os.path.join(work,"lrz_data", name.replace("npy","nc")))
            except FileNotFoundError:
                traceback.print_exc()
                labs = np.load(os.path.join(os.path.join(work,"lrz_data/numpy/label/tiles"),name))[:,-8:].reshape(-1,8,9)
                locs = np.load(os.path.join(os.path.join(work,"lrz_data/numpy/label/metadata"),name))
                ds = nc4.Dataset(os.path.join(work,"lrz_data", name.replace("npy","nc")))
            compat = np.load(os.path.join(work,
                                          "lrz_data/oob/npz/train",
                                          name.replace("npy","npz")))
            lat = ds.variables["latitude"][:]
            lon = ds.variables["longitude"][:]
            com_labels = compat["labels"].transpose()
            compat=compat["properties"]
            #compat = extract(10,compat)/10**2
            #com_labels = extract(10,com_labels)
        except Exception as err:
            print(err)
            continue
        
        locs=locs[:,:,0].astype(int)+1
        locs_0=lon[0,locs[:,0],locs[:,1]]
        locs_1=lat[0,locs[:,0],locs[:,1]]
        labs = np.argmax(np.sum(labs,2),1)
        npz = np.load(load)
        mask = npz["cloud_mask"][0]
        
        for take in range(9):
        
            somerad = npz["properties"][take]
            somerad = np.where(somerad<=0,np.nan,somerad)
            M=ax4[take].pcolormesh(lon[0],lat[0],somerad,cmap="Greys",
                             transform=ccrs.PlateCarree(),shading="auto",rasterized=True)    
            maxima[take]=max(np.nanmax(somerad),maxima[take])
            print("max",np.nanmax(somerad))
            
        
        mesh=ax.pcolormesh(lon[0],lat[0],mask.astype(bool), cmap="Blues",
                           shading="auto",transform=ccrs.PlateCarree(),
                           norm=Normalize(0,5),rasterized=True)
        #ax.set_tick_params(tick1On=False,tick2On=False,label1On=False,label2On=False)
        scatt=ax.scatter(locs_0,locs_1,c=labs,s=1.8, cmap=cMap,
                         transform=ccrs.PlateCarree(),norm=Normalize(0,7),rasterized=True)
        CS=ax3.scatter(locs_0,locs_1,c=labs ,s=1.8,cmap=cMap,
                       transform=ccrs.PlateCarree(),norm=Normalize(0,7),rasterized=True)
        
        
        pred=np.load(os.path.join(predicted,name))
        print(somerad.shape, pred.transpose().shape,name)
        mesh2=ax2.pcolormesh(lon[0],lat[0],pred, cmap=cMap,
                             shading="auto",transform=ccrs.PlateCarree(),
                             norm=Normalize(0,7),rasterized=True)
        
        """    
        fig,ax=plt.subplots()
        ax.xaxis.set_major_locator(MultipleLocator(3))
        ax.yaxis.set_major_locator(MultipleLocator(3))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        
        matrixwise =np.zeros((com_labels.shape[1]*3,com_labels.shape[2]*3))
        for (i,j) in product(range(com_labels.shape[1]), range(com_labels.shape[2])):
            for k in range(9):
                matrixwise[i*3+k%3,j*3+k//3]=com_labels[k,i,j]
                
        diversity = np.sum(com_labels!=0,0)
        print(com_labels.shape, diversity.shape)
        aw= np.argwhere(diversity==np.max(diversity))
        print(aw.shape)
        if len(aw.shape)>2:
            mx,my = aw[0]
        else:
            mx,my = aw
        fig2,ax2=plt.subplots()
        ax2.xaxis.set_major_locator(MultipleLocator(3))
        ax2.yaxis.set_major_locator(MultipleLocator(3))
        cmask=ax2.pcolormesh(compat[-1,mx-3:mx+4,my-2:my+3], cmap="Blues")
        #fig2.colorbar(cmask)
        mx*=3
        my*=3
        mx=min(matrixwise.shape[0]-12,max(mx,9))
        my=min(matrixwise.shape[1]-9,max(my,6))
        mat=ax.pcolormesh(matrixwise[mx -9:mx +12,my -6:my+9],cmap="inferno")
        ax.grid(1,"major", lw=3,c="white")
        ax.grid(1,"minor", lw=0.5, ls="--")
        ax.coastlines()
        fig.colorbar(mat)
        """
    
    ax.coastlines()
    ax2.coastlines()
    ax3.coastlines()
    
    cbar=fig.colorbar(scatt,ticks=np.linspace(0.5,6.5,8))
    cbar3=fig3.colorbar(CS,ticks=np.linspace(0.5,6.5,8))
    cbar2=fig2.colorbar(mesh2,ticks=np.linspace(0.5,6.5,8))
    
    cbar.ax.set_yticklabels(ctnames)
    cbar3.ax.set_yticklabels(ctnames)
    cbar2.ax.set_yticklabels(ctnames)
    fig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    
    fig.savefig(os.path.join(work,"CUMULO.ps"))
    fig2.savefig(os.path.join(work,"pred.ps"))
    fig3.savefig(os.path.join(work,"onlycs.ps"))
    
    cbar4=[fig4[i].colorbar(ScalarMappable(norm=Normalize(0,maxima[i]),cmap="Greys")) for i in range(9)]
    ax4[4].set_title(u"Cloud top pressure [hPa]")
    [fig4[i].tight_layout() for i in range(9)]
    [ax4[i].coastlines() for i in range(9)]
    
    
    [fig4[take].savefig(os.path.join(work,"{}_maybe.eps".format(take))) for take in range(9)]
