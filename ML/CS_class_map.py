#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 14:10:29 2021

@author: arndt, partly from CUMULO
plots the base CloudSat labels for one year. 
Also saves normalized class distributions.
reguires raw 2B-CLDCLASS-LIDAR data (not CUMULO)
"""
import cartopy.crs as ccrs
import glob
import os
import sys
import numpy as np
import pickle5 as pickle
from pyhdf.SD import SD,SDC
from pyhdf.HDF import HDF
from pyhdf.VS import VS
import multiprocessing as mlp
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] =400
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
from datetime import datetime,timedelta
import warnings


def random_argmax(occ,classes):
    """
    unnecessary complex function to get one of the argmaxes instead of always the first one
    """
    i = int(np.random.randint(classes))
    if len(occ.shape)>1:
        return (np.argmax(np.roll(occ,i, 1),1)-i)%classes
    else:
        return (np.argmax(np.roll(occ,i))-i)%classes


def get_coordinates(cloudsat_path, verbose=0):
    f = HDF(cloudsat_path, SDC.READ)
    vs = f.vstart()

    vdata_lat = vs.attach('Latitude')
    vdata_long = vs.attach('Longitude')
    time_start =vs.attach("TAI_start")
    time_series = vs.attach("Profile_time")
    
    latitudes = vdata_lat[:]
    longitudes = vdata_long[:]
    #print("time",time_series[:][0],time_start[:][0][0],len(time_series[:]))
    time = [x[0]+time_start[:][0][0] for x in time_series[:]]
    
    assert len(latitudes) == len(longitudes), "cloudsat hdf corrupted"

    if verbose:
        print("hdf information", vs.vdatainfo())
        print('Nb pixels: ', len(latitudes))
        print('Lat min, Lat max: ', min(latitudes), max(latitudes))
        print('Long min, Long max: ', min(longitudes), max(longitudes))


    # close everything
    vdata_lat.detach()
    vdata_long.detach()
    time_start.detach()
    time_series.detach()
    vs.end()
    f.close()

    return np.array(latitudes).flatten(), np.array(longitudes).flatten(), np.array(time).flatten()


def get_layer_information(cloudsat_path, verbose=0):
    """ Returns
    CloudLayerType: -9: error, 0: non determined, 1-8 cloud types
    
    """


    sd = SD(cloudsat_path, SDC.READ)

    if verbose:
        # List available SDS datasets.
        print("hdf datasets:", sd.datasets())

    # get cloud types at each height
    layer_types=sd.select('CloudLayerType').get()

    occurrences = np.zeros((layer_types.shape[0], 8))

    for occ, labels in zip(occurrences, layer_types):

        for l in labels:

            # keep only cloud types (no 0 or -9)
            if l > 0:
                occ[l-1] += 1
    most_frequent = random_argmax(occurrences,8)
    return most_frequent

def get_CS_frame(name):
    lat,lon ,time= get_coordinates(name)
    labels = get_layer_information(name)
    
    pddict = {"lat":lat,"lon":lon,"time":time,"labels":labels}
    df =pd.DataFrame(pddict,dtype="float32")
    return df


def get_gg(inp):
    cname, location_fraction=inp
    print(location_fraction.head())
    labels=location_fraction.loc[:,cname]
    
    
    print(labels.head())
    
    
    
    lat = np.array([x for x,y,z in labels.index])
    lon = np.array([y for x,y,z in labels.index])
    
    un_lat = np.unique(lat)
    un_lon = np.unique(lon)
    #gg=np.empty((len(step_lat),len(step_lon)))*np.nan
    gg=np.zeros((len(un_lat),len(un_lon)))
    tot =np.zeros((len(un_lat),len(un_lon)))
    
    for x,y,z in zip(lat,lon,labels):
        i = np.argwhere(un_lat==x)
        j= np.argwhere(un_lon==y)
        
        gg[i,j] = labels.loc[(x, y)].mean()
        
    #lonlon,latlat = np.meshgrid(step_lon,step_lat)
    lonlon,latlat = np.meshgrid(un_lon,un_lat)
    
    to_save=np.stack((lonlon,latlat,gg))
    if len(files)== len(glob.glob(os.path.join(lidar,"*hdf"))):
        np.save(os.path.join(lidar,"../../frames","CS_{}.npy".format(cname)), to_save)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        #gg=np.where(sum_per_cell!=0,gg/sum_per_cell,0)
    return (lonlon,latlat,gg)

if __name__=="__main__":
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    cMap= ListedColormap(['gold', 'green', 'blue','red', "cyan",
                          "lime", "black", "magenta"])#aquamarine
    work=os.environ["WORK"]

    lidar = os.path.join(work,"CloudSat/lidar")
    
    files = glob.glob(os.path.join(lidar,"*hdf"))[:]
    pool=mlp.Pool(min(40,len(files)))
    try:
        df=pd.read_pickle("test_CS.pkl")
    except FileNotFoundError:
        
        assert len(files)>0,os.listdir(lidar)
        print(len(files))
        
        dfs=pool.map(get_CS_frame,files)
        df=pd.concat(dfs,ignore_index=True)
    print(len(df))
    
    fig,ax=plt.subplots()
    df.labels.hist(bins=np.arange(9),ax=ax,density=True)
    fig.savefig(os.path.join(lidar,"../../stats","histCS.eps"))
    
    
    
    fig=plt.figure(figsize=(12,7),dpi=800)
    ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    
    indiv=ax.scatter(df.lon, df.lat,c=df.labels,s=0.05 ,cmap=cMap,
                        transform=ccrs.PlateCarree(),norm=Normalize(0,8))
    cbar=fig.colorbar(indiv, orientation="vertical",fraction=0.08,pad=0.02,
                        shrink=0.8)
    cbar.ax.tick_params(labelsize=20)
    
    ax.set_xticks([-160,-120,-80,-40,0,40,80,120,160])
    ax.set_yticks([-80,-40,0,40,80])
    ax.set_xticklabels([-160,-120,-80,-40,0,40,80,120,160], fontsize=16)
    ax.set_yticklabels([-80,-40,0,40,80], fontsize=16)
    ax.set_title("CS ctypes scatter",fontsize=15)
    fig.tight_layout()
    cbar.ax.get_yaxis().set_ticks(np.arange(8)*1.0+0.5)
    cbar.ax.get_yaxis().set_ticklabels( [ 'Ci', 'As', 'Ac', 'St', 'Sc', 'Cu', 'Ns', 'Dc'],
                                       fontsize=27)
    fig.savefig(os.path.join(lidar,"../../stats","CS_Scatter.eps"))
    
    df*=5
    df=df.round({"lat":-1,"lon":-1})/5
    df.to_pickle("test_CS.pkl")
    
    start=datetime.fromisoformat("1993-01-01")
    dates = [start+timedelta(seconds=x) for x in df.time]
    df.time=dates
    """
    gpby=df.groupby(["lat","lon"])
    
    labels = [np.argmax(np.bincount(df.labels.iloc[x].values.astype(int),minlength=8)) 
              for x in gpby.groups.values()]
    lat = np.array([x for x,y in gpby.groups.keys()])
    lon = np.array([y for x,y in gpby.groups.keys()])
    
    
    un_lat = np.unique(lat)
    un_lon = np.unique(lon)
    print(len(un_lat), len(un_lon))
    #gg=np.empty((len(step_lat),len(step_lon)))*np.nan
    gg=np.ones((len(un_lat),len(un_lon)))*-1
    for x,y,z in zip(lat,lon,labels):
        i = np.argwhere(un_lat==x)
        j= np.argwhere(un_lon==y)
        gg[i,j] = z
    #lonlon,latlat = np.meshgrid(step_lon,step_lat)
    gg=np.where(gg<0,np.nan,gg)
    lonlon,latlat = np.meshgrid(un_lon,un_lat)
    sum_per_cell=np.copy(gg)
    print(latlat.shape,lonlon.shape,gg.shape, np.max(latlat), np.max(lonlon), 
          np.sum(np.isnan(gg)))
    fig=plt.figure(figsize=(12,7),dpi=800)
    ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    
    scatter=ax.pcolormesh(lonlon, latlat,gg,cmap=cMap,shading="nearest",
                        transform=ccrs.PlateCarree(),norm=Normalize(0,8))
    cbar=fig.colorbar(scatter, orientation="vertical",fraction=0.08,pad=0.02,
                        shrink=0.8)
    cbar.ax.tick_params(labelsize=20)
    
    ax.set_xticks([-160,-120,-80,-40,0,40,80,120,160])
    ax.set_yticks([-80,-40,0,40,80])
    ax.set_xticklabels([-160,-120,-80,-40,0,40,80,120,160], fontsize=16)
    ax.set_yticklabels([-80,-40,0,40,80], fontsize=16)
    ax.set_title("Most common CS ctype",fontsize=15)
    fig.tight_layout()
    cbar.ax.get_yaxis().set_ticks(np.arange(8)*1.0+0.5)
    cbar.ax.get_yaxis().set_ticklabels( [ 'Ci', 'As', 'Ac', 'St', 'Sc', 'Cu', 'Ns', 'Dc'],
                                       fontsize=27)
    fig.savefig(os.path.join(lidar,"../../stats","CS_most_common.eps"))
    
   
    """
    allfig=plt.figure(figsize=(10,9),dpi=600)
    allaxes = np.array([allfig.add_subplot(4,2,pos+1,projection=ccrs.PlateCarree()) for pos in range(8)])
    tempfig,tempaxes=plt.subplots(4,2,figsize=(10,9),dpi=600,sharex =True)
    tempaxes=tempaxes.flatten()
    
    ctnames = [ "Ci", "As", "Ac", "St", "Sc", "Cu", "Ns", "Dc"]
    amount = df.groupby(["lat","lon",pd.Grouper(key="time",freq="1D")]).size()
    
    #df=df.round({"lat":0,"lon":0})
    
    location_fraction=pd.DataFrame(columns=ctnames,index=amount.index)
    
    
    per_time=df.groupby(pd.Grouper(key="time",freq="1W")).size()
    print(per_time.head())
    for pos,cname in enumerate(ctnames):
        sub = df[df.labels==pos]
        
        gpby=sub.groupby(["lat","lon",pd.Grouper(key="time",freq="1D")])
        
        labels = gpby.size()
        
        assert len(sub)==np.sum(labels)
        labels=labels/amount.loc[labels.index]
        location_fraction.loc[labels.index,cname]=labels
        
        sub = df[df.labels==pos]
        temporal=sub.groupby(pd.Grouper(key="time",freq="1W"))
        std=temporal.labels.std()
        
        days=temporal.groups.keys()
        temporal=temporal.size()/per_time
        
        
        bottom = np.where(temporal.values-std.values<0,0,temporal.values-std.values)
        top=temporal.values+std.values
        top=top.squeeze()
        bottom=bottom.squeeze()
        temporal.plot(ax=tempaxes[pos])
        tempaxes[pos].fill_between(days, (bottom), (top), color='b', alpha=.1, label=u"$\sigma$")
    location_fraction.fillna(0,inplace=True)
    cloudmax = np.max(location_fraction.iloc[:,3:].max())
    print("cloudmax",cloudmax)
    print(location_fraction.head().sum(1))
    
    ggs=pool.map(get_gg,zip(ctnames,[location_fraction for _ in ctnames]))
    for pos,mesh in enumerate(ggs):
        lonlon,latlat,gg=mesh
        aax=allaxes[pos]
        scatter=aax.pcolormesh(lonlon,latlat,gg,cmap="gist_stern",shading="nearest",
                                transform=ccrs.PlateCarree(),vmin=0,vmax=cloudmax)
        aax.coastlines()
        if pos>5:
            aax.set_xticks([-140,-70,0,70,140],minor=False)
            aax.set_xticklabels([-140,-70,0,70,140], fontsize=14)
        if pos%2==0:
            aax.set_yticks([-80,-40,0,40,80],minor=False)
            aax.set_yticklabels([-80,-40,0,40,80], fontsize=14)
        aax.set_title("{}".format(ctnames[pos]),fontsize=15)
        aax.coastlines()
        
        allcb_ax = allfig.add_axes([0.9, 0.1, 0.03, .78],label=cname)
        allcbar = allfig.colorbar(scatter, cax=allcb_ax)
        #allcbar =allfig.colorbar(scatter, ax=allaxes.ravel().tolist(),
        #                         orientation="vertical",fraction=0.08,pad=0.2,
        #                        shrink=0.8)
        allcbar.ax.tick_params(labelsize=14)
        allfig.suptitle("Relative cloud class occurrence",fontsize=20)
    #allfig.tight_layout()
    allfig.savefig(os.path.join(lidar,"../../stats","CS_allctypes.eps"))
    tempfig.tight_layout()
    tempfig.savefig(os.path.join(lidar,"../../frames", "CS_temporal.eps"))
