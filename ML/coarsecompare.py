#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:08:10 2022

@author: arndt
Plots difference between different coarse graining resolution
"""

import time
from distributed import Client, progress, as_completed
import pandas as pd


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 600
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import sys
from distributed.client import futures_of
from datetime import datetime,timedelta
import os
import traceback
import dask.array as da
import glob
import dask.dataframe as dd
from scipy.stats import spearmanr,pearsonr


def get_max_fraction_simple(df):
    Sum = df.groupby(["lat", "lon"]).sum()
    Sum=Sum.loc[:,ctnames]
    Sum.columns=range(len(Sum.columns))
    Sum = Sum.idxmax(axis=1).to_frame("mc")
    
    return Sum


if __name__=="__main__":
    

    cMap= ListedColormap(['gold', 'green', 'blue','red', "cyan",
                          "lime", "black", "magenta"])#aquamarine
    print(sys.argv)
    if len(sys.argv)>=3:
        client = Client(sys.argv[2])
    else:
        SCHEDULER_FILE = glob.glob("/scratch/b/b309177/scheduler*.json")[0]
        
        if SCHEDULER_FILE and os.path.isfile(SCHEDULER_FILE):
            client = Client(scheduler_file=SCHEDULER_FILE)
        
        
            
    print(client.dashboard_link)
    plt.close("all")
    #dss = "/dss/dssfs02/pn56su/pn56su-dss-0004/"
    #work = dss+"work/"
    work = "/work/bd1179/b309177/"
    ctnames = [ "Ci", "As", "Ac", "St", "Sc", "Cu", "Ns", "Dc"]
    axlabel_dict={"clear": "Clear sky fraction","Ci":"Cirrus/Cirrostratus fraction",
                  "As":"Altostratus fraction", "Ac":"Altocumulus fraction",
                  "St":"Stratus fraction", "Sc": "Stratocumulus fraction",
                  "Cu": "Cumulus fraction", "Ns": "Nimbostratus fraction",
                  "Dc": "Deep convection fraction","clear_p": "Predicted clear sky fraction",
                  "Ci_p":"Predicted Cirrus/Cirrostratus fraction",
                  "As_p":"Predicted Altostratus fraction", 
                  "Ac_p":"Predicted Altocumulus fraction",
                  "St_p":"Predicted Stratus fraction", "Sc_p": "Predicted Stratocumulus fraction",
                  "Cu_p": "Predicted Cumulus fraction", "Ns_p": "Predicted Nimbostratus fraction",
                  "Dc_p": "Predicted Deep convection fraction",
                  "cwp":"total water path", "twp":"total water path",
                  "lwp": "liquid water path", "iwp":"ice water path",
                  "cod": "cloud optical depth", "tsurf": "surface temperature",
                  "stemp_cloudy": "surface temperature", "cee": "emissivity",
                  "ptop": "cloud top pressure", "htop": "cloud top height",
                  "ttop": "cloud top temperature", "cerl": "liquid droplet radius",
                  "ceri": "ice particle radius","ctp":"cloud top pressure"}
    
    #client.wait_for_workers(1)
    name=sys.argv[1] # this wants up to num files
    root=len(name)
    files = glob.glob(os.path.join(work,"frames/parquets",name+"*.parquet"))
    fig = plt.figure(figsize=(10,8))
    axindexes = {"10":0,"50":1,"100":2}
    print(files) 
    for num,file in enumerate(files):
            
        _,name = os.path.split(file)
        print(name)
        start=name[root:]
        end=start.find("_")
        resolu = start[:end]
        axindex = axindexes[resolu]    
        try:
            sptemp = ["time","lat", "lon"] 
            clear=dd.read_parquet( file,
                                  columns =sptemp+["clear"],
                               chunksize=1_000_000    )
        
            df=dd.read_parquet( file,
                               columns =sptemp+ctnames,
                               chunksize=1_000_000)
                
            
            
            times=0
        except Exception:
            traceback.print_exc()
            sptemp = [ "lat", "lon"] 
            clear=dd.read_parquet( os.path.join(work, "frames/parquets", 
                                                sys.argv[1]), 
                                  columns =sptemp+["clear"],
                               chunksize=1_000_000    )
            df=dd.read_parquet( os.path.join(work, "frames/parquets", 
                                             sys.argv[1]), columns =sptemp+ctnames,
                           chunksize=1_000_000    )
    
    
        print("df loaded", df.npartitions)
        #samplefrac=1
        rounddict={key:{"lat": 0, "lon": 0,"time":0}[key] for key in sptemp}
        if len(sys.argv)>3:
            seas = sys.argv[3]
            name=name.replace("ESACCI","ESACCI_{}".format(seas))
            #samplefrac*=2
            df.time=dd.to_datetime(df.time,unit="D")
            months={"DJF":(12,1,2),"MAM":(3,4,5),"JJA":(6,7,8),"SON":(9,10,11)}
            one,two,thr = months[seas]
            inseason = ((df.time.dt.month==one)|
                      (df.time.dt.month==two)|
                      (df.time.dt.month==thr))
            df=df.loc[inseason]
            df.time=df.time.astype(int)/1e9/3600
            print(df.time.mean().compute(),df.time.max().compute())
        
       
        twodegreegrid=0
        if not twodegreegrid:
            df=df.round(rounddict)
        else:
            df=(df/2).round(rounddict)*2
            name="2"+name
        s=df.loc[:,ctnames].sum(axis="columns")
        df[[x+"n" for x in ctnames]] = df.loc[:,ctnames].truediv(s,axis="index")
        df=df.drop(ctnames,axis=1)
        df.columns=sptemp+ctnames
        
        
        df_label=client.persist(df)
        progress(df_label)
        del df
        most_common=client.submit(get_max_fraction_simple,df_label).result()
        most_common=most_common.persist()
        progress(most_common)
        gpby_lat=df_label.groupby("lat")
        zonal=gpby_lat.agg(["mean", "std"]).iloc[:,-16:]
        
        
        most_common.columns=["ctype"]
        print("beforeindex") 
        index=client.compute(most_common.index,)
        print("afterindex")
        #client.wait_for_workers(12)
        index=index.result()
        progress(index)
        
        coords=np.stack(list(index)).astype("float16")
        del index
        print(coords.shape)
        colors = client.compute(most_common.ctype.to_dask_array(),).result()
            
        
        ax=fig.add_subplot(3,2,(axindex*2+1),projection=ccrs.PlateCarree())
        ax.coastlines()
        print(resolu)
        resolu_indeg = int(resolu)*0.05
        
        ax.set_title("Predicted on cells of size: {}°".format(resolu_indeg))
        if resolu=="100":
            ax.set_xticks([-160,-120,-80,-40,0,40,80,120,160])
            ax.set_xticklabels([-160,-120,-80,-40,0,40,80,120,160], fontsize=13)
        ax.set_yticklabels([-80,-40,0,40,80], fontsize=13)
        ax.set_yticks([-80,-40,0,40,80])
        lon = coords[:,1]
        lat = coords[:,0]
        un_lat = np.unique(lat)
        un_lon = np.unique(lon)
        print(len(un_lat), len(un_lon))
        
        print(coords.shape, np.max(lat), np.max(lon), np.min(lat), np.min(lon))
        #gg=np.empty((len(step_lat),len(step_lon)))*np.nan
        gg=np.empty((len(un_lat),len(un_lon)))
        for x,y,z in zip(lat,lon,colors):
            i = np.argwhere(un_lat==x)
            j= np.argwhere(un_lon==y)
            gg[i,j] = z
            
        lonlon,latlat = np.meshgrid(np.unique(lon),np.unique(lat))
        print("colors:",np.unique(colors))
        print(latlat.shape,lonlon.shape,gg.shape, np.max(latlat), np.max(lonlon), 
              np.sum(np.isnan(gg)))
        scatter=ax.pcolormesh(lonlon, latlat,gg,cmap=cMap,shading="nearest",
                           norm=Normalize(0,8), transform=ccrs.PlateCarree())
        
        
        
        
        ax=fig.add_subplot(3,2,axindex*2+2)
        zonal=zonal.compute()
        zonal.sort_index(inplace=True)
        
        i=2
        if i*2<zonal.shape[1]:
            zonal.iloc[:,i*2].plot(ax=ax, label="mean")
            if resolu=="10":
                ax.set_title(zonal.columns[i*2][0])
            if resolu!="100":
                ax.set_xticklabels(ax.get_xticklabels(),visible=False)
                ax.set_xlabel("")
            bottom = zonal.iloc[:,i*2]-zonal.iloc[:,i*2+1]
            bottom = np.where(bottom<0, 0,bottom)
            top = zonal.iloc[:,i*2]+zonal.iloc[:,i*2+1]
            ax.fill_between(zonal.index.values, (bottom), (top), color='b', alpha=.1, label=u"$\sigma$")
            ax.grid()
        if num==2:
            ax.legend(fontsize=15)
        
        if num==len(files)-1:
            fig.tight_layout()
            cb_ax = fig.add_axes([0.04, 0.05, 0.01, .85])  
            cbar = fig.colorbar(scatter, orientation ="vertical",cax=cb_ax)
            cbar.ax.tick_params(direction="in", labelleft=True, labelright=False)
            cbar.ax.get_yaxis().set_ticks(np.arange(8)*1.0+0.5)
            cbar.ax.get_yaxis().set_ticklabels( [ 'Ci', 'As', 'Ac', 'St', 'Sc', 'Cu', 'Ns', 'Dc'],
                                               fontsize=18)
            
            
            fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_")+"coarsecomp.svg"))
