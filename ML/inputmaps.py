#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:01:58 2021

@author: arndt
Plots global maps of the input variables as well as timeseries
"""
from distributed import Client, progress
import numpy as np
import matplotlib
matplotlib.use("PS")
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 400
from matplotlib.colors import ListedColormap, Normalize
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import sys
import os
import traceback
import dask.dataframe as dd
import glob
import time
from datetime import datetime,timedelta
import matplotlib.dates as mdates



def customize(ax):
    ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())
    ax.set_xticks(range(-160, 200, 40), crs=ccrs.PlateCarree())
    ax.set_yticks(range(-80, 120, 40), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.coastlines()


if __name__ == "__main__":
    print(matplotlib.get_backend())
    print(sys.argv)
    if len(sys.argv)>=3:
        client = Client(sys.argv[2])
        client.restart()
    else:
            while True:
                try:
                    SCHEDULER_FILE = glob.glob("/scratch/b/b309177/scheduler*.json")[0]
                    
                    if SCHEDULER_FILE and os.path.isfile(SCHEDULER_FILE):
                        client = Client(scheduler_file=SCHEDULER_FILE)
                    break
                except IndexError:
                    time.sleep(10)
    print(client.dashboard_link)
    plt.close("all")
    #dss = "/dss/dssfs02/pn56su/pn56su-dss-0004/"
    #work = dss+"work/"
    work = "/work/bd1179/b309177/"
    ctnames = [ "Ci", "As", "Ac", "St", "Sc", "Cu", "Ns", "Dc"]
    propnames = ['twp', 'lwp', 'iwp', 'cer_liq', 'cer_ice', 'cot', 'ctp', 'stemp_cloudy']
    
    
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
    
    
    ranges = [(0,800),(0,750),(0,500),(0,14),(0,28),(0,50),(0,950),(175,350)]
    ranges_dict={x:y for x,y in zip(propnames,ranges)}
    
    client.wait_for_workers(1)
    try:
        sptemp = ["time","lat", "lon"] 
        df=dd.read_parquet( os.path.join(work, "frames/parquets", sys.argv[1]), columns =sptemp+propnames,
                           chunksize=1_000_000)
        times=0
    except Exception:
        traceback.print_exc()
        sptemp = [ "lat", "lon"] 
        df=dd.read_parquet( os.path.join(work, "frames/parquets", sys.argv[1]), columns =sptemp+propnames,
                       chunksize=1_000_000    )
    
    name=sys.argv[1]
    
    print("df loaded", df.npartitions)
    
    dtypes = {x:"float16" for x in sptemp}
    for cname in ctnames :
        dtypes[cname]="float32"
        
    df=df.sample(frac=0.001, replace=False, random_state=22345)
    rounddict={key:{"lat": 0, "lon": 0, "time":0,
                    'twp':0, 'lwp':0, 'iwp':0, 'cer_liq':1, 'cer_ice':1, 'cot':1, 
                    'ctp':0, 'stemp_cloudy':0}[key] for key in sptemp+propnames}
    units ={'twp':"g/m²", 'lwp':"g/m²", 'iwp':"g/m²", 'cer_liq':"µm", 'cer_ice':"µm", 'cot':"-", 
                    'ctp':"hPa", 'stemp_cloudy':"K"}
    longnames ={'twp':"Total water path", 'lwp':"Liquid water path", 'iwp':"Ice water path",
            'cer_liq':"Effective radius of liquid particles", 'cer_ice':"Effective radius of ice particles",
            'cot':"Cloud Optical Thickness", 
                    'ctp':"Cloud top Pressure", 'stemp_cloudy':"Surface Temperature"}
    
    
    #exclude july 2010
    start = datetime.fromisoformat("1970-01-01")
    start_july = datetime.fromisoformat("2010-07-01")
    end_july = datetime.fromisoformat("2010-07-31")
    ex_july = (df.time>(end_july-start).days)|( df.time<(start_july-start).days)
    print(ex_july)
    df=df[ex_july]
    
    
    print(df.time.head())
    df_label=df.round(rounddict)
    df_label=client.persist(df_label)
    progress(df_label)
    
    regional = df_label.groupby(["lat","lon"]).mean()
    
    for j in propnames[:]:
        i=regional.loc[:,j]
        
        prop=i.compute()
        prop=prop.reset_index(level=["lat","lon"])
        
        un_lat = np.unique(prop.lat)
        un_lon = np.unique(prop.lon)
        colors=prop.loc[:,j].values
        gg=np.ones((len(un_lat),len(un_lon)))*-1
        for x,y,z in zip(prop.lat,prop.lon,colors):
            I = np.argwhere(un_lat==x)
            J= np.argwhere(un_lon==y)
            gg[I,J] = z
        gg=np.where(gg<0,np.nan,gg)
        lonlon,latlat = np.meshgrid(un_lon,un_lat)
        
        #client.wait_for_workers(16)
        fig=plt.figure(j)
        ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    
        customize(ax)
        vmi,vma = ranges_dict[j]
        scatter=ax.pcolormesh(lonlon, latlat,gg,cmap="Greens",shading="nearest",
                                transform=ccrs.PlateCarree())#,vmin=vmi,vmax=vma)

        ax.set_title("{} [{}]".format(longnames[j],units[j]), fontsize=15)
    
        cbar = fig.colorbar(scatter, orientation ="vertical", fraction=0.12,pad=0.01,
                            shrink=0.55)
        cbar.ax.tick_params(axis="y", labelsize=15)
        fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_")+j+".eps"))
        print(j, "done")
        del colors
    
    iwp=regional.loc[:,"iwp"].compute()
    cwp=regional.loc[:,"twp"].compute()
    
    
    iwp=iwp.reset_index(level=["lat","lon"])
    cwp=cwp.reset_index(level=["lat","lon"])
    prop=iwp.copy()
    prop.iwp = iwp.iwp.values/cwp.twp.values
    print(prop.head())
    print(iwp.head())
    print(cwp.head())
    
    un_lat = np.unique(prop.lat)
    un_lon = np.unique(prop.lon)
    print(prop.columns)
    colors=prop.loc[:,"iwp"].values
    gg=np.ones((len(un_lat),len(un_lon)))*-1
    for x,y,z in zip(prop.lat,prop.lon,colors):
        I = np.argwhere(un_lat==x)
        J= np.argwhere(un_lon==y)
        gg[I,J] = z
    gg=np.where(gg<0,np.nan,gg)
    lonlon,latlat = np.meshgrid(un_lon,un_lat)
    
    
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    
    customize(ax)
    vmi,vma = (0,1)
    
    assert np.all(gg.shape==lonlon.shape),lonlon.shape
    scatter=ax.pcolormesh(lonlon, latlat, gg,cmap="Greens",shading="nearest",
                            transform=ccrs.PlateCarree(),vmin=vmi,vmax=vma)
    ax.set_title("Relative ice water path", fontsize=16)
    
    cbar = fig.colorbar(scatter, orientation ="vertical", fraction=0.12,pad=0.01,
                        shrink=0.55)
    cbar.ax.tick_params(axis="y", labelsize=15)
    #fig.tight_layout()
    plt.draw()
    fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_")+"relice.eps"))
    print("relice", "done")
    
    
    if "times" in globals():
        df_now = df_label[df_label.lat<0]     
        fig, ax =plt.subplots(4,2, sharex=True, figsize=(10,10))
        ax=ax.flatten()
        for i,cname in enumerate(propnames):
            print(cname)
            sub = df_now.loc[:,["lat","lon","time",cname]]
            gpby=sub.groupby("time")
        
            temporal=gpby.agg(["mean", "std"]).iloc[:,4:]
            temporal=temporal.compute()
            temporal.sort_index(inplace=True)
    
            temporal.iloc[:,0].plot(ax=ax[i], label="mean")
            ax[i].set_title(temporal.columns[0][0])
            bottom = temporal.iloc[:,0]-temporal.iloc[:,1]
            bottom = np.where(bottom<0, 0,bottom)
            top = temporal.iloc[:,0]+temporal.iloc[:,1]
            days = temporal.index.values
            start=datetime.fromisoformat("1970-01-01")
            dates = [start+timedelta(days=x) for x in days]
            ax[i].fill_between(dates, (bottom), (top), color='b', alpha=.1, label=u"$\sigma$")
            ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax[i].grid()
            ax[i].legend(fontsize=15)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_")+"tempprops.eps"))
            
