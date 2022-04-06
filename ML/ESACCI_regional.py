#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 11:20:03 2021

@author: arndt
"""
import time
from distributed import Client, progress, as_completed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 400
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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def FI(arr,val):
    """
    Finds the first instance of arr being equal to val
    Parameters
    ----------
    arr : np array
        array 
    val : float
        class

    Returns
    -------
    int
        first time arr is val

    """
    return np.argmin(np.abs(arr-val))


def hist_heights(x,gp,bindict):
    """
    given a preset array of bins, find the corresponding histogram values

    Parameters
    ----------
    x : np array
        partition of dataframe in this case
    gp : str
        name of column to look at
    bindict : dict
         bins to use

    Returns
    -------
    a : np.ndarray
        histogram

    """
    a,_= np.histogram(x,bins=bindict[gp])
    return a


def customize(ax):
    """
    
    makes a good looking world map plot
    Parameters
    ----------
    ax : matplotlib axis
        axis to customize

    Returns
    -------
    None.

    """
    ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())
    ax.set_xticks(range(-160, 200, 40), crs=ccrs.PlateCarree())
    ax.set_yticks(range(-80, 120, 40), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=14)
    ax.coastlines()

def map_remote(df,gp,bindict):
    """
    in the dask cluster, compute histograms of the columns of a dask dataframe

    Parameters
    ----------
    df : dask.dataframae.DataFrame
        a dataframe
    gp : str
        name of column.
    bindict : dict
        dict of numpy arrays

    Returns
    -------
    out : np array
        the histograms
    gp : same as what goes in
    bindict : same as what goes in

    """
    df=df.loc[:,gp]
    if gp=="time":
        df-=df.min()
        with open("timemax.txt","w+") as file:
            ma=df.max().compute()
            mi=df.min().compute()
            print(ma, mi, file=file)
            
            bindict[gp]=np.arange(mi,ma)
    out = df.map_partitions(hist_heights,gp,bindict)
    if gp=="time":
        with open("timemax.txt","a+") as file:
            print(bindict["time"], file=file)
    return out,gp,bindict
    

def get_max_fraction_simple(df):
    """
    finds dominatn cloud type per lat/lon grid cell
    i.e. type that most often has the largest fraction

    Parameters
    ----------
    df : dask.dataframe.DataFrame
        cloud type fraction lat/lon/time

    Returns
    -------
    Sum : dataframe
        index of the cloud type where the fraction per lat/lon is largest

    """
    Sum = df.groupby(["lat", "lon"]).sum(split_out=int(df_label.npartitions/10))
    Sum=Sum.loc[:,ctnames]
    Sum.columns=range(len(Sum.columns))
    Sum = Sum.idxmax(axis=1).to_frame("mc")
    
    
    return Sum


if __name__=="__main__":
    

    cMap= ListedColormap(['gold', 'green', 'blue','red', "cyan",
                          "lime", "black", "magenta"])#aquamarine
    print(sys.argv)
    if len(sys.argv)>=3:
        #client IP adress
        client = Client(sys.argv[2])
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
    try:
        sptemp = ["time","lat", "lon"]
        #clear sky fraction
        clear=dd.read_parquet( os.path.join(work, "frames/parquets", 
                                            sys.argv[1]),
                              columns =sptemp+["clear"],
                           chunksize=1_000_000    )
        #cloud fractions
        df=dd.read_parquet( os.path.join(work, "frames/parquets", sys.argv[1]),
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

    name=sys.argv[1]
        
    print("df loaded", df.npartitions)
    #samplefrac=1
    rounddict={key:{"lat": 0, "lon": 0,"time":0}[key] for key in sptemp}
    if len(sys.argv)>3:
        #analyse seasonal subset of the data
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
    """
    #possibly also plot the clear sky fraction
    clear = clear.sample(frac=samplefrac, replace=False, random_state=22345)
    clear = clear.loc[:,["lat","lon","clear"]]
    clear = clear.groupby(["lat","lon"]).mean().compute()
    print("first compute done")
    clear=clear.reset_index(level=["lat","lon"])
    
    fig=plt.figure(figsize=(12,7))
    ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    cloud_fraction=ax.scatter(clear.lon,clear.lat,c=clear.clear,s=0.0005,norm=Normalize(0,1),
                              transform=ccrs.PlateCarree())
    cbar=fig.colorbar(cloud_fraction, orientation="horizontal",fraction=0.12,pad=0.12,
                            shrink=0.8)
    cbar.ax.tick_params(labelsize=20)
    ax.coastlines()
    ax.set_xticks([-160,-120,-80,-40,0,40,80,120,160])
    ax.set_yticks([-80,-40,0,40,80])
    ax.set_xticklabels([-160,-120,-80,-40,0,40,80,120,160], fontsize=16)
    ax.set_yticklabels([-80,-40,0,40,80], fontsize=16)
    ax.set_title("Clear sky fraction",fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(work,"stats",name.replace(".parquet","_csf.eps")))
    
    """
    del clear
    
    #df=df[df.lat>-66]
    #df=df[df.lat<-23]
    #name=name.replace("ESACCI","ESACCI_sstormtrack")
    
    # bin on a two degree grid instead of one, making comparison with cloudsat better
    twodegreegrid=1
    if not twodegreegrid:
        df=df.round(rounddict)
    else:
        df=(df/2).round(rounddict)*2
        name="2"+name
    #compute the sum to normalize
    s=df.loc[:,ctnames].sum(axis="columns")
    df[[x+"n" for x in ctnames]] = df.loc[:,ctnames].truediv(s,axis="index")
    df=df.drop(ctnames,axis=1)
    df.columns=sptemp+ctnames
    #exclude july 2010 because that has wrong data
    start = datetime.fromisoformat("1970-01-01")
    start_july = datetime.fromisoformat("2010-07-01")
    end_july = datetime.fromisoformat("2010-07-31")
    ex_july = (df.time>(end_july-start).days)|( df.time<(start_july-start).days)
    df=df[ex_july]
    
    df_label=client.persist(df)
    progress(df_label)
    del df # I dont think that does anything
    
    
    allfig=plt.figure(figsize=(10,9),dpi=600)
    allaxes = np.array([allfig.add_subplot(4,2,pos+1,
                                           projection=ccrs.PlateCarree()) 
                        for pos in range(8)])
    cloudmax=0
    pmeshdict={}
    
    if "2ESA" in name:
        # if we are on two degree grid, load CloudSat data and compare
        for pos,cname in enumerate(ctnames):
            CS=np.load(os.path.join(work,"frames","CS_{}.npy".format(cname)))
            if pos==0:
                CS_norm=np.zeros(CS[2].shape)
            CS_norm+=CS[2]
            
    corrmean=0
    
    ESAnorm=df_label.loc[:,["lat","lon"]+ctnames]
    ESAnorm=ESAnorm.groupby(["lat","lon"]).sum() # fraction of each class per cell
    
    ESAnorm=ESAnorm.sum(1).compute() # total cloud amount per cell
    ESAnorm = ESAnorm.to_frame("cloud_amount")
    
    
    
    for pos,cname in enumerate(ctnames):
        #compute maximum cloud fraction for color bar scale
        cloud = df_label.loc[:,["lat","lon",cname]]
        cloud = cloud.groupby(["lat","lon"]).sum()
        cloud=cloud.compute()
        
        cloud.loc[:,cname]=cloud.loc[:,cname]/ESAnorm.iloc[:,0]
        cloudmax=max(cloud.loc[:,cname].max(),cloudmax)
    print("cloudmax",cloudmax)
    #individual cloud densities        
    for pos,cname in enumerate(ctnames):
        cloud = df_label.loc[:,["lat","lon",cname]]
        cloud = cloud.groupby(["lat","lon"]).sum()
        #going to plot the total amount of each cloud type per cell
        #relative to the total amount of clout in each cell
        cloud=cloud.compute()
        
        cloud.loc[:,cname]=cloud.loc[:,cname]/ESAnorm.iloc[:,0]#relative
        cloud.reset_index(level=["lat","lon"],inplace=True)
        #im goint to need this to manually create a grid
        un_lat = np.unique(cloud.lat)
        un_lon = np.unique(cloud.lon)
        print(len(un_lat), len(un_lon))
        colors=cloud.loc[:,cname].values
        #gg=np.empty((len(step_lat),len(step_lon)))*np.nan
        gg=np.ones((len(un_lat),len(un_lon)))*-1
        #i have the locations of the cells and theri values in order, 
        #but now i populate a grid with that
        for x,y,z in zip(cloud.lat,cloud.lon,colors):
            i = np.argwhere(un_lat==x)
            j= np.argwhere(un_lon==y)
            gg[i,j] = z
        #i dont care bout points wehre there is no cloud
        gg=np.where(gg<0,np.nan,gg)
        lonlon,latlat = np.meshgrid(un_lon,un_lat)
        pmeshdict[cname]=np.stack((gg,lonlon,latlat))
        print(latlat.shape,lonlon.shape,gg.shape, np.max(latlat), np.max(lonlon), 
              np.sum(np.isnan(gg)))
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
        if "2ESA" in name:
            CS=np.load(os.path.join(work,"frames","CS_{}.npy".format(cname)))
            CS_lon = CS[0]
            CS_lat = CS[1]
            
            if pos==0:
                stash=np.zeros(CS[2].shape)
            CS = CS[2]
            
            
            figdiff=plt.figure(figsize=(12,7))
            axdiff=figdiff.add_subplot(1,1,1,projection=ccrs.PlateCarree())
            
            
            #for colorbar
            minlat=max(np.min(CS_lat),np.min(latlat))
            maxlat=min(np.max(CS_lat),np.max(latlat))
            minlon=max(np.min(CS_lon),np.min(lonlon))
            maxlon=min(np.max(CS_lon),np.max(lonlon))
            
            difflon,difflat=np.meshgrid(np.arange(minlon,maxlon,2),np.arange(minlat,maxlat,2))
            
            #first we find the are of the date that is within the grid cell
            diffCS = CS[FI(CS_lat[:,0],minlat):FI(CS_lat[:,0],maxlat),
                        FI(CS_lon[0],minlon):FI(CS_lon[0],maxlon)]
            #and the the corresponding cloud amount
            CS_norm_temp = CS_norm[FI(CS_lat[:,0],minlat):FI(CS_lat[:,0],maxlat),
                        FI(CS_lon[0],minlon):FI(CS_lon[0],maxlon)]
            
            #and we use that to normalize
            diffCS/=CS_norm_temp
            diffCS=np.where(np.isnan(diffCS),0,diffCS)
            assert difflon.shape==diffCS.shape, (difflon.shape, diffCS.shape)
            diffESA=gg[FI(latlat[:,0],minlat):FI(latlat[:,0],maxlat),
                        FI(lonlon[0],minlon):FI(lonlon[0],maxlon)]
            

            
            
            
            print(np.sum(diffCS==0),diffCS.shape)
            print(np.sum(diffESA==0),diffESA.shape)
            diff = diffESA-diffCS
            #find how large the deviation is at the important points
            p90th = np.percentile(diffESA,90)
            qtdiff=np.where(diffESA>p90th,diff,np.nan)
            np.save("diffESA"+cname,diffESA)
            corr = pearsonr(diffESA.flatten(),diffCS.flatten())[0]
            corrmean+=corr
            print("{}, ESA: {:.3}, CS: {:.3}, 90Abs: {:.3}".format(cname,np.mean(diffESA),
                                                          np.mean(diffCS),
                                                          np.mean(diffESA[diffESA>p90th])))
            print("{}: Corr: {:.3}, diff {:.3}, 90thdiff: {:.3}".format(cname,corr,
                                                                        np.mean(diff),
                                                                        np.nanmean(qtdiff)))
            
            diffplot=axdiff.pcolormesh(difflon, difflat,diff,cmap="gist_stern",shading="nearest",
                                transform=ccrs.PlateCarree(),
                                vmin=-max(np.max(diffCS),np.max(diffESA)),
                                vmax = max(np.max(diffCS),np.max(diffESA)))
            
            cbar=figdiff.colorbar(diffplot, orientation="vertical",fraction=0.06,pad=0.02,
                                shrink=0.75)
            cbar.ax.tick_params(labelsize=16)
            customize(axdiff)
            axdiff.set_title("Mean {} difference".format(cname),fontsize=21)
            figdiff.tight_layout()
            figdiff.savefig(os.path.join(work,"stats",name.replace(".parquet","_diff{}.eps".format(cname))), bbox_inches="tight")

        
        meshplot=ax.pcolormesh(lonlon, latlat,gg,cmap="gist_stern",shading="nearest",
                            transform=ccrs.PlateCarree(),vmin=0,vmax=cloudmax)
        
        cbar=fig.colorbar(meshplot, orientation="vertical",fraction=0.08,pad=0.02,
                            )
        cbar.ax.tick_params(labelsize=12)
        customize(ax)
        ax.set_title("Mean {} fraction".format(cname),fontsize=21)
        xmin, xmax = ax.get_xbound()
        ymin, ymax = ax.get_ybound()
         
        
        y2x_ratio = (ymax-ymin)/(xmax-xmin)
        fig.set_figheight(fig.get_figwidth()* y2x_ratio)
        fig.savefig(os.path.join(work,"stats",name.replace(".parquet","_{}.svg".format(cname))))
        
        del cloud
    
    print("should sum to 1",sum([x for x,y,z in pmeshdict.values()]))
    
    for pos,cname in enumerate(ctnames):    
        gg,lonlon,latlat = pmeshdict[cname]
        aax=allaxes[pos]
        meshplot=aax.pcolormesh(lonlon,latlat,gg,cmap="gist_stern",shading="nearest",
                                transform=ccrs.PlateCarree(),vmin=0,vmax=cloudmax)
        
        if pos>5:
            aax.set_xticks([-140,-70,0,70,140])
            aax.set_xticklabels([-140,-70,0,70,140], fontsize=14)
        if pos%2==0:
            aax.set_yticks([-80,-40,0,40,80])
            aax.set_yticklabels([-80,-40,0,40,80], fontsize=14)
        aax.set_title("{}".format(cname),fontsize=15)
        aax.coastlines()
    
    allcb_ax = allfig.add_axes([0.9, 0.1, 0.03, .78])
    allcbar = allfig.colorbar(meshplot, cax=allcb_ax)
    #allcbar =allfig.colorbar(meshplot, ax=allaxes.ravel().tolist(),
    #                         orientation="vertical",fraction=0.08,pad=0.2,
    #                        shrink=0.8)
    allcbar.ax.tick_params(labelsize=12)
    allfig.suptitle("Mean cloud-type fractions",fontsize=20)
    #allfig.tight_layout()
    allfig.savefig(os.path.join(work,"stats",name.replace(".parquet","_allctypes.svg")))
    
    

    fig, ax = plt.subplots(3,figsize=(8,8))
    bindict ={"lat":np.linspace(-90,90,900),"lon":np.linspace(-180,180,1800),"time":10}
    
    
    #very convoluted way to count how much data I have at each spatiotemporal cell
    futures=[]
    for i in sptemp:
        futures.append(client.submit(map_remote,df_label,i,bindict))
    client.wait_for_workers(2)
    for i,fut in enumerate(as_completed(futures)):
        height ,gp,bindict= fut.result()
        bins=bindict[gp]
        height.compute_chunk_sizes()
        try:
            shape = len(bins)-1
        except TypeError:
            shape = bins-1
        height=height.reshape(-1,shape).sum(axis=0)
        height=height.compute()
        ax[i].bar(bins[:-1], height,label=gp,width=bins[1]-bins[0])
        ax[i].set_xlabel(gp,fontsize=17)   
        ylim_bot = max(0,np.min(height)*0.9)
        ylim_top = np.max(height)*1.02     
        ax[i].set_ylim(ylim_bot,ylim_top)
    
    fig.tight_layout()
    fig.savefig(os.path.join(work, "stats", name.replace(".parquet","_")+"counts.eps"))

    #most dominant cloud type
    most_common=client.submit(get_max_fraction_simple,df_label).result()
    print(most_common.columns)
   
    
    #each pixel is assigned the cloud type which is largest than the mean occurence of the cloud type
    means = df_label.loc[:,ctnames].mean()
    gpby = df_label.groupby(["lat", "lon"])
    most_increased = gpby.mean(split_out=int(df_label.npartitions/10))
    most_increased = most_increased.loc[:,ctnames]/means
    most_increased.columns=np.arange(len(ctnames))
    most_increased = most_increased.idxmax(axis=1).to_frame()
    
    most_common.columns=["mc ctype"]
    most_increased.columns=["mi ctype"]
    #most_increased = client.persist(most_increased)
    most_common = client.persist(most_common)
    
     
    gpby_lat=df_label.groupby("lat")
    #gpby_lon= df_label.groupby("lon")
    #meridional = gpby_lon.agg(["mean", "std"],split_out=int(df_label.npartitions/10)).iloc[:,-16:]
    zonal=gpby_lat.agg(["mean", "std"],split_out=int(df_label.npartitions/10)).iloc[:,-16:]
    
    print([x.npartitions for x in [most_common, most_increased,  zonal]])
    del gpby, gpby_lat
    results=[]
    for i in [most_common,most_increased]:
        progress(i)
        if "mc" in list(i.columns)[0]:
            plotname="most_common"
        elif "mi" in list(i.columns)[0]:
            plotname="most_increased"    
            
        else:
            print("dunno")
            continue
        i.columns=["ctype"]
        print("beforeindex") 
        index=client.compute(i.index,)
        print("afterindex")
        #client.wait_for_workers(12)
        index=index.result()
        progress(index)
        
        coords=np.stack(list(index)).astype("float16")
        del index
        print(coords.shape)
        colors = client.compute(i.ctype.to_dask_array(),)
        print(type(colors))
        colors=colors.result()
        print(type(colors))
        progress(colors)
        fig=plt.figure(figsize=(50,5))
        ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
        customize(ax)
        lon = coords[:,1]
        lat = coords[:,0]
        un_lat = np.unique(lat)
        un_lon = np.unique(lon)
        print(len(un_lat), len(un_lon))
        
        print(coords.shape, np.max(lat), np.max(lon), np.min(lat), np.min(lon))
        gg=np.empty((len(un_lat),len(un_lon)))
        for x,y,z in zip(lat,lon,colors):
            i = np.argwhere(un_lat==x)
            j= np.argwhere(un_lon==y)
            gg[i,j] = z
            
        lonlon,latlat = np.meshgrid(np.unique(lon),np.unique(lat))
        
        print("colors:",np.unique(colors))
        print(latlat.shape,lonlon.shape,gg.shape, np.max(latlat), np.max(lonlon), 
              np.sum(np.isnan(gg)))
        meshplot=ax.pcolormesh(lonlon, latlat,gg,cmap=cMap,shading="nearest",
                           norm=Normalize(0,8), transform=ccrs.PlateCarree())
        plt.draw() 
        # Get proper ratio here
        xmin, xmax = ax.get_xbound()
        ymin, ymax = ax.get_ybound()
         
        
        y2x_ratio = (ymax-ymin)/(xmax-xmin)
        fig.set_figheight(50 * y2x_ratio)
        
        
        #fig.set_figwidth(50 / y2x_ratio)
        fig.tight_layout()
        cbar = fig.colorbar(meshplot, orientation ="horizontal", fraction=0.12,pad=0.02,
                            )
        cbar.ax.get_xaxis().set_ticks(np.arange(8)*1.0+0.5)
        cbar.ax.get_xaxis().set_ticklabels( [ 'Ci', 'As', 'Ac', 'St', 'Sc', 'Cu', 'Ns', 'Dc'],
                                           fontsize=12)
        
        fig.tight_layout()
        fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_")+plotname+".eps"))
        print(plotname, "done")
            
    del colors,coords,most_common,most_increased, i
    
    
    zonal=zonal.compute()
    zonal.sort_index(inplace=True)
    fig, ax =plt.subplots(4,2, sharex=True, figsize=(10,10))
    ax=ax.flatten()
    for i in range(len(ax)):
        if i*2<zonal.shape[1]:
            zonal.iloc[:,i*2].plot(ax=ax[i], label="mean")
            ax[i].set_title(zonal.columns[i*2][0])
            bottom = zonal.iloc[:,i*2]-zonal.iloc[:,i*2+1]
            bottom = np.where(bottom<0, 0,bottom)
            top = zonal.iloc[:,i*2]+zonal.iloc[:,i*2+1]
            ax[i].fill_between(zonal.index.values, (bottom), (top), color='b', alpha=.1, label=u"$\sigma$")
            ax[i].grid()
    ax[i].legend(fontsize=15)
            
    fig.tight_layout()
    fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_")+"zonal.svg"))
    del zonal
    
    #plot timeseries for each cloud type
    #only takes into account cells in the southern hemisphere where at least
    #once the fraction of the cloud type is rather large
    if "times" in globals():
        df_now = df_label[df_label.lat<0]
        gpby = df_now.loc[:, ["lat","lon"]+ctnames].groupby(["lat","lon"]).max()
        condition = gpby.quantile(0.9)
        goodlocs = gpby[gpby>condition].compute()        
        fig, ax =plt.subplots(4,2, sharex=True, figsize=(10,10))
        ax=ax.flatten()
        for i,cname in enumerate(ctnames):
            select = goodlocs.loc[:,cname]
            sometimes_large = select[select>0]
            sometimes_large = sometimes_large.reset_index()
            large_lat = sometimes_large.lat
            large_lon = sometimes_large.lon
            
            sub = df_now.loc[:,["lat","lon","time",cname]]
       
            is_lat = sub.lat.isin(list(large_lat.values))
            is_lon = sub.lon.isin(list(large_lon.values))
            relevant = sub[is_lat&is_lon]
            relevant = relevant.persist()
            progress(relevant)
        
            gpby= relevant.groupby("time")
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

            
            del select,sometimes_large, large_lat,large_lon,sub,is_lat,is_lon,relevant,gpby
        ax[i].legend(fontsize=15)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_")+"temporal.svg"))
            
        del temporal
        
    """
    meridional=meridional.compute()
    meridional.sort_index(inplace=True)
    fig, ax =plt.subplots(4,2, sharex=True, figsize=(10,10))
    ax=ax.flatten()
    for i in range(len(ax)):
        if i*2<meridional.shape[1]:
            meridional.iloc[:,i*2].to_frame().plot(ax=ax[i])
            ax[i].set_title(meridional.columns[i*2][0])
            bottom = meridional.iloc[:,i*2]-meridional.iloc[:,i*2+1]
            bottom = np.where(bottom<0, 0,bottom)
            top = meridional.iloc[:,i*2]+meridional.iloc[:,i*2+1]
            ax[i].fill_between(meridional.index.values, (bottom), (top),
                               color='b', alpha=.1)
            ax[i].legend(fontsize=17)
            ax[i].grid()
    fig.tight_layout()
    fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_")+"meridional.eps"))
    del meridional
    
    """
