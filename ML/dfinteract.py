#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:43:07 2021

@author: arndt
get correlation heatmaps
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 400
import os
import sys
import matplotlib
from distributed import Client
import dask.dataframe as dd
import seaborn as sns
from matplotlib.lines import Line2D
import glob


def read_df(path):
    split = os.path.splitext(path)
    name=split[0]
    
    if len(glob.glob(name+".*"))==0:
        raise FileNotFoundError(name)
    for i in glob.glob(name+".*"):
        path=i
        if ("pkl" in path) or ("parquet" in path) or ("hdf" in path):
            break
        
    ft=os.path.splitext(path)[-1]
    
    if ft==".pkl":
        return pd.read_pickle(path)
    elif ft==".hdf":
        return pd.read_hdf(path, key="data")
    elif ft==".parquet":
        print("cry")
        return pd.read_parquet(path)
    
if __name__=="__main__":
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    #root="/mnt/work/frames/"
    root = "/work/bd1179/b309177/frames/"
    try:
        name = str(sys.argv[1])
    except Exception:
        name = "ESACCI_frame10_3160_3_0123459.pkl"
    
    path=os.path.join(root, name)
    
    
    df=read_df(path)
    df=df[df.cwp<3000]
    plt.close("all")
    variables =  np.array([0,1,2,3,4,5,9])
    clouds =["clear" , "Ci", "As", "Ac", "St", "Sc", "Cu", "Ns", "Dc"]
    if "test" in path:
        l=df.shape[1]-18
        df.columns = list(df.columns[:l])+clouds+[x+"pred" for x in clouds]
    else:
        l=df.shape[1]-9
        df.columns = list(df.columns[:l])+[x+"pred" for x in clouds]
    
    if "lat" in df.columns:
        l-=1
    if "lon" in df.columns:
        l-=1
    if "time" in df.columns:
        l-=1
    varstring=""
    for i in variables:
        varstring+=str(i)
    
    #stats =df.describe()
    ins = df.iloc[:,:l]
    if "test" in name:
        outs = df.iloc[:,l:l+9]
        #print(outs.describe())
    preds = df.iloc[:,l+9:]
    #print(ins.describe())
    
    #Len=df.shape[-1]
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    """
    nowater = df.iloc[np.where(df.cwp==0)[0],1:]
    noice = df.iloc[np.where(df.iwp==0)[0],np.arange(Len)[np.arange(Len)!=1]]
    #print(nowater.iloc[:,-9:].describe())
    #print(noice.iloc[:,-9:].describe())
    high_cloud = df.iloc[np.where(df.ctp<=500)[0],np.arange(Len)[np.arange(Len)!=4]]
    #print(high_cloud.iloc[:,-9:].describe())
    
    
    fig, ax = plt.subplots(2, figsize=(10,10))
    ip=ax[0].imshow(ins.corr(), norm=norm)
    mp=ax[1].imshow(model_df.corr(), norm=norm)
    ax[0].set_yticks(np.arange(len(ins.columns)))
    ax[0].set_xticks(np.arange(len(ins.columns)))
    ax[1].set_yticks(np.arange(len(model_df.columns)))
    ax[1].set_xticks(np.arange(len(model_df.columns)))
    ax[0].set_xticklabels(list(ins.columns))
    ax[0].set_yticklabels(list(ins.columns))
    ax[1].set_xticklabels(list(model_df.columns))
    ax[1].set_yticklabels(list(model_df.columns))
    fig.colorbar(ip, ax=ax[0])
    fig.colorbar(mp, ax=ax[1])
    fig.tight_layout()
    
    
    """
    
    #variable importance
    
    def lookinrange(df,min_lat,min_lon,max_lat,max_lon,name):
        df=df[df.lat<max_lat]
        df=df[df.lat>min_lat]
        df=df[df.lon<max_lon]
        df=df[df.lon>min_lon]
        name = name.replace(".parquet","_{}{}{}{}.parquet".format(min_lat,min_lon,max_lat,max_lon))
        return df,name
    
    #df,name = lookinrange(df,-90,-180,90,180,name)
    """
    fig,ax=plt.subplots()
    print(df.columns)
    try:
        df.plot.scatter(x="lat",y="lon",c="tsurf_pred",ax=ax)
    except Exception:
        df.plot.scatter(x="lat",y="lon",c="stemp_cloudy",ax=ax)
    fig.savefig("lowcloudrange.eps")
    #print(df.describe)
    """
    if "lat" in df.columns:
        df.pop("lat")
    if "lon" in df.columns:
        df.pop("lon")
    if "time" in df.columns:
        df.pop("time")
    
    legend_elements = [Line2D([0], [0], color='orange', lw=3, label='Median'),
                       Line2D([0], [0], marker='^', color='white', label='Mean',
                                        markerfacecolor='g', markersize=15)]
    print(df.columns)
    #df.columns=["cwp","lwp","iwp","cerl","ceri","cod", "ptop","tsurf"]+clouds
    mins=df.iloc[:,:l].min()
    df.iloc[:,:l]-=mins
    df.iloc[:,:l]/=df.iloc[:,:l].max()
    t=df.quantile(0.9)
    
    """
    lower = df>df.quantile(0.4)
    upper = df<df.quantile(0.6)
    thresholds = lower*upper
    thresholds=thresholds.iloc[:,:l]
    thresholds = np.prod(thresholds,axis=1).values.astype(bool)
    df=df[thresholds]
    print(df.describe())
    fig, ax = plt.subplots(figsize=(14,14),dpi=600)
    medprops =dict(linewidth="3")
    df.Cu.plot(style="." ,ax=ax)
    df.Dc.plot(style="." ,ax=ax)
    
    ax.set_title("on average")
    #ax.set_xticklabels(list(df.columns[l:l+9]), fontsize=13, rotation=-30)
    #fig.suptitle("Variables scaled by: {}".format(maxs.values.astype(int)))
    fig.tight_layout()
    fig.savefig(os.path.join(root, "average.eps"),dpi=600)
    """
    
    name = os.path.splitext(name)[0]
    print("savename",name)
    
    print(df.describe())
    thresholds=t[-9:]
    fig, ax = plt.subplots(3,3,figsize=(14,14),dpi=600)
    ax=ax.flatten()
    medprops =dict(linewidth="3")
    for i in range(9):
        clazz = df.where(df.iloc[:,-(9-i)]>thresholds[i])
        for j in range(len(variables)):
            clazz.boxplot(column=clazz.columns[j], ax=ax[i], 
                                   positions=[j], showfliers=False,showmeans=True,
                                  medianprops=medprops, whis=(0.1,0.9))
        ax[i].legend(handles=legend_elements)
        ax[i].set_title("{} fraction> {}".format(df.columns[i-9][:2],thresholds[i].round(3)))
        ax[i].set_xticklabels(list(df.columns[:len(variables)]), fontsize=13, rotation=-30)
    corr = df.corr().round(2)    
    #fig.suptitle("Variables scaled by: {}".format(maxs.values.astype(int)))
    
    fig.tight_layout()
    fig.savefig(os.path.join(root,"../stats", name+"_stats.eps"),dpi=600)
    #cloud type sensitivitiy
    thresholds=t[:l]
    means = df.describe().iloc[1,-9:]
    fig, ax = plt.subplots(2, int(l//2),figsize=(16,10))
    ax=ax.flatten()
    medprops =dict(linewidth="3")
    for i in range(l):
        clazz = df.where(df.iloc[:,i] > thresholds[i])
        clazz = clazz.iloc[:,-9:]/means
        for j in range(9):
            clazz.boxplot(column=clazz.columns[j], ax=ax[i], 
                                   positions=[j], showfliers=False,showmeans=True,
                                  medianprops=medprops,whis=(0.1,0.9))
        ax[i].legend(handles=legend_elements)
        ax[i].set_title("{} {}".format(df.columns[i][:3],thresholds[i].round(0)))
        ax[i].set_xticklabels([x[:2] for x in df.columns[-9:]])
    #fig.suptitle("Variables scaled by: {}".format(maxs.values.astype(int)))
    fig.tight_layout()
    fig.savefig(os.path.join(root, "../stats", name+"_stats2.eps"))
    
    #variable correlation
    fig, ax = plt.subplots(figsize=(15,15))
    print(df.corr())
    
    cp = sns.heatmap(corr,annot=True,ax=ax,vmin=-1,vmax=1,annot_kws={"size": 18})
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(labelsize=20)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "../stats/ctcorr{}.eps".format(name)))
    
    
    
    fig, ax = plt.subplots(figsize=(15,15))
    print(df.columns)
    corrsub =corr.iloc[-9:,:8]
    cp = sns.heatmap(corrsub,annot=True,ax=ax,vmin=-1,vmax=1,annot_kws={"size": 18})
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(labelsize=20)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "../stats/ctcorr{}_subset.eps".format(name)))
    
    fig, ax = plt.subplots(figsize=(12,12))
    
    corrsub =corr.iloc[:8,:8]
    cp = sns.heatmap(corrsub,annot=True,ax=ax,vmin=-1,vmax=1,annot_kws={"size": 22})
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=22)
    ax.tick_params(labelsize=22)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "../stats/ctcorr{}_inputset.eps".format(name)))
