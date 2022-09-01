#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 13:14:51 2021

@author: arndt
"""
import traceback
import numpy as np
import glob
import os
import pandas as pd
import time
import queue
import threading
import sys
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cbook as cbook
import timeit
from itertools import product,combinations

import dask.array as da
import dask.dataframe as dd
from distributed import Client, progress
from tqdm import tqdm
from memory_profiler import profile

def getstats(df):
    print(df.shape)
    stats = cbook.boxplot_stats(df, labels =df.columns)
    print("stats",stats)
    return stats

def boxplot(df, ax,index):

    df.boxplot(column=df.columns[0], ax=ax,
                   positions=[index], showfliers=False,showmeans=True,
                     medianprops=medprops, notch=True, whis=(0.1,0.9))
    return

def plot_worker(in_q):
    
    for args in iter(in_q.get,"STOP"):
        boxplot(*args)
        
        in_q.task_done()
        

if __name__=="__main__":
    starttime = timeit.default_timer()

    if len(sys.argv)>=2:
        client = Client(sys.argv[1])
    else:
        while True:
            try:
                SCHEDULER_FILE = glob.glob(os.path.join(os.environ["SCR"],"scheduler*.json"))[0]
            
                if SCHEDULER_FILE and os.path.isfile(SCHEDULER_FILE):
                    client = Client(scheduler_file=SCHEDULER_FILE)
                    break
            except IndexError:
                time.sleep(10)
    
    print(client.dashboard_link)

    min_cs =np.ones(9)*1e8
    min_lab=np.ones(9)*1e8
    max_cs=np.ones(9)*-1e8
    max_lab=np.ones(9)*-1e8
    in_q=queue.Queue(maxsize=9)

    work =os.environ["WORK"] 
    date="2021-10-15phy"
    medprops =dict(linewidth="3")

    props = ["cwp", "cod","cer", "phase", "pressure", "height", "top_temp", "emissivity", "surf_temp" ]
    round_vals = [0,1,1,0,0,-1,0,0,0]
    round_dict = {x:y for x,y in zip(props,round_vals)}
    clouds = ["Ci", "As" , "Ac", "St", "Sc", "Cu", "Ns", "Dc"]
    units = ["g/m²", "", "µm","", "hPa", "m", "K", "", "K"]

    def main_fct():
        CS_pkls = glob.glob(os.path.join(work, "parquets/CS*.parquet"))[:]

        print(len(CS_pkls))
        lab_pkls = glob.glob(os.path.join(work,"parquets/labelled{}*.parquet".format(date)))[:]
        print(len(lab_pkls))
        assert len(lab_pkls)*len(CS_pkls)>0
        CS_pkls.sort()
        lab_pkls.sort()
        fig, ax = plt.subplots(figsize=(10,10))
        cloudsat=np.zeros(8)
        labels = np.zeros(8)

        clazz_labs={x :[] for x in range(8)}
        clazz_css={x:[] for x in range(8)}
        min_cs =np.ones(9)*1e8
        min_lab=np.ones(9)*1e8
        max_cs=np.ones(9)*-1e8
        max_lab=np.ones(9)*-1e8
   
    
    
        for k,(cs_pkl, lab_pkl) in tqdm(enumerate(zip(CS_pkls, lab_pkls)), total=min(len(lab_pkls),len(CS_pkls))):
            
            
            try:
                del cs,source,gpby_lab,gpby_cs
            except Exception:
                pass
            cs = dd.read_parquet(cs_pkl,chunksize=1_000_000)
            cs = cs.round(round_dict).astype("float16")
            source = dd.read_parquet(lab_pkl, chunksize=1_000_000).round(round_dict).astype("float16")
            source = source.persist() 
            
            client.wait_for_workers(1) 

            gpby_lab = source.groupby("labels").size().persist()
            gpby_cs = cs.groupby("labels").size().persist()
            amount_cs = gpby_cs.compute()
            
            cloudsat += amount_cs.values[np.argsort(amount_cs.index)]
            if k==min(len(lab_pkls),len(CS_pkls))-1:
                print("pixels in cloudsat:",np.sum(cloudsat))
                ax.bar(range(8), cloudsat/np.sum(cloudsat), width=0.4, label="cloudsat")
            amount_lab=gpby_lab.compute()
            labels += amount_lab.values[np.argsort(amount_lab.index)][1:]
            
            if k==min(len(lab_pkls),len(CS_pkls))-1 :
                print("pixels in prediction:",np.sum(labels))
                print("histo",labels/np.sum(labels))
                
                 
                ax.bar(np.arange(8)+0.3, 
                   labels/np.sum(labels), width=0.4, label="predicted")
                ax.legend(fontsize=18)
                ax.set_xticks(np.arange(8)+0.15)
                ax.set_xticklabels(clouds, fontsize=17)
                fig.savefig(os.path.join(work, "stats", "histo.png"))
                
                histodf = np.vstack((labels,cloudsat))
                histodf = pd.DataFrame(histodf,index=["predicted","cloudsat"],columns=clouds)
                histodf.to_pickle("histo.pkl")
                fig_lab, ax_lab = plt.subplots(4,2,figsize=(24,12))
                fig_cs, ax_cs = plt.subplots(4,2,figsize=(24,12))
                ax_cs=ax_cs.flatten()
                ax_lab=ax_lab.flatten()
                del labels,histodf,fig,ax,cloudsat 
            for i in range(8):
                temp = client.scatter(source[source.loc[:,"labels"]==i].iloc[:,:-1])
               
                clazz_labs[i].append(temp)
                temp= client.scatter(cs[cs.loc[:,"labels"]==i].iloc[:,:-3])
               
                clazz_css[i].append(temp)
                del temp 
                
                if k==min(len(lab_pkls),len(CS_pkls))-1:
                    
                    
                    clazz_lab = client.gather(clazz_labs[i])
                    clazz_lab= client.persist(dd.concat(clazz_lab))
                    progress(clazz_lab)
                    
                    
                    
                    min_lab = list(map(min,zip(min_lab,clazz_lab.iloc[:,:].min(0).compute())))
                    max_lab = list(map(max,zip(clazz_lab.iloc[:,:].max(0).compute(),max_lab)))

                    clazz_cs=client.gather(clazz_css[i])
                    clazz_cs=client.persist(dd.concat(clazz_cs))
                    progress(clazz_cs)
                    
                    
                    
                    min_cs = list(map(min,zip(clazz_cs.iloc[:,:].min(0).compute(),min_cs)))
                    max_cs = list(map(max,zip(clazz_cs.iloc[:,:].max(0).compute(),max_cs)))
                   
            

            if k==min(len(lab_pkls),len(CS_pkls))-1:        
                for i in range(8):
                    if i==0:
                        min_lab=np.array(min_lab)
                        min_cs = np.array(min_cs)
                        max_lab = np.array(max_lab)
                        max_cs = np.array(max_cs)
                        print("mins", min_lab)
                        print("maxs", max_lab)
                    
                    clazz_lab = client.gather(clazz_labs[i])
                    clazz_cs = client.gather(clazz_labs[i])
                    clazz_lab=dd.concat(clazz_lab)
                    clazz_cs=dd.concat(clazz_cs)
                     
                    clazz_lab-=min_lab
                    clazz_cs-=min_cs
                    
                    clazz_lab/=(max_lab-min_lab)
                    clazz_cs/=(max_cs-min_cs)
                    gen = tqdm(range(9))
                    def BP(j):
                        col_lab = clazz_lab.iloc[:,j].to_frame()
                        
                        
                        col_lab=col_lab.compute()
                        assert np.all(~np.isnan(col_lab))
                       
                        
                        col_lab.boxplot(column=col_lab.columns[0], ax=ax_lab[i], 
                                               positions=[j], showfliers=False,showmeans=True,
                                              medianprops=medprops, notch=True)
                        del col_lab

                        col_cs=clazz_cs.iloc[:,j].to_frame().compute()
                       
                        col_cs.boxplot(column=col_cs.columns[0], ax=ax_cs[i], 
                                               positions=[j], showfliers=False,showmeans=True,
                                              medianprops=medprops, notch=True)
                        del col_cs
                        
                        
                    _=list(map(BP,gen))
                    try: 
                        ax_lab[i].set_xticklabels([x+u"\n  [{}]".format( units[k])  for k,x in enumerate(source.columns[:9])], fontsize =12, rotation=-30)
                        ax_cs[i].set_xticklabels([x+u"\n  [{}]".format( units[k]) for k,x in enumerate(cs.columns[:9])], fontsize =12, rotation=-30)
                        ax_lab[i].set_title(clouds[i], fontsize=16)
                        ax_cs[i].set_title(clouds[i], fontsize=16)
                        
                        aspect0 = ax_lab[i].get_aspect()

                        if type(aspect0) == str:
                            aspect0 = 1.0

                        dy = np.abs(np.diff(ax_lab[i].get_ylim()))
                        dx = np.abs(np.diff(ax_lab[i].get_xlim()))

                    except Exception:
                        traceback.print_exc()
            
        return ax_lab,ax_cs,fig_lab,fig_cs      

       
        
        
        

    ax_lab,ax_cs, fig_lab,fig_cs =  main_fct()
    
    legend_elements = [Line2D([0], [0], color='orange', lw=3, label='Median'),
                       Line2D([0], [0], marker='^', color='white', label='Mean',
                                        markerfacecolor='g', markersize=15)]
    ax_lab[4].legend(handles=legend_elements)
    ax_lab[0].legend(handles=legend_elements)
    ax_cs[4].legend(handles=legend_elements)
    ax_cs[0].legend(handles=legend_elements)
    fig_lab.tight_layout()
    fig_lab.savefig(os.path.join(work, "stats","source_varbox_byclass.png"))
    fig_cs.tight_layout()
    fig_cs.savefig(os.path.join(work, "stats","cs_varbox_byclass.png"))

    mat=np.ones((9,8,8))
    names=np.empty((9,8))
    for (i,j) in combinations(range(8), 2):
        if i==j:
            continue
        c1 = dd.concat(clazz_labs[i]).sample(frac=0.01).astype("float64")
        
        c2 = dd.concat(clazz_labs[j]).sample(frac=0.01).astype("float64")
        
        col1=[str(x)+clouds[i] for x in props]#+["labels"]
        
        col2=[str(x)+clouds[j] for x in props]#+["labels"]
        
        
        
        c1.columns=col1
        c2.columns=col2
        for var in range(9):
            c1=c1.reset_index(True)
            c2=c2.reset_index(True)
           

            c=dd.concat([c1.iloc[:,var],c2.iloc[:,var]],axis=1)
            
           
            print(c.mean().compute())
            
            corr =c.corr()
            
            temp=corr.compute().values
             
            mat[var,i,j]=temp[1,0]
            names[var,i]=j
    
    np.save("mat.npy", mat)
    np.save("name.npy",names)
    print(np.median(mat,(1,2)))

