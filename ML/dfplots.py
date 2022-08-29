#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:33:02 2021

@author: arndt
plots several representations of regression results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 400
import os
import sys
from matplotlib.colors import Normalize, LogNorm
import seaborn as sns
from copy import copy
from sklearn.metrics import r2_score
import scipy
import traceback
import glob


def hexbin(x, y,  **kwargs):
    cmap = sns.color_palette("mako_r", as_cmap=True)
    
    plt.hexbin(x, y,  cmap=cmap, **kwargs)
    
    
def load_df(path):
    """Applies the correct loading function depending on datatype
       should really always be .pkl but sometimes its .parquet

    Args:
        path (string): file basename

    Raises:
        FileNotFoundError: wrong name or path

    Returns:
        pandas DataFrame: the data
    """
    split = os.path.splitext(path)
    name=split[0]
    i=0
    while True:
        try:
            path=glob.glob(name+"*")[i]
        except IndexError:
            raise FileNotFoundError
        ft=os.path.splitext(path)[-1]
        if (ft==".pkl") or (ft==".parquet") or (ft==".hdf"):
            break
        else:
            i+=1
    print("ft",ft)
    if ft==".pkl":
        return pd.read_pickle(path)
    elif ft==".hdf":
        return pd.read_hdf(path, key="data")
    elif ft==".parquet":
        print("cry")
        return pd.read_parquet(path)

def adj_r2(true,pred, trees=400):
    r2=r2_score(true,pred)
    return (1-((1-r2)*(len(true)-1)/(len(true)-trees-1)))

if __name__=="__main__":
    
    root=os.environ["WORK"]+"/frames"
    try:
        name = sys.argv[1]
        if sys.argv[1]=="fancy":
            fancy=1
        else:
            fancy =0
    except Exception:
        root = "/mnt/work/frames/"
        name = "testframe100_3177_sep_0123459.pkl"
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    clouds =["clear" , "Ci", "As", "Ac", "St", "Sc", "Cu", "Ns", "Dc"]
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
    
    path=os.path.join(root,name)
    df=load_df(path)
    
    #df=df.iloc[:100]
    plt.close("all")
    l=df.columns.get_loc("clear")
    if isinstance(l,np.ndarray):
        df.columns=list(df.columns[:-9])+[x+"_p" for x in df.columns[-9:]]
        print(df.columns)
        l=df.columns.get_loc("clear")
        
    if "lat" in df.columns:
        l-=1
    if "lon" in df.columns:
        l-=1
    if "time" in df.columns:
        l-=1
    if l<=5:
        dimx=l
        dimy=1
    elif l==6:
        dimx=3
        dimy=2
    elif l==7:
        dimx=l
        dimy=1
    elif l==8:
        dimx=4
        dimy=2
    elif l==9:
        dimx=dimy=3
    elif l==10:
        dimx=5
        dimy=2
    elif l==11:
        dimx=4
        dimy=3
    else:
        raise ValueError("confused {}".format(l))
    
    #stats =df.describe()
    ins = df.iloc[:,:l]
    clouds_p =[]
    #this makes sure that for old files real and predicted cloud types have different column names
    for c in copy(clouds):
        print(c,clouds)
        if c not in df.columns:
            clouds.remove(c)
        else:
            clouds_p.append(c+"_p")
    outs = df.loc[:,clouds]
    
    bins = np.linspace(0,1,20)
    
    exponent=3
    yticks=[100]
    while max(yticks)<len(outs):
        yticks.append(10**exponent)
        exponent+=1
    
    ifig, iax=plt.subplots(dimx,dimy, figsize=(10,10))
    iax=iax.flatten()[:l]
    inhist = ins.hist(ax=iax, log = True, bins=20)
    for ix in iax.flatten():
        ix.set_yticks(yticks)
    #just plotting histograms for all values
    ifig.tight_layout()
    ifig.suptitle("inputs")
    ifig.savefig(os.path.join(root,"../stats","{}".format(
        name.replace("pkl", "inputs.eps"))))
    
    try:
        raise Exception("dont want weights")
        if "test" in name:
            raise Exception("dont need weights in test")
        weights=np.load("weightscloudy.npy")
        if len(weights)!=len(outs):
            raise Exception("old weights", len(weights), len(outs))
    except Exception as err:
        print(err)
        weights = np.ones(len(outs))
        
    ofig, oax=plt.subplots(3,3)
    oax=oax.flatten()[:len(clouds)]
    outhist= outs.hist(ax=oax, log = True, weights = weights, bins=bins)
    for ix in oax.flatten():
        ix.set_yticks(yticks)
    
    ofig.tight_layout()
    ofig.suptitle("targets")
    ofig.savefig(os.path.join(root,"../stats","{}".format(
        name.replace("pkl", "targets.eps"))))
    
    if "test" in name:
        
        preds = df.loc[:,clouds_p]
        print(preds.describe())
        pfig, pax=plt.subplots(3,3)
        pax=pax.flatten()[:len(clouds_p)]
        predhist = preds.hist(ax=pax, log = True, bins=bins ) 
        for ix in pax.flatten():
            ix.set_yticks(yticks)
        pfig.suptitle("predictions")
        pfig.tight_layout()
        pfig.savefig(os.path.join(root,"../stats","{}".format(
            name.replace("pkl", "predictions.eps"))))
        
        
        preds=preds.values[:,1:]
        outs=outs.values[:,1:]
        weights = np.sum(outs,0)/np.sum(outs)
        #this gives some scalar metrics for the predictions
        print("weights",weights)
        print("mean abs deviation",np.mean(np.abs(preds-outs),0), 
              "-->","weighted:",  np.average(np.mean(np.abs(preds-outs),0),weights=weights), 
              "unweighted:",np.mean(np.abs(preds-outs))
              )
        reldev = np.where(outs==0,-1e10,np.abs((preds-outs)/outs))
        reldev = np.ma.masked_equal(reldev,-1e10)
    
        print("mean relative deviation",np.ma.mean(reldev,0)
              ,"-->", "unweighted:",np.ma.mean(reldev),
              "weighted:",np.ma.average(np.ma.mean(reldev,axis=0),weights=weights))
        print("median abs deviation",np.median(np.abs(preds-outs),0),
              "-->", "unweighted:", np.median(np.abs(preds-outs)),
              "weighted:",np.average(np.median(np.abs(preds-outs),0),weights=weights),
              )
        print("median relative deviation",np.ma.median(np.abs(reldev),0),
              "-->","unweighted:",np.ma.median(np.abs(reldev)),
              "weighted:",np.average(np.ma.median(np.abs(reldev),0),weights=weights),
              )
        r2 = r2_score(outs,preds)
        print(r2, adj_r2(outs,preds))
        for i in range(9):
            print(df.iloc[:,-18+i].corr(df.iloc[:,-9+i]))
            print((np.sum(df.iloc[:,-18+i]>df.iloc[:,-9+i]+.1)
                  + np.sum(df.iloc[:,-18+i]<df.iloc[:,-9+i]-.1))/len(df))
        #compare to randomly sampled predictions from the real distributions
        samplemeans=[]
        for i in range(10):    
            stuff =  []
            
            for i in range(outs.shape[1]):
                hist=np.histogram(outs[:,i],bins=100)
                rvh = scipy.stats.rv_histogram(hist)
                stuff.append(rvh.rvs(size=len(preds)))
            samples=np.stack(stuff).transpose()
            
            samples /= np.sum(samples,1).reshape(-1,1)
            samplemeans.append(np.mean(np.abs(samples-outs)))
        print("random abs deviation",np.mean(samplemeans),"+-",np.std(samplemeans))
        
     
    fancy=1
    
    #for the test split joint density plots are created for real/predicted labels
    if fancy:
        print("fancy")
        name = name[name.find("frame"):]
        test = load_df(os.path.join(root,"test"+name))
        train = load_df(os.path.join(root,"train"+name))
        
        train.pop("cm")  
        temp=train.columns.get_loc("clear")
        if isinstance(temp,np.ndarray):
            train.columns=list(train.columns[:-9])+[x+"_p" for x in train.columns[-9:]]
        temp=test.columns.get_loc("clear")
        if isinstance(temp,np.ndarray):
            test.columns=list(test.columns[:-9])+[x+"_p" for x in test.columns[-9:]]
                
       
        outdev = pd.DataFrame(np.abs(test.loc[:,clouds].values-test.loc[:,clouds_p].values),columns=clouds)

        medprops =dict(linewidth="3")
        infig,inax = plt.subplots(dimx,dimy, figsize=(10,10))
        inax=inax.flatten()
        #boxplots inputs
        for i in range(l):
            print("class",i)
            trainin = train.boxplot(column=train.columns[i],ax=inax[i], positions=[0],
                                  medianprops=medprops)
            testin = test.boxplot(column=test.columns[i], ax=inax[i], positions=[1],
                                  medianprops=medprops)
            inax[i].set_title(test.columns[i])
            inax[i].set_xticklabels(["train", "test"])#, "model"])
        outfig,outax = plt.subplots(3,3, figsize=(10,10))
        outax=outax.flatten()[:len(clouds)]
        
        #boxplots outputs
        for i in range(l,l+len(clouds)):
            print("out",i)
            print(train.columns[i], test.columns[i],test.columns[i+len(clouds)])
            trainout = train.boxplot(column=train.columns[i], ax=outax[i-l], 
                             positions=[0], showfliers=False,showmeans=True,
                          medianprops=medprops)
            testout = test.boxplot(column=[test.columns[i]], ax=outax[i-l], 
                                   positions=[1], showfliers=False,showmeans=True,
                                  medianprops=medprops)
            testout2 = test.boxplot(column=[ test.columns[i+len(clouds)]], ax=outax[i-l], 
                                   positions=[2], showfliers=False,showmeans=True,
                                  medianprops=medprops)
            if "random" not in name:
                deviations = outdev.boxplot(column=outdev.columns[i-l], ax=outax[i-l], 
                                     positions=[3], showfliers=True,showmeans=True,
                                  medianprops=medprops)
           
            outax[i-l].set_title(test.columns[i])
            q3=test.iloc[:,i].quantile(0.75)
           
            if "random" not in name:
                outax[i-l].set_xticklabels(["train", "test\n real", "test\n prediction", "test-\n deviations"])#, "model"])
            else:
                outax[i-l].set_xticklabels(["train", "test\n real", "test\n prediction"])
            
            
        outfig.tight_layout()
        infig.tight_layout()
        print(outax[i-l].get_xticklabels())
        print(outax[i-l].get_xticks())
        outfig.savefig(os.path.join(root,"../stats","outfig"+name[:-4]))
        
        infig.savefig(os.path.join(root,"../stats","infig"+name[:-4]))
        
        #correllation plots
        cutoff= int(len(test))
        test.columns=list(range(l))+list(test.columns[l:l+len(clouds)])+[x+"_p" for x in test.columns[l:l+len(clouds)]]
        fullframe_base = pd.DataFrame(test.iloc[:cutoff,l:])
        #get histogram ranges, exclude zeros
        for partial_min, partial_max in [(1e-4,1)]:
            bmax=0
            bmin=1
            for i in range(len(clouds)):
                print("clouds",i)
                    
                real_partial_min=fullframe_base.iloc[:,i]>partial_min
                pred_partial_min =fullframe_base.iloc[:,i+len(clouds)]>partial_min
                pred_partial_max =fullframe_base.iloc[:,i+len(clouds)]<=partial_max
                real_partial_max=fullframe_base.iloc[:,i]<=partial_max
                index = real_partial_min.values*pred_partial_min.values*real_partial_max.values*pred_partial_max.values
                
                
                fullframe=fullframe_base[index]
                real_vanish=fullframe.iloc[:,i]==0
                pred_vanish =fullframe.iloc[:,i+len(clouds)]<1e-3
                index = real_vanish.values*pred_vanish.values
                plotframe=fullframe[~index]
                gs = min(20,len(np.unique(plotframe.Ci)))
                marg_kws= {"bins":np.linspace(0,1,gs)}
                print(plotframe.iloc[:,i+len(clouds)].sum())
                hist_x,_ = np.histogram(plotframe.iloc[:,i],marg_kws["bins"])
                hist_y,_ = np.histogram(plotframe.iloc[:,i+len(clouds)],marg_kws["bins"])
                
                try:
                    bmax = max(np.max(hist_x),np.max(hist_y), bmax)
                except Exception:
                    continue
            #joint density plots
            for i in range(len(clouds)):
                
                try:
                    print("plots", i,bmin,bmax)
                    
                    real_partial_min=fullframe_base.iloc[:,i]>partial_min
                    pred_partial_min =fullframe_base.iloc[:,i+len(clouds)]>partial_min
                    pred_partial_max =fullframe_base.iloc[:,i+len(clouds)]<=partial_max
                    real_partial_max=fullframe_base.iloc[:,i]<=partial_max
                    index = real_partial_min.values*pred_partial_min.values*real_partial_max.values*pred_partial_max.values
                    #fullframe=fullframe_base[index]
                    real_vanish=fullframe.iloc[:,i]==0
                    pred_vanish =fullframe.iloc[:,i+len(clouds)]<1e-3
                    index = real_vanish.values*pred_vanish.values
                    #plotframe=fullframe[~index]
                    marg_kws= {"bins":np.linspace(0,1,20)}
            
                    jointplot = sns.jointplot(data=plotframe, x=plotframe.columns[i], 
                                              y=plotframe.columns[i+len(clouds)], kind="hex",
                                              marginal_kws=marg_kws, marginal_ticks=True
                                             )
                    jointplot.ax_joint.clear()
                    jointplot.plot_joint(hexbin,gridsize=int(min(20,len(np.unique(plotframe.Ci)))), norm=LogNorm(vmin=bmin, vmax=bmax))
                    jointplot.ax_joint.plot([0,0.5,1],[0,0.5,1],"r",linewidth=2)
                    jointplot.ax_joint.plot([0,0.5],[0,1],"k",linewidth=2)
                    jointplot.ax_joint.plot([0,1],[0,0.5],"k",linewidth=2)
                    jointplot.ax_joint.plot([0,0.5,.9],[0.1,.6,1],"m--",linewidth=2)
                    jointplot.ax_joint.plot([.1,.5,1],[0,0.4,.9],"m--",linewidth=2)
                    jointplot.ax_marg_x.set_yscale("log")
                    jointplot.ax_marg_y.set_xscale("log")
                    jointplot.ax_joint.set_ylabel(axlabel_dict[plotframe.columns[i+len(clouds)]], fontsize=20)
                    jointplot.ax_joint.set_xlabel(axlabel_dict[plotframe.columns[i]], fontsize=20)
                    jointplot.ax_marg_y.set_xticks([jointplot.ax_marg_x.get_yticks()[0], 
                                                    jointplot.ax_marg_x.get_yticks()[-1]])
                    jointplot.ax_marg_y.tick_params(labelsize=9)
                    jointplot.ax_marg_x.tick_params(labelsize=9)
                    jointplot.ax_joint.tick_params( labelsize=18)
                    jointplot.ax_joint.set_ylim(partial_min,partial_max)
                    jointplot.ax_joint.set_xlim(partial_min,partial_max)
                    jfig= jointplot.fig
                    jfig.tight_layout()
                    cbax = jfig.add_axes([.38, .78, .3, .05])
                    cbar=plt.colorbar(cax=cbax, orientation = "horizontal")
            
                    cbar.set_ticks(np.logspace(int(np.log10(bmin)), int(np.log10(bmax)), 3))
                    cbar.set_ticklabels(["{:.1e}".format(x) for x in np.logspace(int(np.log10(bmin)),int(np.log10(bmax)),3)])
                    cbar.ax.tick_params(labelsize=12)
                    if partial_max!=1 or partial_min!=1e-4:
                        jfig.savefig(os.path.join(root,"../stats","joint{}_{}_{}.eps".format(name[:-4], plotframe.columns[i],partial_max)))
                    else:
                        jfig.savefig(os.path.join(root,"../stats","joint{}_{}.eps".format(name[:-4], plotframe.columns[i])))
                except Exception:
                    traceback.print_exc()
