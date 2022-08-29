#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:47:19 2021

@author: arndt
trains the random forest and applies it to a test split of the data
"""

import torch

import numpy as np
from datetime import datetime

import timeit
import os
import glob
import sys
import sklearn.ensemble as ens
import joblib
from src.loader import CumuloDataset, SummedDataset,summed_collate,sample_equal
from src.utils import Normalizer, get_chunky_sampler, get_avg_tiles, get_avg_all
from src.utils import get_dataset_statistics, get_ds_labels_stats
import warnings
import pandas as pd
import multiprocessing
from prefetch_generator import BackgroundGenerator
import traceback
from tqdm import tqdm


def read_df(path):
    split = os.path.splitext(path)
    name=split[0]
    try:
        path=glob.glob(name+"*")[0]
    except IndexError:
        raise FileNotFoundError
    ft=os.path.splitext(path)[-1]
    
    if ft==".pkl":
        return pd.read_pickle(path)
    elif ft==".hdf":
        return pd.read_hdf(path, key="data")
    elif ft==".parquet":
        print("cry")
        return pd.read_parquet(path)

if __name__=="__main__":
    client=None
    
    expdate = str(datetime.today())
    abs_start = timeit.default_timer()
    
    clouds =["clear" , "Ci", "As", "Ac", "St", "Sc", "Cu", "Ns", "Dc"]
    clouds_p = [x+"_p" for x in clouds]
    names = [ "cwp","lwp","iwp","cerl","ceri","cod", "ctp", "cth", "ctt", "cee","tsurf"]   
    
    # switches
    classes = 9
    try:
        workers = min(multiprocessing.cpu_count(),int(os.environ["SLURM_CPUS_PER_TASK"]))-1
    except KeyError:
        traceback.print_exc()
        workers = multiprocessing.cpu_count()-1
    
    pim = False #pin memory in dataloader
    batch_size = 15 
    
    resolution =100 # size of grid box to average over
    num_chunks =  5000 # number of samples to draw from each file
    strat_combine = "nophase"  # flag for the dataloader to do something special
    
    
    dummy =0
    num_files =int(sys.argv[2])
    train =0  # train if true or test if false
    warn = 0 #print warnings
    
    
    date = "2021-10-15phy"
    work= os.environ["WORK"]
    nc_dir = os.path.join(work,"lrz_data/")
    
    model_dir = os.path.join(nc_dir,"oob","results",date)
    npz_dir = os.path.join(nc_dir,"oob","npz" )
    label_dir = os.path.join(nc_dir,"oob","results",date,
                               "best/out/predicted-label-random")
    
    train_dir = os.path.join(npz_dir,"train")
    test_dir = os.path.join(npz_dir,"test")
    
    
    
    num_files=min(int(len(glob.glob(train_dir+"/*.npz"))), num_files,
                  int(len(glob.glob(label_dir+"/*.npy"))))
    
    if not dummy:
        with open("experiment_log.txt", "a+") as of:
            print("simpleforest.py("+expdate +") : "+str(sys.argv[1]), file=of)
    
    print("files", num_files)
    
    variables =  np.array([0,2,3,4,5,6])#variables to load
    varstring=""
    for i in variables:
        varstring+=str(i)
    
    assert num_files >0, train_dir
    
    if warn==0:
        warnings.filterwarnings("ignore")
    #hyperparameters for Random Forest
    p = {'depth': 17, #maximum tree depth
              'features' :"sqrt", #how many features to use per tree
              'samples' :0.7, # bagging fraction
              'alpha': 0, #pruning alpha
              "weights":  "dev", # key for a weighting function defined below
              "min" : 2 # min leaf size
              }
    
    trainstart = timeit.default_timer()
    
    # %% run
    x,y,cm =[],[],[]
    #different indexing for the variables which includes cwp explictly
    vars_for_summed_ds = list(variables+1)
    vars_for_summed_ds.insert(0,0)
    vars_for_summed_ds = np.array(vars_for_summed_ds, dtype=int)
    
    if train:
        loaderstart = timeit.default_timer()
        try:
            #raise FileNotFoundError("dont wanna load")
            inout=read_df( os.path.join(work,"frames",
                                             "trainframe{}_{}_{}_{}.pkl".format(resolution, 
                                                                    num_files,
                                                                    strat_combine,
                                                              varstring)))
            x=inout.values[:,:len(vars_for_summed_ds)]
            
            y=inout.values[:,len(vars_for_summed_ds):-1]
            assert x.shape[1]+y.shape[1]+1==inout.shape[1]
            del inout
        except FileNotFoundError:
            traceback.print_exc()
            tr_ds = SummedDataset(train_dir ,label_dir=label_dir, normalizer=None, 
                              indices=np.arange(num_files), variables = vars_for_summed_ds,
                              chunksize = resolution, output_tiles=False,
                              label_fct="rel", filt = None,
                             transform = None, subsample = num_chunks,
                             rs=0, combine_strat=strat_combine)
            
            if tr_ds.random_nonsense:
                strat_combine="random"
            sampler = get_avg_all(tr_ds)
            trainloader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size,
                                                  sampler=sampler,collate_fn=summed_collate,
                                                  num_workers=workers, pin_memory=0)
            tot_l = len(trainloader)
            trainloader=BackgroundGenerator(trainloader)
        
            assert len(tr_ds)>0, len(tr_ds)
            for j,i in tqdm(enumerate(trainloader),total=tot_l, file=sys.stdout):
                fractions = i[2]
                if type(fractions) is not np.ndarray:
                    fractions = fractions.numpy()
                interesting = np.array(np.any(fractions[:,1:]>0, 1))
                fn = i[0]
                
                if type(fractions) is not np.ndarray or type(i[1]) is not np.ndarray:
                    x.append(i[1][interesting].numpy())
                else:
                    x.append(i[1][interesting])
                assert np.max(x[-1])<1e6,np.where(x[-1]>=1e6)
                y.append(np.copy(fractions[ interesting]))
           
            x=np.vstack(x).reshape(-1,len(vars_for_summed_ds))
            y=np.vstack(y).squeeze()
            print(x.shape, y.shape)
            assert not (np.any(np.isinf(x)) | np.any(np.isnan(x))), np.where(np.any(np.isinf(x)|np.isnan(x)))
            assert not (np.any(np.isinf(y)) | np.any(np.isnan(y))), np.where(np.any(np.isinf(y)|np.isnan(y)))  
            assert np.mean(y[:,-1])>0, y.shape
            inout = np.hstack((x,y))
            assert np.mean(inout[:,-1])>0
            if "nophase" not in tr_ds.combine_strat:
                cols = list(np.array(names)[vars_for_summed_ds])
            else:
                cols = list(np.array(["cwp","cer","cod","ctp","cth","ctt","cee","tsurf"])[vars_for_summed_ds])

            cols.extend(clouds)
            
            if"comb" in tr_ds.combine_strat:
                cols.remove("Sc")
            
            inout = pd.DataFrame(inout, columns=cols)
            inout.to_pickle( os.path.join(work,"frames",
                                             "trainframe{}_{}_{}_{}.pkl".format(resolution, 
                                                                    num_files,
                                                                    strat_combine,
                                                                    varstring)))
            
            x = inout.iloc[:,:len(vars_for_summed_ds)].values
            y =inout.iloc[:,len(vars_for_summed_ds):].values
            assert x.shape[1]+y.shape[1]==inout.shape[1],(x.shape,y.shape,inout.shape)
            assert y.shape[1]==9,y.shape
            if not dummy:
                del inout,fractions,interesting               
        print("done with loader", timeit.default_timer()-loaderstart)
        best=0
        

        backend="loky"
        
              
        model = ens.RandomForestRegressor(n_estimators = 400,n_jobs=20, 
                                          oob_score=True, 
                                          max_depth=p["depth"],
                                          max_features=p["features"],
                                          max_samples=p["samples"],
                                          ccp_alpha = p["alpha"],
                                          min_samples_leaf = p["min"])
        try:
            model.n_jobs = min(int(950/(x.nbytes+y.nbytes)*1e9)-1,int(os.environ["SLURM_CPUS_PER_TASK"]))
        except KeyError:
            model.n_jobs = min(int(500/(x.nbytes+y.nbytes)*1e9)-1,200)
 
        print("training", model.n_jobs)
    
    
        if p["weights"]=="bins":
            bn=50
            weights=[]
            hist, bins = np.histogram(y[:,0], bins=bn, density = True)
            w_i = np.array([1/(hist[np.argmin(np.abs(bins[1:]-y[i,0]))])**2 for i in range(len(y))])
            
        elif p["weights"]=="cloudy":    
            w_i = np.array([1-y[i,0] for i in range(len(y))])
            
        elif p["weights"]=="dev":
            means = np.mean(y,0).reshape(1,-1)
            reldev = np.linalg.norm(y-means,1,1)

            w_i=reldev
        elif p["weights"]=="qtdev":
            means = np.mean(y,0).reshape(1,-1)
            reldev = np.linalg.norm(y-means,1,1)
            qt=np.quantile(y,0.9,0,keepdims=True)
            for ct in np.argsort(qt)[0][-3:]:
                reldev = np.where(y[:,ct]>qt[:,ct],reldev/2,reldev)
            for ct in np.argsort(qt)[0][:3]:
                
                reldev+=np.where(y[:,ct]>qt[:,ct],1,0)
            w_i=reldev
       
        if p["weights"] is not None:
            weights = np.where(np.isfinite(w_i), w_i, 1)
            #np.save("weights{}.npy".format(p["weights"  ]), weights )
            
            print("fitting")
            with joblib.parallel_backend(backend): 
                model.fit(x,y, sample_weight = weights)
            print("fit done")
            joblib.dump(model, os.path.join(work, "models",
                                    "viforest{}_{}_{}_{}.pkl".format(resolution, 
                                                            num_files,
                                                            strat_combine,
                                                            varstring)))
            
            loss = model.score(x,y)
        else:
            
            print("fitting")
            with joblib.parallel_backend(backend): 
                model.fit(x,y)
            print("fit done")
            joblib.dump(model, os.path.join(work, "models",
                                        "viforest{}_{}_{}_{}.pkl".format(resolution, 
                                                            num_files,
                                                            strat_combine,
                                                            varstring)))
            
            loss = model.score(x,y)
        print("training score", loss,"out of bag score", model.oob_score_)

        
        if model.oob_score_>best:
            print("best", p)
            best=model.oob_score_
            
            joblib.dump(model, os.path.join(work, "models",
                                    "viforest{}_{}_{}_{}.pkl".format(resolution, 
                                                            num_files,
                                                            strat_combine,
                                                            varstring)))
            

    else:
        try:
            model = joblib.load(os.path.join(work,"models",
                                         "viforest{}_{}_{}_{}.pkl".format(resolution, 
                                                                num_files,
                                                                strat_combine,#"random",#
                                                                varstring)))
        except FileNotFoundError:
            traceback.print_exc()
            model=None
        te_ds = SummedDataset(test_dir ,label_dir=label_dir, normalizer=None,
                      indices=None, 
                      variables =vars_for_summed_ds,
                      chunksize = resolution, output_tiles=False,
                      label_fct="rel", filt = None,
                     transform = None,
                     subsample=num_chunks*10,
                     combine_strat= strat_combine)
        te_sampler = get_avg_all(te_ds)
        testloader = torch.utils.data.DataLoader(te_ds, batch_size=batch_size,
                                              sampler=te_sampler,collate_fn=summed_collate,
                                              num_workers=workers, pin_memory=0)

    print("traintime: {}".format(timeit.default_timer()-trainstart))
    if train:
        sys.exit()
    # %%
    mi,mo,mx = [], [],[]
    for i,t_all in tqdm(enumerate(testloader),total=len(testloader), file=sys.stdout):
        _,tx, ty =t_all[0], t_all[1].numpy(), t_all[2].numpy()
        interesting = np.array(np.any(ty[:,1:]>0, 1))
        tx=tx[interesting]
        ty=ty[ interesting]
        if model is not None:
            t = model.predict(tx)
            mo.append(t)
        mi.append( ty)
        mx.append( tx )
        
   
    mi = np.vstack(mi).astype(np.float32)
    if model is not None:
        mo=np.vstack(mo).astype(np.float32)
    mx = np.vstack(mx).astype(np.float32)
    if model is not None:
        inout = np.hstack((mx,mi,mo)).astype(np.float32)
        if "nophase" not in te_ds.combine_strat:
            cols = list(np.array(names)[vars_for_summed_ds])
        else:
            cols=list(np.array(["cwp","cer","cod","ctp","cth","ctt","cee","tsurf"])[vars_for_summed_ds])
        cols.extend(clouds)
        cols.extend(clouds_p)
        
        if "comb" in te_ds.combine_strat:
            cols.remove("Sc_p")
            cols.remove("Sc")
        
        print(vars_for_summed_ds, cols, inout.shape,mi.shape, mo.shape, mx.shape)
        print("size totalarray", inout.nbytes)
        inout = pd.DataFrame(inout, columns=cols)
    
        print(mi.shape, mo.shape, inout.shape)
        print("mean abs deviation",np.mean(np.abs(mi-mo),0), "-->", np.mean(np.abs(mi-mo)))
        reldev = np.where(mi==0,0,np.abs((mi-mo)/mi))
        reldev = np.ma.masked_equal(reldev,0)
                          
        print("mean relative deviation",np.ma.mean(reldev,0), "-->", np.ma.mean(reldev))
        print("median abs deviation",np.median(np.abs(mi-mo),0),"-->",np.median(np.abs(mi-mo)))
        print("median relative deviation",np.ma.median(np.abs(reldev),0), "-->",np.ma.median(np.abs(reldev)))
        inout.to_pickle( os.path.join(work,"frames",
                                             "testframe{}_{}_{}_{}.pkl".format(resolution, 
                                                                    num_files,
                                                                    strat_combine,
                                                                    varstring)))
    else:
        inout = np.hstack((mx,mi)).astype("float32")
        cols = list(np.array(names)[vars_for_summed_ds])
        cols.extend(clouds)
        
        if "comb" in te_ds.combine_strat:
            cols.remove("Sc")
        inout=pd.DataFrame(inout,columns=cols)
        inout.to_pickle(( os.path.join(work,"frames",
                                             "nopredframe{}_{}_{}_{}.pkl".format(resolution, 
                                                                    num_files,
                                                                    strat_combine,
                                                                    varstring))))
    
    absend = timeit.default_timer()
    print(absend-abs_start)
