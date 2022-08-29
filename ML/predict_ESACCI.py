#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:49:20 2021

@author: arndt
uses trained RF to predict on ESACCI
handles files sequentially, saves filenames of previously handled files
therefore for one experiment the same batchsize should always be used because it is checked if the BATCHES of files already have been handled
"""

import numpy as np
import torch
import timeit

import os 
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from prefetch_generator import BackgroundGenerator

import joblib
from datetime import datetime
from src.loader import ESACCIDataset, ESACCI_collate
from src.utils import Normalizer, get_chunky_sampler, get_avg_tiles, get_avg_all
from src.utils import get_dataset_statistics, get_ds_labels_stats
from tqdm import tqdm
import traceback
import multiprocessing as mp
import signal

def bgsave(x,fn,outpath):
    """call in subprocess to save in background"""
    pq.write_to_dataset(x,outpath)
    np.save(outpath.replace(".parquet","_fn.npy"),np.array(fn))
    return

def signal_handler(signum, handler):
    raise KeyboardInterrupt("signal")

import warnings

if __name__=="__main__":
    abs_start = timeit.default_timer()
    
    
    signal.signal(signal.SIGTERM, signal_handler)
    
    ctx = mp.get_context("forkserver")
    
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    
    
    clouds =["clear" , "Ci", "As", "Ac", "St", "Sc", "Cu", "Ns", "Dc"]
    properties = ["lwp", "iwp","cer_liq", "cer_ice","cot","ctp","cth",  "ctt","cee","stemp_cloudy"
                   ,"cfc"]
    """
    dtypes = {"clear": "float32" , "Ci", "As", "Ac", "St", "Sc", "Cu", "Ns", "Dc","lwp",
              "iwp","cer_liq", "cer_ice","cot","ctp","cth",  "ctt","cee","stemp_cloudy"
                   ,"cfc"}
    """
    # switches
    classes = 9
    
    
    pim = False
    batch_size =15 
    
    resolution = 100
    
    rel_or_abs = "sep"
    
    regional = 0
    if regional:
        reg_code = "new"
    else:
        reg_code =""
    hyper = 0
    num_files = 1000
    warn = 0
    create_new=0
    
    
    work= os.environ["WORK"]  
    model_dir = os.path.join(work, "models")
    variables = np.array([9])
    variables_ds = np.array([0,10])
    varstring = ""
    for i in variables: varstring+=str(i)
    time = "de" #mn monthlymean, dn dailymean, de daily instnataneous
    
    with open("experiment_log.txt", "a") as of:
        print("predict_ESACCI.py("+str(datetime.today())+") : "+sys.argv[1], file=of)
        
    folder =os.path.join(os.environ["WORK"],"ESACCI/daily/chunkable")
    if "d" in time:
        assert "daily" in folder
    elif "m" in time:
        assert "monthly" in folder
    
    
    te_ds = ESACCIDataset(root_dir=folder, normalizer=None, 
                      indices=None,  variables = variables_ds,
                      chunksize = 10 , output_tiles=False, time_type = time,
                      subsample=int(2e6))
    
    outpath =os.path.join(work,"frames/parquets",
                                 "ESACCI_{}frame{}_{}_{}_{}.parquet".format(time[0]+time[-1]+reg_code,
                                                                    resolution, 
                                                        num_files,
                                                        te_ds.chunksize,
                                                        varstring
                                                        ))
    
    try:
        filenames = list(np.load(outpath.replace(".parquet","_fn.npy"), allow_pickle=True))
        filenames = [[os.path.basename(y) for y in x] for x in filenames]
        for fn in filenames:
            for single in fn:
                single = os.path.join(folder,single)
                te_ds.file_paths.remove(single)
    except FileNotFoundError:
        if os.path.exists(outpath):
            os.system("rm -r {}".format(outpath))
        filenames = []
    if len(filenames)>0:
        batch_size=len(filenames[0])
    
    
    loaderstart = timeit.default_timer()
    
    
    print(te_ds.variables, len(te_ds))
    sampler = get_avg_all(te_ds, random=False)
    testloader = torch.utils.data.DataLoader(te_ds, batch_size=batch_size,
                                          sampler=sampler,collate_fn=ESACCI_collate,
                                          num_workers=batch_size, pin_memory=0)
    
    model = joblib.load(os.path.join(work,"models",
                                         "viforest{}_{}_{}_{}.pkl".format(resolution, 
                                                                num_files,
                                                                rel_or_abs,
                                                                varstring)))
    
    model.n_jobs=-1
    print(type(model))
    
    
    gen = tqdm(enumerate(BackgroundGenerator(testloader)), total=len(testloader),file=sys.stdout)
    
    properties=list(np.array(properties)[variables])
    baseline = np.arange(-90,90,0.1)
    try:
        for j,out in gen:
            fn,i,locs = out
            fn=[os.path.basename(x) for x in np.unique(fn)]
            assert len(i)==len(locs),(i.shape,locs.shape)
            if j<len(testloader)-2:
                if len(filenames)>0:
                    assert len(fn)==len(filenames[0]),(fn,filenames[0])
            filenames.append(fn)
            interesting = torch.ones(len(i),dtype=bool)
            """
            if regional:
                #interesting *= locs[:,1]>90
                #interesting *= locs[:,1]<135
                interesting *= locs[:,0]>5
                interesting *= locs[:,0]<15
            """
            if torch.sum(interesting)==0:
                continue
            
            assert not torch.any(torch.isnan(locs)) and not torch.any(torch.isinf(locs))
            x_np = i.numpy()[interesting][:]
            
            t=model.predict(x_np)
            
            locations = locs.numpy()[interesting][:len(x_np)]
           
            inout= pd.DataFrame(np.hstack((x_np,locations,t)),columns=["twp"]+properties+["lat","lon","time"]+clouds)
            
            inout = pa.Table.from_pandas(inout)
            
            try:
                proc.join()
            except Exception:
                pass
            proc =ctx.Process(target=bgsave,args=(inout,filenames, outpath))
            proc.start()
            gen.set_description("Dataloader, prediction and saving")
            if timeit.default_timer()-abs_start>7.5*3600:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        
        proc.join()
    print("done")
        
