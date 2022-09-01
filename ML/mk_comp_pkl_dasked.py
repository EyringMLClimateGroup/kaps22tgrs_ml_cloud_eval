"""
terrible code that creates gigantic parquet files
these files can be used to crossreference properties and cloud types from IResNEt prediction
"""
import numpy as np
import glob
import os
import pandas as pd
import multiprocessing as mlp
import sys
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import dask.array as da
import dask.dataframe as dd
from distributed import Client, progress, as_completed
import multiprocessing as mlp
from tqdm import tqdm
from memory_profiler import profile
import tracemalloc
import traceback

def random_argmax(occ):
    i = da.random.randint(8)
    if len(occ.shape)>1:
        return (da.argmax(da.roll(occ,i, axis=1),1)-i)%8
    else:
        return (da.argmax(da.roll(occ,i))-i)%8
 
def process(labelled,source, cloudsat):

    try:
        labelled = da.where(da.isnan(labelled),-1,labelled).astype(np.int16)

        cloudsat_variables , cloudsat_locs, interesting = deal_with_tiles(cloudsat)
        if da.sum(interesting).compute()==0:
            return
        cloudsat=da.sum(cloudsat[interesting,-8:],(2,3)).astype(np.int16)
        [x.compute_chunk_sizes() for x in [cloudsat, cloudsat_variables, cloudsat_locs]] 
        
        cloudsat_v_l_l = da.hstack((cloudsat_variables.reshape(-1,9), random_argmax(cloudsat).reshape(-1,1), cloudsat_locs.reshape(-1,2)))
        
        source[:3] = da.where(~(source[:3]>=0), 0, source[:3])
        interesting = da.all(source>=0,1)
        source = source[interesting]
        labelled = labelled.reshape(-1,1)[interesting]
        [x.compute_chunk_sizes() for x in [source, labelled]]
        
        source_all =da.hstack((source, labelled))
        
        cloudsat_dist = da.bincount(random_argmax(cloudsat), minlength=8)
        labelled_dist = da.bincount(labelled[labelled>=0], minlength=8)
        
        return cloudsat_v_l_l, source_all, cloudsat_dist, labelled_dist      
    except FileNotFoundError as err:
        traceback.print_exc() 
        return
def load(f):
    name = os.path.basename(f)
    labelled = da.from_array(np.load(os.path.join(labelled_files, name)))
    source = np.load(os.path.join(npz,"train", name.replace("npy", "npz")))
    source = da.from_array(source["properties"].reshape(9,-1).transpose().astype(np.float16))
    cloudsat = da.from_array(np.load(os.path.join(cloudsat_files, name)))
    return labelled,source,cloudsat

def getstack(f,pool,arg):
    out=[]
    
    for elem in tqdm(arg,total=len(arg)):
        try:
            labelled,source,cloudsat=load(elem)
        except FileNotFoundError:
            #traceback.print_exc()
            continue
        
        client.wait_for_workers(2) 
        labelled= client.scatter(labelled)
        source = client.scatter(source)
        cloudsat=client.scatter(cloudsat)
        out.append(client.submit(process, labelled,source, cloudsat))
    return out

def bincount_flat(x):
    return da.bincount(x.flatten(), minlength=8)

def deal_with_tiles(arr):
    tile = arr[:,13:24].astype(np.float64)
    locs=tile[:,:2]
    tile = da.where(tile<0,np.nan, tile)
    
    tile= tile[:,2:]
    
    interesting = da.all(da.any(tile>0,(2,3)),(1))
    interesting2 = np.all(np.any(tile.compute()>0,(2,3)),(1))
    assert np.all(interesting==interesting2)
    if da.sum(interesting).compute()==0:
        tile_out=da.empty(tile.shape[0],1)
        locs=da.empty(tile.shape[0], 2)
    else:

        tile=tile[interesting]
        tile.compute_chunk_sizes()
        locs=locs[interesting]
       
        locs = da.mean(locs, (2,3))
        tile_out = da.zeros((tile.shape[0], tile.shape[1]))
        tile_out[:,da.array([0,1,2,4,5,6,7,8])]= da.nanmean(tile[:,
                                                     da.array([0,1,2,4,5,6,7,8])],
                                                            (2,3))
        tile[:,3] = da.where(~(tile[:,3]>=0),0,tile[:,3])
        if np.prod(tile.shape)>0: 
            mostlyphase=tile[:,3].reshape(-1,9).astype(int)
            mostlyphase = da.stack(list(map(bincount_flat,mostlyphase)))
        
            tile_out[:,3] = da.nanargmax(mostlyphase,axis=1)
    return tile_out.astype(np.float16), locs.astype(np.float16), interesting

def main_fct(namelist_chunked,start,cloudsat_dist, labelled_dist):

    for k,namelist_chunk in enumerate(namelist_chunked):
        
        if k<=start:
            continue
        print("{}/{}".format(k,len(namelist_chunked)))

        pool=None#mlp.Pool(10)
        stack = getstack(process, pool, namelist_chunk)
            

        try:
            ls1,ls2,ls3,ls4 = [],[],[],[]
            results = client.gather(stack)
             
            for x in results:#as_completed(stack, with_results=False):
                    
            
                if x is not None:
                    ls1.append(x[0])
                    ls2.append(x[1])
                    ls3.append(x[2])
                    ls4.append(x[3])

                del x
        
           
        except ValueError as err:
            print("118",err)
            continue
        
        cloudsat_all=da.vstack(ls1).squeeze()
        source_all=da.vstack(ls2).squeeze()
        del ls1, ls2
        cloudsat_dist+=(da.sum(da.stack(ls3),0)).copy()
        labelled_dist+=(da.sum(da.stack(ls4),0)).copy()
        del ls3,ls4

        cs = dd.from_array(cloudsat_all)
        del cloudsat_all
        cs.columns=props+["labels","lat", "lon"]

        print("saving")
        cs.to_parquet(os.path.join(work,"parquets/CS{}.parquet".format(k)))
        del cs
        source = dd.from_array(source_all)
        del source_all
        source.columns=props+["labels"]
        source.to_parquet(os.path.join(work,"parquets/labelled{}{}.parquet".format(date,k)))
        del source

        [client.cancel(x) for x in stack] 
        for x in  stack:
            del x
        del results
        del stack
        if k >240:
            break

        return

if __name__=="__main__":
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

    work =os.environ["WORK"] 
    date="2021-10-15phy"
    labelled_files = os.path.join(work, 
                      "lrz_data/oob/results/{}/best/out/predicted-label-random".format(date))
    cloudsat_files = os.path.join(work, "lrz_data/oob/numpy/label/tiles")
    npz = os.path.join(work, "lrz_data/oob/npz/")
    cloudsat_dist = da.zeros(8, dtype=np.float128)
    labelled_dist = da.zeros(8, dtype=np.float128)
    

    props = ["cwp", "cod","cer", "phase", "pressure", "height", "top_temp", "emiss.", "surf_temp" ]
    units = ["g/m²", "", "µm","", "hPa", "m", "K", "", "K"]
    clouds = ["Ci", "As" , "Ac", "St", "Sc", "Cu", "Ns", "Dc"]

    namelist = glob.glob(os.path.join(labelled_files, "*.npy"))
    namelist = list(np.random.choice(namelist,2000,replace=False))
    namelist.sort()

    chunksize=1000
    namelist_chunked = [namelist[i:i+chunksize] for i in range(0,len(namelist),chunksize)]
    print(len(namelist_chunked))
    start=0
    CS_outpkls = glob.glob(os.path.join(work, "parquets/CS*.parquet"))
    
    for outpkl in CS_outpkls:
        end=outpkl.find(".parquet")
        CS = outpkl.find("CS")
        start=max(start,int(outpkl[CS+2:end]))
    print("alread have {} files".format(start))
    for go in range(start,len(namelist_chunked)):
        main_fct(namelist_chunked,go,cloudsat_dist, labelled_dist)
