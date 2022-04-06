    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 17:07:11 2022

@author: arndt
gets correlation between instantaneous and daily mean predictions
"""

from distributed import Client, progress
import numpy as np
import sys
import os
import traceback
import glob
import dask.dataframe as dd
from scipy.stats import spearmanr,pearsonr
from multiprocessing import Pool

    
def standardize(x):
    return(x-x.mean()/x.std())

if __name__=="__main__":
    
    pool=Pool(2)
    print(sys.argv)
    if len(sys.argv)>=3:
        client = Client(sys.argv[2])
    else:
        SCHEDULER_FILE = glob.glob("/scratch/b/b309177/scheduler*.json")[0]
        
        if SCHEDULER_FILE and os.path.isfile(SCHEDULER_FILE):
            client = Client(scheduler_file=SCHEDULER_FILE)
        
            
    print(client.dashboard_link)
    #dss = "/dss/dssfs02/pn56su/pn56su-dss-0004/"
    #work = dss+"work/"
    d={0:"a",1:"b",2:"c",3:"d",4:"e",5:"f",6:"g",7:"h"}
    work = "/work/bd1179/b309177/"
    ctnames = [ "Ci", "As", "Ac", "St", "Sc", "Cu", "Ns", "Dc"]
    
    #client.wait_for_workers(1)
    name=sys.argv[1]
    files = glob.glob(os.path.join(work,"frames/parquets",name))
    files += glob.glob(os.path.join(work,"frames/parquets",
                                    name.replace("deframe","dnframe")))
    res=[]
    assert len(files)==2,files 
    for num,file in enumerate(files):
    
        sptemp = ["time","lat", "lon"] 
    
    
        res.append(dd.read_parquet( file,
                           columns =sptemp+ctnames,
                           chunksize=1_000_000).sample(frac=1)
            )
        
     
    deframe=res[0]
    dnframe=res[1]
    dboth = client.persist(dd.merge(deframe,dnframe, how="inner",
                                    suffixes=("e","n"),
                                    on=["lat","lon","time"]))
    progress(dboth)
    corr=dboth.corr().compute()
    print(corr.loc[[x+"e" for x in ctnames],[x+"n" for x in ctnames]])
    np.save(os.path.join(work,"stats/corr_inst_daily.npy"),corr)
    sys.exit()
    for cname in ctnames:
        
        deframe=res[0]
        dnframe=res[1]
            
        #samplefrac=1
        """
        rounddict={key:{"lat": 0, "lon": 0,"time":0}[key] for key in sptemp}
        
        twodegreegrid=0
        if not twodegreegrid:
            df=df.round(rounddict)
        """
        
        
        deframe = deframe.loc[:,["lat","lon","time"]+[cname]]
        dnframe = dnframe.loc[:,["lat","lon","time"]+[cname]]
        print(deframe.head(),dnframe.head())
        dboth = client.persist(dd.merge(deframe,dnframe, how="outer",
                                        suffixes=("e","n"),
                                        on=["lat","lon","time"]))
        
        print("joined")
        dboth=dboth.compute()
        dboth=dboth.set_index(keys=["lat","lon","time"])
        e=dboth.iloc[:,0].values
        n=dboth.iloc[:,1].values
        del dboth,deframe,dnframe
        good=~(np.isnan(e)|np.isnan(n))
        e=e[good]
        n=n[good]
        assert ~np.any(np.isnan(e)), e
        assert ~np.any(np.isnan(n)), n
        en=pool.map(standardize,[e,n])
        print("standardized")
        e,n=en
        print(e.mean(),n.mean(),e.std(),n.std(),len(e))
        corr=np.sum(e*n)/len(e)#pearsonr(e,n)
        print(cname,corr)
