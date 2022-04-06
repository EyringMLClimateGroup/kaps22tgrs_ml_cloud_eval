import pandas as pd
import joblib
import os
import sys
from sklearn.inspection import permutation_importance as pi
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 400

import traceback
import warnings
#from shap import TreeExplainer as te

#work="/dss/dssfs02/pn56su/pn56su-dss-0004/work"

if __name__=="__main__":
    with warnings.catch_warnings():
        
        
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        warnings.filterwarnings("ignore",category=UserWarning)
        work = "/work/bd1179/b309177/"
        models=os.path.join(work,"models")
        frames=os.path.join(work, "frames")
        name=sys.argv[1]
        scoring = ['r2', 'neg_mean_squared_error']
        samplesize=int(2e6)

        traceback.print_exc()
        model=joblib.load(os.path.join(models, name))
        model.njobs=100

        df = pd.read_pickle(os.path.join(frames, name.replace("viforest","testframe")))

        print(df.shape)
        
        x=df.iloc[:samplesize,:-18].values
        y=df.iloc[:samplesize,-18:-9].values
        columns=[x in df.columns if x != "ctp" else "ptop"]
        #sample=df.iloc[10000:,:-18].sample(n=200).values
        
        
        del df
        print("test",columns)
        """
        expl = te(model,data=sample)
        print("explainer")
        s_vals = expl.shap_values(X=x,y=y)
        print("s_vals")
        pred=model.predict(x)
        print(np.abs(s_vals.sum(1) + expl.expected_value - pred).max())
        sys.exit()
        """
        
        
        fig, ax = plt.subplots(figsize=(12,12))
        r_multi=pi(model,x,y, n_jobs=1,n_repeats=20, scoring=scoring)
        for num,metric in enumerate(r_multi):
            print(f"{metric}")
            pm = r_multi[metric]
            for i in pm.importances_mean.argsort()[::-1]:
                if pm.importances_mean[i] - 2 * pm.importances_std[i] > 0:
                    print(f"{columns[i]}"
                                  f"{pm.importances_mean[i]:.3f}"
                                  f" +/- {pm.importances_std[i]:.3f}")
        
        
            
            if num==1:
                pm.importances_mean*=10
                pm.importances_std*=10
                metric="MSE*10"
            elif num==0:
                metric="R2 Score"
            ax.bar(x=np.arange(x.shape[1])+num*0.5,height=pm.importances_mean,
                   yerr=pm.importances_std, label=metric, 
                   #tick_label =columns[:-18],
                   width=0.4)
        ax.set_xticks(np.arange(x.shape[1]))
        ax.set_xticklabels(columns[:-18])
        ax.tick_params(labelsize=20)
        ax.set_title("Permutation importance (test)", fontsize=20)
        fig.legend(fontsize=20,loc="upper left")
        fig.tight_layout()
        fig.savefig(name.replace(".pkl","testpm.eps"))
        
            
            
        
        
        
        df = pd.read_pickle(os.path.join(frames, name.replace("viforest","trainframe")))
        print(df.shape)
        df.pop("cm")
        columns=[x in df.columns if x != "ctp" else "ptop"]
        x=df.iloc[:samplesize,:-9].values
        y=df.iloc[:samplesize,-9:].values
        del df
        print("train",columns)
        fig, ax = plt.subplots(figsize=(12,12))
        r_multi=pi(model,x,y, n_jobs=1,n_repeats=20, scoring=scoring)
        for num,metric in enumerate(r_multi):
            print(f"{metric}")
            pm = r_multi[metric]
            for i in pm.importances_mean.argsort()[::-1]:
                if pm.importances_mean[i] - 2 * pm.importances_std[i] > 0:
                    print(f"{columns[i]}"
                                  f"{pm.importances_mean[i]:.3f}"
                                  f" +/- {pm.importances_std[i]:.3f}")
        
            
            if num==1:
                pm.importances_mean*=10
                pm.importances_std*=10
                metric="MSE*10"
            elif num==0:
                metric="R2 Score"
            print(columns[:-9])
            ax.bar(x=np.arange(x.shape[1])+num*0.5,height=pm.importances_mean,
                   yerr=pm.importances_std, label=metric, 
                   #tick_label =columns[:-9],
                   width=0.4)
        ax.set_xticks(np.arange(x.shape[1]))
        ax.set_xticklabels(columns[:-9])
        ax.tick_params(labelsize=20)
        ax.set_title("Permutation importance (train)", fontsize=20)
        fig.legend(fontsize=20,loc="upper left")
        fig.tight_layout()
        fig.savefig(name.replace(".pkl","trainpm.eps"))
            
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        forest_importances = pd.Series(importances, index=columns[:-9])
        
        fig, ax = plt.subplots(figsize=(12,12))
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title(metric)
        ax.set_title("Feature importances using MDI",fontsize=22)
        ax.set_ylabel("Mean decrease in impurity",fontsize=22)
        #ax.set_xticklabels(columns)
        ax.tick_params(labelsize=22)
        fig.tight_layout()
        fig.savefig(name.replace(".pkl","fimp.eps"))
        
        
