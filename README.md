# Machine-learned cloud classes from satellite data for process-oriented climate model evaluation  
This project allows the assignment of cloud type distribution to coarse-resolution climate data as explained in the paper "Machine-learned cloud classes from satellite data for process-oriented climate model evaluation" (http://arxiv.org/abs/2205.00743).  
Author: Arndt Kaps, arndt.kaps@dlr.de  

This work requires pixel wise labelled cloud data which was obtained using the CUMULO framework from Zantedeschi et al: https://github.com/FrontierDevelopmentLab/CUMULO . Given input data from MODIS as well as the corresponding label files from CUMULO, a random forest (RF) is trained to predict relative cloud type amount for large grid cells. The grid cell size as well a training features can be chosen by the user. The MODIS data as well as the labels are then averaged to coarse grid cells during the training process.  
The trained RF can then be applied to other climate data. We used the cloud product from ESA CCI as a proof of concept by first coarse graining it.  
Finally a number of statistics and plotting scripts are used to produce the plots shown in the paper. The predictions are saved in the ```.parquet``` format to be consumed in the statistics scripts using the dask library.  
  
---------------
  
###To reproduce:
```  
conda create -n ml_eval python=3.9 matplotlib seaborn numpy pandas dask distributed scipy cartopy pyhdf pytorch torchvision netCDF4 tqdm joblib pyarrow scikit-learn pip
pip install prefetch-generator
```
First adjust the static paths used in all files to match filesystem and data locations. Make sure the ```train``` flag in ```simpleforest.py``` is set to 1.  
```  

python simpleforest.py 100000 "train forest with up to 100k files" 
# set train=0 and run the above command again to apply the trained model to a test split
# this gives output train*.pkl test*.pkl  

# now train/test statistics can be plotted for performance evaluation
python dfinteract.py test*.pkl  # correlation between various metrics
python dfplots.py test*.pkl # statistics of the input/outout distributions
python permimportance.py viforest*.pkl # to obtain feature importances of the model  

# now apply the trained model to new data. The dataloader only works with preprocessed ESA CCI files in npz format
python predict_ESACCI.py "this does predictions. even sequentially for insane amounts of data if you have time-limit constraints"  

# processing of the potentially gigantic parquet file requires a running dask server
# the below scripts will look for a scheduler file for this server
python ESACCI_regional.py ESACCI*.parquet
python inputmaps.py ESACCI*.parquet
python coarsecompare.py ESACCI*.parquet
```
