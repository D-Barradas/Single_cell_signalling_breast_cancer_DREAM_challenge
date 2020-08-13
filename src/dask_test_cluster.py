#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import dask


# In[2]:


#!conda install -c conda-forge dask-ml


# In[1]:


#get_ipython().system('module list ')


# In[2]:


#import cudf
import sys, os
import pandas as pd
import dask
from dask_ml.preprocessing import StandardScaler
import dask.dataframe as dd
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold ,train_test_split, cross_val_predict , cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,classification_report, confusion_matrix,accuracy_score,matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor


# In[3]:


#import dask_xgboost
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar
#from dask_cuda import LocalCUDACluster
from  dask_ml.model_selection import train_test_split ,RandomizedSearchCV
#rom  dask_ml.model_selection import train_test_split 
from dask_ml.xgboost import XGBRegressor
#from xgboost import XGBRegressor #; print('XGBoost Version:', xgb.__version__)
#import dask_cudf
import subprocess
from multiprocessing import Process, freeze_support

# In[4]:


np.random.seed(seed = 101)

from dask import compute, persist
from dask.distributed import Client, progress , wait 

from dask_jobqueue import SLURMCluster
#cluster = SLURMCluster(queue="debug", project="k1423", processes=32,  cores = 64, memory = "256GB", walltime="00:30:00" )
cluster = SLURMCluster(queue="workq", project="k1423", processes=64,  cores = 128, memory = "512GB", walltime="24:00:00" )
client = Client(cluster)
#client = Client(os.environ.get("DISTRIBUTED_ADDRESS"))

# In[8]:
#client = Client('scheduler-address:8786')

#cmd = "hostname --all-ip-addresses"
#process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
#output, error = process.communicate()
#IPADDR = str(output.decode()).split()[0]

#cluster = LocalCUDACluster(ip=IPADDR)

print ("cluster running")
# In[ ]:





# In[9]:


#client


# In[10]:


size = 1005120
npartitions = 8


# In[11]:


big_df = pd.DataFrame()

all_files = [x for x in os.listdir("data/EDITED_CELL_LINES_FILES_COMPLETE/") if ".csv" in x and x != "subchallenge_1_template_data.csv"  ]

#target cell lines subch one
target_cell_lines = ['AU565', 'EFM19', 'HCC2218', 'LY2', 'MACLS2', 'MDAMB436']

#targert genes subchallemnge one
target_genes = ['p.ERK', 'p.Akt.Ser473.', 'p.S6', 'p.HER2', 'p.PLCg2']

dtype = {'treatment': pd.api.types.CategoricalDtype(['EGF' , 'full' , 'iEGFR' , 'iMEK' , 'iPI3K' , 'iPKC'])}
for m in all_files:
    #print (m)
    df_temp = dd.read_csv("data/EDITED_CELL_LINES_FILES_COMPLETE/%s"%(m),dtype=dtype)
#     ddf_temp = dd.from_pandas(df_temp,npartitions=npartitions)
    big_df = dd.concat([big_df,df_temp],axis=0)
    df_temp = pd.DataFrame()


# In[12]:



train_df = big_df[~big_df["cell_line"].isin(target_cell_lines)]

train_df = train_df[train_df["treatment"]!="full"]
train_df =train_df.drop(['Unnamed: 0', 'Unnamed: 0.1'],axis=1)


# In[13]:



X = train_df.drop(target_genes,axis=1)
y = train_df[target_genes]


# In[14]:


print ("cat4egorize")

X = X.categorize(columns=["treatment"])

print ("dummies")

my_dummies = dd.get_dummies(X["treatment"])


X= X.drop(['treatment', 'cell_line', 'time', 'cellID', 'fileID'],axis=1)


# In[15]:


#  y.columns


# In[16]:


# test  = my_dummies.compute()
print ("filling NA")
# y = y.compute()
with ProgressBar():
    for m in y.columns :
        y["%s_c"%(m)] =y[m].fillna(y[m].mean() )#, inplace=True )


# In[17]:


# test.head()
y = y.drop(target_genes,axis=1)


# In[18]:


with ProgressBar():
    X = dd.concat([X,my_dummies],axis=1)
    


# In[17]:


# with ProgressBar():
#     X = dask_cudf.from_dask_dataframe(X)


# In[19]:

print ("scaling")

with ProgressBar():
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_data = scaler.transform(X)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(scaled_data,y['p.ERK_c'],test_size=0.33, random_state=101)


# In[21]:


params = {
  'num_rounds':   100,
  'max_depth':    8,
  'max_leaves':   2**8,
 # 'n_gpus':       1,
'booster':'gbtree', 
'base_score':3 ,   
 # 'tree_method':  'gpu_hist',
  'objective':    'reg:squarederror',
  'grow_policy':  'lossguide'
}


# In[22]:


## Optional: persist training data into memory
X_train = X_train.persist()
y_train = y_train.persist()
#print ("computing")
#with ProgressBar():
#    X_train = X_train.compute()
#    y_train = y_train.compute()


# In[ ]:
print ("training")
bst = XGBRegressor(
                 n_jobs=-1,                 
                 n_estimators=10000,       
                 verbosity=2,
                 objective='reg:squarederror'
                 )
bst.fit( X_train, y_train) 
#bst = dask_xgboost.train( params, X_train, y_train, num_boost_round=params['num_rounds'])

#for tuning parameters
parameters_for_testing = {
    'colsample_bytree':[0.4,0.6,0.8],
    'gamma':[0,0.03,0.1,0.3],
    'min_child_weight':[1.5,6,10],
    'learning_rate':[0.1,0.07],
    'max_depth':[3,5],
    'n_estimators':[10000],
    'reg_alpha':[1e-5, 1e-2,  0.75],
    'reg_lambda':[1e-5, 1e-2, 0.45],
    'subsample':[0.6,0.95]  
}

print("hyperparameters")
#gb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
#     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)
cv = KFold(10, shuffle=True)

rsearch1 = RandomizedSearchCV(estimator = bst, param_distributions = parameters_for_testing, n_jobs=-1,iid=False,n_iter = 100, cv = cv ,scoring='neg_mean_squared_error')
rsearch1.fit(X_train, y_train)

print ('######################################################')
print (rsearch1.grid_scores_)
print('best params')
print (rsearch1.best_params_)
print('best score')
print (rsearch1.best_score_)


#bst.save_model('0001.model')






