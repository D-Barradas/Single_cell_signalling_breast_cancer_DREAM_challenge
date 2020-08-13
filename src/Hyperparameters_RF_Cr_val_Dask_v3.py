#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
from dask import compute, persist
from dask.distributed import Client, progress
from dask.diagnostics import ProgressBar
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler
from dask_ml.model_selection import train_test_split# ,KFold,RandomizedSearchCV, GridSearchCV, HyperbandSearchCV #, RandomizedSearchCV
import matplotlib.pyplot as plt
import dask_xgboost as dxgb
import xgboost as xgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold,RandomizedSearchCV , GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,matthews_corrcoef
import joblib
import numpy as np


# In[2]:

import dask_jobqueue
from dask_jobqueue import SLURMCluster
cluster = SLURMCluster(cores=32,
    #                    processes=16,
                        memory="128GB",
                        walltime="24:00:00",
                        queue="workq",
                        interface="ipogif0",

                        project="k1423"
                       )
print(cluster.job_script())
cluster.scale(jobs=16)

#cluster.adapt(minimum=2 , maximum=14 , wait_count=60)
# In[3]:


client = Client(cluster)
#client = Client()


# In[4]:
df = []
files = ['184B5.csv',
'BT483.csv',
'HCC1428.csv',
'HCC1806.csv',
'HCC202.csv',
'Hs578T.csv',
'MCF12A.csv',
'MDAMB231.csv',
'MDAMB468.csv',
'SKBR3.csv',
'UACC3199.csv',
'ZR751.csv']
# for x in os.listdir("data"):
#for x in files:i
#    print (x)
#    df_temp = dd.read_csv("data/%s"%(x) )  #,dtype=dtype)
#    df.append(df_temp)

all_files = [x for x in os.listdir("data/EDITED_CELL_LINES_FILES_COMPLETE/") if ".csv" in x and x != "subchallenge_1_template_data.csv"  ]

for m in all_files:
##     print (x)
    df_temp = dd.read_csv("data/EDITED_CELL_LINES_FILES_COMPLETE/%s"%(m))
    df_temp = df_temp.drop(['Unnamed: 0', 'Unnamed: 0.1'],axis=1)
##     df_temp = dd.read_csv("data/%s"%(x) )  #,dtype=dtype)
    df.append(df_temp)

#df = []
for x in files : # os.listdir("data"):
#    print (x)
    df_temp = dd.read_csv("data/%s"%(x) )  #,dtype=dtype)
    df.append(df_temp)


# In[5]:


ddf = dd.concat(df,axis=0)

subCh1=['AU565', 'EFM19', 'HCC2218', 'LY2', 'MACLS2', 'MDAMB436']

subCh2=['184B5', 'BT483', 'HCC1428', 'HCC1806', 'HCC202', 'Hs578T',
       'MCF12A', 'MDAMB231', 'MDAMB468', 'SKBR3', 'UACC3199', 'ZR751']


subCh3=['BT20', 'BT474', 'BT549', 'CAL148', 'CAL51', 'CAL851', 'DU4475',
       'EFM192A', 'EVSAT', 'HBL100', 'HCC1187', 'HCC1395', 'HCC1419',
       'HCC1500', 'HCC1569', 'HCC1599', 'HCC1937', 'HCC1954', 'HCC2185',
       'HCC3153', 'HCC38', 'HCC70', 'HDQP1', 'JIMT1', 'MCF7',
       'MDAMB134VI', 'MDAMB157', 'MDAMB175VII', 'MDAMB361', 'MDAMB415',
       'MDAMB453', 'MFM223', 'MPE600', 'MX1', 'OCUBM', 'T47D', 'UACC812',
       'UACC893', 'ZR7530']

# In[6]:
train = ddf[~ddf["cell_line"].isin(subCh3)]
test = ddf[ddf["cell_line"].isin(subCh3)]

#ddf = ddf.drop(["cell_line"],axis=1)

print ("filling NA")
# y = y.compute()
with ProgressBar():
    for m in ["p.HER2","p.PLCg2"] :
#         print (df[m])
         train["%s_c"%(m)] =train[m].fillna(train[m].mean() )#, inplace=True )
         test["%s_c"%(m)] =test[m].fillna(test[m].mean() )
 #       ddf["%s_c"%(m)] =ddf[m].fillna(ddf[m].mean() )#, inplace=True )


# In[7]:


#ddf = dd.get_dummies(ddf.categorize()).persist()
train = dd.get_dummies(train.categorize()).persist()
test = dd.get_dummies(test.categorize()).persist()

# In[8]:


#ddf = ddf.drop(["p.HER2","p.PLCg2","cellID","fileID"],axis=1)
train = train.drop(["p.HER2","p.PLCg2","cellID","fileID"],axis=1)
test = test.drop(["p.HER2","p.PLCg2","cellID","fileID"],axis=1)


# In[9]:


rounds = {}
genes= ['b.CATENIN','cleavedCas','CyclinB','GAPDH','IdU','Ki.67','p.4EBP1','p.Akt.Ser473.', 'p.AKT.Thr308.',
        'p.AMPK','p.BTK','p.CREB','p.ERK','p.FAK','p.GSK3b','p.H3','p.HER2_c','p.JNK','p.MAP2K3',
        'p.MAPKAPK2','p.MEK','p.MKK3.MKK6','p.MKK4','p.NFkB','p.p38','p.p53','p.p90RSK','p.PDPK1',
        'p.PLCg2_c','p.RB','p.S6','p.S6K','p.SMAD23','p.SRC','p.STAT1','p.STAT3','p.STAT5']

for x in range(len (genes)):
#     print (x)
    my_list = ['b.CATENIN','cleavedCas','CyclinB','GAPDH','IdU','Ki.67','p.4EBP1','p.Akt.Ser473.',
               'p.AKT.Thr308.','p.AMPK','p.BTK','p.CREB','p.ERK','p.FAK','p.GSK3b','p.H3','p.HER2_c','p.JNK',
               'p.MAP2K3','p.MAPKAPK2','p.MEK',   'p.MKK3.MKK6','p.MKK4','p.NFkB','p.p38','p.p53','p.p90RSK',
               'p.PDPK1','p.PLCg2_c','p.RB','p.S6','p.S6K','p.SMAD23','p.SRC','p.STAT1','p.STAT3','p.STAT5']

    if x+1 <len (genes):
#         my_list = genes 
#         print (x,genes[x], genes[x+1])
        my_list.pop(x)
#         my_list.pop(x)
#         my_list.pop(x+1)
#         print ( my_list )
#         rounds[x] = { "test":[genes[x], genes[x+1]] , "train":my_list}
        rounds[x] = { "test":[genes[x] ], "train":my_list}
#         y = df[[genes[x], genes[x+1]]]


# In[10]:


feats=  ['time','treatment_EGF', 'treatment_full', 'treatment_iEGFR', 'treatment_iPI3K','treatment_iPKC', 'treatment_iMEK']

for k in  rounds :
    for ff in feats: 
        rounds[k]["train"].append(ff)
        


# In[11]:
choice = int(sys.argv[1])
print (choice, type)
print ( rounds[choice]["test"] )

#X = ddf[rounds[choice]["train"]]
#y = ddf[rounds[choice]["test"]]
X_train = train[rounds[choice]["train"]]
y_train = train[rounds[choice]["test"]]

print ("####")
# In[12]:




# In[13]:


print ("split")
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=101)
X_test = test[rounds[choice]["train"]]
y_test = test[rounds[choice]["test"]]

# In[ ]:

X_train = X_train.to_dask_array(lengths=True)
X_test = X_test.to_dask_array(lengths=True)
y_train = y_train.to_dask_array(lengths=True)

print ("scaling")
#scaler = StandardScaler()
#scaler.fit(X_train)
#scaled_data = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

print ("training")

# In[ ]:
base_model = dxgb.XGBRegressor(objective='reg:squarederror',tree_method='hist',verbosity=3,n_jobs=-1,n_estimators=1000,learning_rate=0.010  ,max_depth=0,max_leaves=4,grow_policy='lossguide' )

with joblib.parallel_backend('dask'):
    base_model.fit(X_train, y_train.flatten())
#base_model.save_model('base_line_no_max_deph_lr_%f_%i.model'%(lr,leaves))
#    
predictions = base_model.predict( X_test)
predictions = client.persist(predictions)
#    
#print ("########")
#print ("R^2:",r2_score(y_test.compute(), predictions.compute()))
#print ("MAE:",mean_absolute_error(y_test.compute(), predictions.compute()))
#print ("MSE:",mean_squared_error(y_test.compute(), predictions.compute()))
 
p = predictions.to_dask_dataframe(columns=rounds[choice]["test"])
p.to_csv("my_result_for_%s_SubCh3"%(rounds[choice]["test"][0]))

#parameters_for_testing = {
#    'colsample_bytree':[0.4,0.6,0.8],
#    'gamma':[0,0.03,0.1,0.3],
#    'min_child_weight':[1.5,6,10],
#    'learning_rate':[0.1,0.07],
#    'max_depth':[3,5],
#    'n_estimators':[10000],
#    'reg_alpha':[1e-5, 1e-2,  0.75],
#    'reg_lambda':[1e-5, 1e-2, 0.45],
#    'subsample':[0.6,0.95]  
#}




# In[ ]:

#print ("base model")
#base_model = RandomForestRegressor(n_estimators = 10, random_state = 101,n_jobs = -1)
#with joblib.parallel_backend('dask'):
#    base_model.fit(X_train.compute(), y_train.compute().values.ravel())
#for lr in [0.001 , 0.005, 0.01, 0.05 , 0.1,0.5 , 1.0]:
#     #for leaves in [4,16,64,512,1024]:
#     for leaves in [4,16,32,64]:
#         base_model = dxgb.XGBRegressor(objective='reg:squarederror',tree_method='hist',verbosity=3,n_jobs=-1,n_estimators=1000,learning_rate=lr  ,max_depth=0,max_leaves=leaves,grow_policy='lossguide' )
#         with joblib.parallel_backend('dask'):
#              base_model.fit(scaled_data, y_train.flatten())
#         base_model.save_model('base_line_no_max_deph_lr_%f_%i.model'%(lr,leaves))
#    
#         predictions = base_model.predict( X_test)
#         predictions = client.persist(predictions)
#    
#         print ("########")
#         print ("lr :%f , leaves:%i " %(lr,leaves))
#         print ("R^2:",r2_score(y_test.compute(), predictions.compute()))
#         print ("MAE:",mean_absolute_error(y_test.compute(), predictions.compute()))
#         print ("MSE:",mean_squared_error(y_test.compute(), predictions.compute()))
    
    
    #     plt.scatter(y_test.compute(), predictions.compute())
    #     plt.xlabel('True Values [MAE]')
    #     plt.ylabel('Predictions [MAE]')
    #     plt.axis('equal')
    #     plt.axis('square')
    #     plt.xlim([0,plt.xlim()[1]])
    #     plt.ylim([0,plt.ylim()[1]])
    #     _ = plt.plot([-100, 100], [-100, 100])
    # plt.savefig("MAE_base_model_%s.png"%(key),format="png")
    #     plt.savefig("MAE_base_model_%i.png"%(lr),format="png")
    #
    ## For quick response
    #n_examples = 4 * len(X_train)
    #n_params = 8
    #
    ## In practice, HyperbandSearchCV is most useful for longer searches
    ## n_examples = 15 * len(X_train)
    ## n_params = 15
    #
    #max_iter = n_params  # number of times partial_fit will be called
    # 
    #
    #
    ## In[ ]:
    #
    #print ("RandCV")
    #rf_random = RandomizedSearchCV(estimator = base_model, 
    #                               param_distributions = random_grid, n_iter = 10, 
    #                               cv = cv, verbose=2, random_state=101, n_jobs = -1)
    #
    #
    #
    ## In[ ]:
    #
    #
    #with joblib.parallel_backend('dask'):
    #    rf_random.fit(scaled_data, y_train.flatten())
    #
    #
    ## In[ ]:
    #print ("#### ")
    #
    #print (rf_random.best_params_)
    #
client.shutdown()
