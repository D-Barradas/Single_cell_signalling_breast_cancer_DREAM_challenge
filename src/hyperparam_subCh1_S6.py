#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
import pandas as pd
import dask
import dask.dataframe as dd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold ,RandomizedSearchCV ,train_test_split, cross_val_predict , cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,classification_report, confusion_matrix,accuracy_score,matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
# import matplotlib.pyplot as plt
# # import seaborn as sns
# import cesium as cs
# from cesium import featurize
#from dask_ml.preprocessing import DummyEncoder
import numpy as np
np.random.seed(seed = 101)
# import xgboost as xgb; print('XGBoost Version:', xgb.__version__)


# In[ ]:





# In[2]:


my_colums_to_gene_names= { 
'b.CATENIN':'CTNNB1',
'cleavedCas':'BCAR1',
 'CyclinB':'CCNB1',
 'GAPDH':'GAPDH',
 'Ki.67': 'MKI67',
 'p.4EBP1':'EIF4EBP1',
 'p.Akt.Ser473.':'AKT1',
 'p.AKT.Thr308.':'AKT1',
 'p.AMPK':'PRKAB1',
 'p.BTK':'BTK',
 'p.CREB':'CREB1',
 'p.ERK':'MAPK3',
'p.FAK':'PTK2', 
 'p.GSK3b':'GSK3B', 
 'p.H3':'H3F3A', 
 'p.HER2':'ERBB2', 
 'p.JNK':'MAPK8', 
 'p.MAP2K3':'MAP2K3', 
 'p.MAPKAPK2':'MAPKAPK2',
'p.MEK':'MAP2K1', 
 'p.MKK3.MKK6':'MAP2K3', ## given taht in the network MAP2K3 has bigger conectivity I preserved this 
#  'p.MKK3.MKK6b':'MAP2K6', ### I guess both are interfacing and phospholirated
 'p.MKK4':'MAP2K4', 
 'p.NFkB':'NFKB1', 
 'p.p38':'MAPK1', 
 'p.p53':'TP53',
'p.p90RSK':'RPS6KA1', 
 'p.PDPK1':'PDPK1', 
 'p.PLCg2':'PLCG2', 
 'p.RB':'RB1', 
 'p.S6':'RPS6', 
 'p.S6K':'RPS6KB1', 
 'p.SMAD23':'SMAD2',
    'p.SRC':'SRC', 
 'p.STAT1':'STAT1', 
 'p.STAT3':'STAT3', 
 'p.STAT5':'STAT5A'
}


# In[3]:


big_df = pd.DataFrame()
all_files = [x for x in os.listdir("data/EDITED_CELL_LINES_FILES_COMPLETE/") if ".csv" in x and x != "subchallenge_1_template_data.csv"  ]

#target cell lines subch one
target_cell_lines = ['AU565', 'EFM19', 'HCC2218', 'LY2', 'MACLS2', 'MDAMB436']

#targert genes subchallemnge one
target_genes = ['p.ERK', 'p.Akt.Ser473.', 'p.S6', 'p.HER2', 'p.PLCg2']

dtype = {'treatment': pd.api.types.CategoricalDtype(['EGF' , 'full' , 'iEGFR' , 'iMEK' , 'iPI3K' , 'iPKC'])}
print ("reading files")
for m in all_files:
    #print (m)
    df_temp = pd.read_csv("data/EDITED_CELL_LINES_FILES_COMPLETE/%s"%(m),dtype=dtype)
    big_df = pd.concat([big_df,df_temp],axis=0)


# In[4]:


train_df = big_df[~big_df["cell_line"].isin(target_cell_lines)]

train_df = train_df[train_df["treatment"]!="full"]
train_df =train_df.drop(['Unnamed: 0', 'Unnamed: 0.1'],axis=1)


# In[5]:


#print (train_df.columns)


# In[6]:


# train_df = train_df.categorize(columns="treatment")  
# %time
# train_df.compute()


# In[7]:


X = train_df.drop(target_genes,axis=1)
y = train_df[target_genes]


# In[8]:


#print ("cat4egorize")

#X = X.categorize(columns=["treatment"])

print ("dummies")

my_dummies = pd.get_dummies(X["treatment"])


X= X.drop(['treatment', 'cell_line', 'time', 'cellID', 'fileID'],axis=1)

X = pd.concat([X,my_dummies],axis=1)
# In[9]:


# de = DummyEncoder()


# In[10]:


# print ("get dummies")
# # my_dummies = pd.get_dummies(X["treatment"])

# X = de.fit_transform(X)
#from dask.diagnostics import ProgressBar

#with ProgressBar():
#y= y.compute()


# In[11]:


print ("filling NA")
# y = y.compute()

for m in y.columns :
    y[m].fillna(y[m].mean() , inplace=True )
# X.fillna(0)
# y.fillna(0)


# In[ ]:
print ("scaling")

#with ProgressBar():
scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)

print ("release")
df_temp = pd.DataFrame()
big_df = pd.DataFrame()
train_df = pd.DataFrame()
# In[ ]:


#X_train, X_test, y_train, y_test = train_test_split(scaled_data,y['p.ERK'],test_size=0.33, random_state=101)
#X_train, X_test, y_train, y_test = train_test_split(scaled_data,y['p.Akt.Ser473.'],test_size=0.33, random_state=101)
X_train, X_test, y_train, y_test = train_test_split(scaled_data,y['p.S6'],test_size=0.33, random_state=101)
#X_train, X_test, y_train, y_test = train_test_split(scaled_data,y['p.HER2'],test_size=0.33, random_state=101)
#X_train, X_test, y_train, y_test = train_test_split(scaled_data,y['p.PLCg2'],test_size=0.33, random_state=101)


# In[ ]:

print ("train base model")
base_model = RandomForestRegressor(verbose=2,n_jobs=-1,random_state=101,n_estimators=1000)


# In[ ]:


base_model.fit(X_train, y_train)


# In[ ]:




# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 5000, stop = 7000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 220, num = 22)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 8 ]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[ ]:


cv = KFold(5, shuffle=True)


# In[ ]:


rf_random = RandomizedSearchCV(estimator = base_model, param_distributions = random_grid, n_iter = 100, cv = cv, verbose=2, random_state=101, n_jobs = -1)


# In[ ]:


rf_random.fit(X_train,y_train )


# In[ ]:


print (rf_random.best_params_)


# In[ ]:


# y_pred=rf_random.predict(X_test)


# In[ ]:


# print (classification_report(y_test, y_pred))


# In[ ]:


# print ("MCC %f"%(matthews_corrcoef(y_test, y_pred)))
# print ("Accuracy %f"%(accuracy_score(y_test, y_pred)))


# In[ ]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# In[ ]:


base_accuracy = evaluate(base_model, X_test, y_test)
predictions = base_model.predict(X_test)


# In[ ]:



print ("base model")
print ("R^2:",r2_score(y_test, predictions))
print ("MAE:",mean_absolute_error(y_test, predictions))
print ("MSE:",mean_squared_error(y_test, predictions))


# In[ ]:


best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test )

predictions = rf_random.predict(X_test)
print ("best RFR")
print ("R^2:",r2_score(y_test, predictions))
print ("MAE:",mean_absolute_error(y_test, predictions))
print ("MSE:",mean_squared_error(y_test, predictions))

# In[ ]:


print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

