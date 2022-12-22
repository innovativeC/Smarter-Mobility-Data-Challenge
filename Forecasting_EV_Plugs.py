#!/usr/bin/env python
# coding: utf-8

# ### Loading all the libraries

# In[ ]:


import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import shutil


# In[ ]:


#Loading the datasets given
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
df_holidays = pd.read_csv('holiday_dates.csv', parse_dates=['dates_only'],delimiter=';')
train


# ### Defining functions

# In[ ]:


#Function to perform label encoding of categorical columns
def label_encode(df, col):
    le = LabelEncoder()
    le.fit(df[col].values.tolist())
    new_vals = le.transform(df[col].values.tolist())
    
    df[col] = new_vals
    return df

#Function to train a lightgbm model
def run_lightgbm(numboostRND, earlyStop, catfeatls, tr, val):
    # lgb hyper-parameters
    params = {'metric': 'mae',
          'num_leaves': 356,
          'learning_rate': 0.008,
          'feature_fraction': 0.75,
#           'bagging_fraction': 0.75,
          'bagging_freq': 5,
          'force_col_wise' : True,
          'subsample' : 0.9,
          'random_state': 10}
 
    # Train LightGBM model
    lgb_model = lgb.train(params=params,
                          train_set=tr,
                          num_boost_round=numboostRND,
                          valid_sets=(tr, val),
                          early_stopping_rounds=earlyStop,
                          categorical_feature=catfeatls,
                          verbose_eval=200)
    
    return lgb_model


# ### Engineering Station level data

# In[ ]:


#Joining Test data to train set
train['status'] = 'train'
test['status'] = 'test'
test[['Available', 'Charging', 'Passive', 'Other']] = np.nan
alldata = train.append(test, ignore_index=True)

alldata['date'] = pd.to_datetime(alldata['date'])
alldata['Postcode'] = alldata['Postcode'].astype(str)

#preprocessing
alldata['dates_only'] = alldata.date.dt.date
alldata['dates_only'] = pd.to_datetime(alldata['dates_only'])
alldata = pd.merge(alldata,df_holidays, how='left', on='dates_only')

alldata['is_holiday'].fillna(0, inplace=True)
alldata['is_weekend'] = alldata['dow'] >5

alldata.drop('dates_only',axis=1 ,inplace=True)
alldata['is_holiday'] = alldata['is_holiday'].astype('int')
alldata['is_weekend'] = alldata['is_weekend'].astype('int')

groupsAvail = alldata[['date','Station','Available']].groupby(by=['date','Station']).agg({'Available':'max'})
groupsCharge = alldata[['date','Station','Charging']].groupby(by=['date','Station']).agg({'Charging':'max'})
groupsPass = alldata[['date','Station','Passive']].groupby(by=['date','Station']).agg({'Passive':'max'})
groupsOth = alldata[['date','Station','Other']].groupby(by=['date','Station']).agg({'Other':'max'})

def genLags(df, grp, nlags):
    
    for i in range(1,nlags+1):
        new_grp = grp.groupby(level='Station').shift(i)
        new_grp.reset_index(inplace=True)
        
        df[grp.columns[0]+'_lag'+str(i)+''] = new_grp[grp.columns[0]]
        
    return df
        
alldataAvail = alldata.copy()
alldataCharge = alldata.copy()
alldataPass = alldata.copy()
alldataOth = alldata.copy()

alldataAvail = genLags(alldataAvail, groupsAvail, 15)
alldataCharge = genLags(alldataCharge, groupsCharge, 15)
alldataPass = genLags(alldataPass, groupsPass, 15)
alldataOth = genLags(alldataOth, groupsOth, 15)

alldataAvail


# In[ ]:


alldatafinal = pd.concat([alldataAvail,alldataCharge[['Charging_lag1','Charging_lag2','Charging_lag3',
'Charging_lag4','Charging_lag5','Charging_lag6','Charging_lag7','Charging_lag8','Charging_lag9','Charging_lag10',
'Charging_lag11','Charging_lag12','Charging_lag13','Charging_lag14','Charging_lag15']],alldataPass[['Passive_lag1','Passive_lag2','Passive_lag3',
'Passive_lag4','Passive_lag5','Passive_lag6','Passive_lag7','Passive_lag8','Passive_lag9','Passive_lag10',
'Passive_lag11','Passive_lag12','Passive_lag13','Passive_lag14','Passive_lag15']],alldataOth[['Other_lag1','Other_lag2','Other_lag3','Other_lag4','Other_lag5','Other_lag6',
'Other_lag7','Other_lag8','Other_lag9','Other_lag10','Other_lag11','Other_lag12','Other_lag13','Other_lag14',
'Other_lag15']]], axis =1)

alldatafinal['Postcode'] = alldatafinal['Postcode'].astype(int)

alldatafinaltrain = alldatafinal[alldatafinal['status'] == 'train']
alldatafinaltest = alldatafinal[alldatafinal['status'] == 'test']
alldatafinaltest = alldatafinaltest.drop(['Available','Charging','Passive','Other','status'],axis=1)


# ### Modelling strategy for station level data

# In[ ]:


alldatafinaltrain = label_encode(alldatafinaltrain,'Station')
alldatafinaltrain = label_encode(alldatafinaltrain,'area')
alldatafinaltrain = label_encode(alldatafinaltrain,'date')

alldatafinaltest = label_encode(alldatafinaltest,'Station')
alldatafinaltest = label_encode(alldatafinaltest,'area')
alldatafinaltest = label_encode(alldatafinaltest,'date')


for target in ['Available','Charging','Passive','Other']:
    y = alldatafinaltrain[target]
    x = alldatafinaltrain.drop(['Available','Charging','Passive','Other'], axis =1)
    x = x.drop('status',axis=1)

    x.fillna(0,inplace=True)

    train_x = x[:-91]
    val_x = x[-91:]

    train_y = y[:-91]
    val_y = y[-91:]

    lgbtrainset = lgb.Dataset(train_x, train_y)
    lgbvalidset = lgb.Dataset(val_x, val_y)
    
    lgb_model = run_lightgbm(95000, 9000, ['Station', 'Longitude','Latitude', 'tod','dow', 'Postcode', 'area','trend', 'is_holiday'], lgbtrainset, lgbvalidset)
    
    lgb_model.save_model('lgb_station'+target+'.pkl')


# ### Engineering Area level data

# In[ ]:


train_area = train.groupby(['date', 'area']).agg({'Available': 'sum',
                                                        'Charging': 'sum',
                                                        'Passive': 'sum',
                                                          'Other': 'sum',
                                                          'tod': 'max',
                                                          'dow': 'max',
                                                          'trend': 'max',
                                                            'Latitude': 'mean',
                                                            'Longitude': 'mean',}).reset_index()

test_area = test.groupby(['date', 'area']).agg({
    'tod': 'max',
    'dow': 'max',
    'Latitude': 'mean',
    'Longitude': 'mean',
    'trend': 'max'}).reset_index()



#Joining Test data to train set
train_area['status'] = 'train'
test_area['status'] = 'test'
test_area[['Available', 'Charging', 'Passive', 'Other']] = np.nan
alldata_area = train_area.append(test_area, ignore_index=True)

alldata_area['date'] = pd.to_datetime(alldata_area['date'])

#preprocessing
alldata_area['dates_only'] = alldata_area.date.dt.date
alldata_area['dates_only'] = pd.to_datetime(alldata_area['dates_only'])

alldata_area = pd.merge(alldata_area,df_holidays, how='left', on='dates_only')

alldata_area['is_holiday'].fillna(0, inplace=True)
alldata_area['is_weekend'] = alldata_area['dow'] >5

alldata_area.drop('dates_only',axis=1 ,inplace=True)
alldata_area['is_holiday'] = alldata_area['is_holiday'].astype('int')
alldata_area['is_weekend'] = alldata_area['is_weekend'].astype('int')

groupsAvail = alldata_area[['date','area','Available']].groupby(by=['date','area']).agg({'Available':'max'})
groupsCharge = alldata_area[['date','area','Charging']].groupby(by=['date','area']).agg({'Charging':'max'})
groupsPass = alldata_area[['date','area','Passive']].groupby(by=['date','area']).agg({'Passive':'max'})
groupsOth = alldata_area[['date','area','Other']].groupby(by=['date','area']).agg({'Other':'max'})

def genLags(df, grp, nlags):
    
    for i in range(1,nlags+1):
        new_grp = grp.groupby(level='area').shift(i)
        new_grp.reset_index(inplace=True)
        
        df[grp.columns[0]+'_lag'+str(i)+''] = new_grp[grp.columns[0]]
        
    return df
        
alldata_areaAvail = alldata_area.copy()
alldata_areaCharge = alldata_area.copy()
alldata_areaPass = alldata_area.copy()
alldata_areaOth = alldata_area.copy()

alldata_areaAvail = genLags(alldata_areaAvail, groupsAvail, 15)
alldata_areaCharge = genLags(alldata_areaCharge, groupsCharge, 15)
alldata_areaPass = genLags(alldata_areaPass, groupsPass, 15)
alldata_areaOth = genLags(alldata_areaOth, groupsOth, 15)

alldata_areafinal = pd.concat([alldata_areaAvail,alldata_areaCharge[['Charging_lag1','Charging_lag2','Charging_lag3',
'Charging_lag4','Charging_lag5','Charging_lag6','Charging_lag7','Charging_lag8','Charging_lag9','Charging_lag10',
'Charging_lag11','Charging_lag12','Charging_lag13','Charging_lag14','Charging_lag15']],alldata_areaPass[['Passive_lag1','Passive_lag2','Passive_lag3',
'Passive_lag4','Passive_lag5','Passive_lag6','Passive_lag7','Passive_lag8','Passive_lag9','Passive_lag10',
'Passive_lag11','Passive_lag12','Passive_lag13','Passive_lag14','Passive_lag15']],alldata_areaOth[['Other_lag1','Other_lag2','Other_lag3','Other_lag4','Other_lag5','Other_lag6',
'Other_lag7','Other_lag8','Other_lag9','Other_lag10','Other_lag11','Other_lag12','Other_lag13','Other_lag14',
'Other_lag15']]], axis =1)

alldata_areafinal


# In[ ]:


alldata_areafinaltrain = alldata_areafinal[alldata_areafinal['status'] == 'train']
alldata_areafinaltest = alldata_areafinal[alldata_areafinal['status'] == 'test']
alldata_areafinaltest = alldata_areafinaltest.drop(['Available','Charging','Passive','Other','status'],axis=1)


# ### Trend Analysis on area level

# In[ ]:


##On area level
fig1 = px.line(alldata_areafinaltrain, x = 'date', y = 'Available', color = 'area')
fig1.show()

fig1 = px.line(alldata_areafinaltrain, x = 'date', y = 'Passive', color = 'area')
fig1.show()


# ### Modelling strategy for area level data

# In[ ]:


alldata_areafinaltrain = label_encode(alldata_areafinaltrain,'area')
alldata_areafinaltrain = label_encode(alldata_areafinaltrain,'date')

alldata_areafinaltest = label_encode(alldata_areafinaltest,'area')
alldata_areafinaltest = label_encode(alldata_areafinaltest,'date')

for target in ['Available','Charging','Passive','Other']:
    y = alldata_areafinaltrain[target]
    x = alldata_areafinaltrain.drop(['Available','Charging','Passive','Other'], axis =1)
    x = x.drop('status',axis=1)

    x.fillna(0,inplace=True)

    train_x = x[:-16]
    val_x = x[-16:]

    train_y = y[:-16]
    val_y = y[-16:]

    lgbtrainset_area = lgb.Dataset(train_x, train_y)
    lgbvalidset_area = lgb.Dataset(val_x, val_y)

    lgb_model = run_lightgbm(95000, 9000, ['Longitude','Latitude', 'tod', 'dow', 'area', 'trend', 'is_holiday'], lgbtrainset_area, lgbvalidset_area)
    
    lgb_model.save_model('lgb_area'+target+'.pkl')


# ### Engineering Global level data

# In[ ]:


train_global = train.groupby('date').agg({'Available': 'sum',
                                                  'Charging': 'sum',
                                                  'Passive': 'sum',
                                                  'Other': 'sum',
                                                  'tod': 'max',
                                                  'dow': 'max',
                                                  'trend': 'max',}).reset_index()

test_global = test.groupby('date').agg({
    'tod': 'max',
    'dow': 'max',
    'trend': 'max'}).reset_index()


#Joining Test data to train set
train_global['status'] = 'train'
test_global['status'] = 'test'
test_global[['Available', 'Charging', 'Passive', 'Other']] = np.nan
alldata_global = train_global.append(test_global, ignore_index=True)

alldata_global['date'] = pd.to_datetime(alldata_global['date'])

#preprocessing
alldata_global['dates_only'] = alldata_global.date.dt.date
alldata_global['dates_only'] = pd.to_datetime(alldata_global['dates_only'])

alldata_global = pd.merge(alldata_global,df_holidays, how='left', on='dates_only')

alldata_global['is_holiday'].fillna(0, inplace=True)
alldata_global['is_weekend'] = alldata_global['dow'] >5

alldata_global.drop('dates_only',axis=1 ,inplace=True)
alldata_global['is_holiday'] = alldata_global['is_holiday'].astype('int')
alldata_global['is_weekend'] = alldata_global['is_weekend'].astype('int')

groupsAvail = alldata_global[['date','Available']].groupby(by='date').agg({'Available':'max'}).reset_index()
groupsCharge = alldata_global[['date','Charging']].groupby(by='date').agg({'Charging':'max'}).reset_index()
groupsPass = alldata_global[['date','Passive']].groupby(by='date').agg({'Passive':'max'}).reset_index()
groupsOth = alldata_global[['date','Other']].groupby(by='date').agg({'Other':'max'}).reset_index()

def genLags(df, grp, nlags):
    
    for i in range(1,nlags+1):
        df[grp.columns[1]+'_lag'+str(i)+''] = grp[grp.columns[1]].shift(i)
        
    return df
        
alldata_globalAvail = alldata_global.copy()
alldata_globalCharge = alldata_global.copy()
alldata_globalPass = alldata_global.copy()
alldata_globalOth = alldata_global.copy()

alldata_globalAvail = genLags(alldata_globalAvail, groupsAvail, 15)
alldata_globalCharge = genLags(alldata_globalCharge, groupsCharge, 15)
alldata_globalPass = genLags(alldata_globalPass, groupsPass, 15)
alldata_globalOth = genLags(alldata_globalOth, groupsOth, 15)

alldata_globalfinal = pd.concat([alldata_globalAvail,alldata_globalCharge[['Charging_lag1','Charging_lag2','Charging_lag3',
'Charging_lag4','Charging_lag5','Charging_lag6','Charging_lag7','Charging_lag8','Charging_lag9','Charging_lag10',
'Charging_lag11','Charging_lag12','Charging_lag13','Charging_lag14','Charging_lag15']],alldata_globalPass[['Passive_lag1','Passive_lag2','Passive_lag3',
'Passive_lag4','Passive_lag5','Passive_lag6','Passive_lag7','Passive_lag8','Passive_lag9','Passive_lag10',
'Passive_lag11','Passive_lag12','Passive_lag13','Passive_lag14','Passive_lag15']],alldata_globalOth[['Other_lag1','Other_lag2','Other_lag3','Other_lag4','Other_lag5','Other_lag6',
'Other_lag7','Other_lag8','Other_lag9','Other_lag10','Other_lag11','Other_lag12','Other_lag13','Other_lag14',
'Other_lag15']]], axis =1)

alldata_globalfinal


# In[ ]:


alldata_globalfinaltrain = alldata_globalfinal[alldata_globalfinal['status'] == 'train']
alldata_globalfinaltest = alldata_globalfinal[alldata_globalfinal['status'] == 'test']
alldata_globalfinaltest = alldata_globalfinaltest.drop(['Available','Charging','Passive','Other','status'],axis=1)


# ### Trend analysis on global level

# In[ ]:


##On global level
fig1 = px.line(alldata_globalfinaltrain, x = 'date', y = 'Available')
fig1.show()

fig1 = px.line(alldata_globalfinaltrain, x = 'date', y = 'Passive')
fig1.show()


# ### Modelling strategy for global level data

# In[ ]:


alldata_globalfinaltrain = label_encode(alldata_globalfinaltrain,'date')
alldata_globalfinaltest = label_encode(alldata_globalfinaltest,'date')

for target in ['Available','Charging','Passive','Other']:
    y = alldata_globalfinaltrain[target]
    x = alldata_globalfinaltrain.drop(['Available','Charging','Passive','Other'], axis =1)
    x = x.drop('status',axis=1)

    x.fillna(0,inplace=True)

    train_x = x[:-96]
    val_x = x[-96:]

    train_y = y[:-96]
    val_y = y[-96:]

    lgbtrainset_global = lgb.Dataset(train_x, train_y)
    lgbvalidset_global = lgb.Dataset(val_x, val_y)
    
    lgb_model = run_lightgbm(95000, 9000, ['tod', 'dow', 'trend', 'is_holiday'], lgbtrainset_global, lgbvalidset_global)

    lgb_model.save_model('lgb_global'+target+'.pkl')


# ### Publishing results on test set

# In[ ]:


stationpredicted = alldatafinaltest.copy()
areapredicted = alldata_areafinaltest.copy()
globalpredicted = alldata_globalfinaltest.copy()

#predict on station level
for target in ['Available','Charging','Passive','Other']:
    lgmodel = lgb.Booster(model_file='lgb_station'+target+'.pkl')
    stationpredicted[target] = lgmodel.predict(alldatafinaltest)

#predict on area level
for target in ['Available','Charging','Passive','Other']:
    lgmodel = lgb.Booster(model_file='lgb_area'+target+'.pkl')
    areapredicted[target] = lgmodel.predict(alldata_areafinaltest)
    
#predict on global level
for target in ['Available','Charging','Passive','Other']:
    lgmodel = lgb.Booster(model_file='lgb_global'+target+'.pkl')
    globalpredicted[target] = lgmodel.predict(alldata_globalfinaltest)

stationpredicted = stationpredicted[['date','Station','area','Available','Charging','Passive','Other']]
areapredicted = areapredicted[['date','area','Available','Charging','Passive','Other']]
globalpredicted = globalpredicted[['date','Available','Charging','Passive','Other']]


# In[ ]:


stationpredicted.to_csv('./sample_result_submission/station.csv', index=False)
areapredicted.to_csv('./sample_result_submission/area.csv', index=False)
globalpredicted.to_csv('./sample_result_submission/global.csv', index=False)


# In[ ]:


shutil.make_archive("sample_result_submission",
                        'zip', 'sample_result_submission')


