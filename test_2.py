#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 17:13:29 2023

@author: pace
"""

import matrixprofile as mp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tslearn.matrix_profile import MatrixProfile
import matplotlib.transforms as mtransforms
from sklearn.preprocessing import MinMaxScaler 
 
### Data Preparation 

df = pd.read_csv("https://ucdp.uu.se/downloads/ged/ged221-csv.zip",
                 parse_dates=['date_start','date_end'],low_memory=False)
df_tot = pd.DataFrame(columns=df.country.unique(),index=pd.date_range(df.date_start.min(),
                                          df.date_end.max()))
df_tot=df_tot.fillna(0)
for i in df.country.unique():
    df_sub=df[df.country==i]
    for j in range(len(df_sub)):
        if df_sub.date_start.iloc[j] == df_sub.date_end.iloc[j]:
            df_tot.loc[df_sub.date_start.iloc[j],i]=df_tot.loc[df_sub.date_start.iloc[j],i]+df_sub.best.iloc[j]
        else:
            df_tot.loc[df_sub.date_start.iloc[j]:
            df_sub.date_end.iloc[j],i]=df_tot.loc[df_sub.date_start.iloc[j]: \
                                                  df_sub.date_end.iloc[j],i]+ \
                                                  df_sub.best.iloc[j]/ \
                                                  (df_sub.date_end.iloc[j]- \
                                                  df_sub.date_start.iloc[j]).days 
df_tot=df_tot.resample('W').sum() 
df_tot_m=df_tot.resample('M').sum() 


# =============================================================================
# Monthly 
# =============================================================================

h_max=3
win=12
#### extract all country 
test_2=df_tot_m[df_tot_m!=0].dropna(axis=1,thresh=390)
test_2[test_2.isna()]=0
n_test=(test_2-test_2.mean())/test_2.std()
n_test=np.array(n_test).T.reshape((n_test.shape[0]*n_test.shape[1],))
exclude=[]
for i in range(1,len(test_2.columns)+1):
    exclude=exclude+[*range(i*(len(test_2))-(h_max+win),i*(len(test_2)))]

#### extract series
df_ind = df_tot_m.reset_index(drop=True)
seq=[]
for i in df_tot_m.columns:
    for j in range(len(df_ind.loc[:,i][df_ind.loc[:,i]==0].index[:-1])):
        if df_ind.loc[:,i][df_ind.loc[:,i]==0].index[j+1]-df_ind.loc[:,i][df_ind.loc[:,i]==0].index[j] > 50:#h_max+win:
            seq.append(df_tot_m.loc[df_tot_m.loc[:,i][df_tot_m.loc[:,i]==0].index[j]:df_tot_m.loc[:,i][df_tot_m.loc[:,i]==0].index[j+1],i].iloc[1:])
n_test=[]
to=0
exclude=[]
for i in seq:
    #i=(i-i.min())/(i.max()-i.min())
    i=(i-i.mean())/i.std()
    n_test=np.concatenate([n_test,i])
    to=to+len(i)
    exclude=exclude+[*range(to-(h_max+win),to)]

              
##### Matrix profile analysis

prof = mp.compute(n_test,windows=[12,200])
profile = mp.discover.motifs(prof,k=15,max_neighbors=200,radius=5)
#mp.visualize(profile)

l_mot=[]
df_std=[]
df_mean=[]
for mot in range(len(profile['motifs'])):
    motifs = profile['motifs'][mot]['motifs']
    neigh = profile['motifs'][mot]['neighbors']
    comb=motifs+neigh
    if len(comb)>5:
        l=[]
        for el in comb:
            if el[1] not in exclude:
                l.append(el[1])
        l_mot.append(l)        
        plt.figure(figsize=(8,6))
        std=pd.DataFrame()
        for i in l:
            plt.plot([*range(12)],n_test[i:i+12],color='blue',alpha=0.1)
            plt.plot([*range(11,15)],n_test[i+11:i+15],color='red',alpha=0.1)
            #n_t=(n_test[i:i+15]-n_test[i:i+15].min())/(n_test[i:i+15].max()-n_test[i:i+15].min())
            n_t=n_test[i:i+15]
            std=pd.concat([std,pd.Series(n_t)],axis=1)
        plt.plot([*range(12)],std.mean(axis=1)[:12],color='blue')
        plt.plot([*range(11,15)],std.mean(axis=1)[11:15],color='red')
        plt.fill_between([*range(12)],std.mean(axis=1)[:12]-std.std(axis=1).iloc[:12],std.mean(axis=1)[:12]+std.std(axis=1).iloc[:12],color='blue',alpha=0.2)      
        plt.fill_between([*range(11,15)],std.mean(axis=1)[11:15]-std.std(axis=1).iloc[11:15],std.mean(axis=1)[11:15]+std.std(axis=1).iloc[11:15],color='red',alpha=0.2) 
        df_mean.append(std.mean(axis=1))  
        df_std.append(std.std(axis=1)) 
        plt.title('Motif '+str(mot))
        plt.show()
        
df_std=pd.DataFrame(df_std)    
df_mean=pd.DataFrame(df_mean) 


# =============================================================================
# Weekly
# =============================================================================

h_max=10
win=52

#### extract series
df_ind = df_tot.reset_index(drop=True)
seq=[]
for i in df_tot.columns:
    for j in range(len(df_ind.loc[:,i][df_ind.loc[:,i]==0].index[:-1])):
        if df_ind.loc[:,i][df_ind.loc[:,i]==0].index[j+1]-df_ind.loc[:,i][df_ind.loc[:,i]==0].index[j] > 50:#h_max+win:
            seq.append(df_tot.loc[df_tot.loc[:,i][df_tot.loc[:,i]==0].index[j]:df_tot.loc[:,i][df_tot.loc[:,i]==0].index[j+1],i].iloc[1:])
n_test=[]
to=0
exclude=[]
for i in seq:
    i=(i-i.min())/(i.max()-i.min())
    #i=(i-i.mean())/i.std()
    n_test=np.concatenate([n_test,i])
    to=to+len(i)
    exclude=exclude+[*range(to-(h_max+win),to)]


##### Matrix profile analysis

prof = mp.compute(n_test,windows=[52,200])
profile = mp.discover.motifs(prof,k=15,max_neighbors=200,radius=3)
#mp.visualize(profile)

l_mot=[]
df_std=[]
df_mean=[]
for mot in range(len(profile['motifs'])):
    motifs = profile['motifs'][mot]['motifs']
    neigh = profile['motifs'][mot]['neighbors']
    comb=motifs+neigh
    if len(comb)>5:
        l=[]
        for el in comb:
            if el[1] not in exclude:
                l.append(el[1])
        l_mot.append(l)        
        plt.figure(figsize=(8,6))
        std=pd.DataFrame()
        for i in l:
            plt.plot([*range(52)],n_test[i:i+52],color='blue',alpha=0.1)
            plt.plot([*range(51,61)],n_test[i+51:i+61],color='red',alpha=0.1)
            #n_t=(n_test[i:i+15]-n_test[i:i+15].min())/(n_test[i:i+15].max()-n_test[i:i+15].min())
            n_t=n_test[i:i+61]
            std=pd.concat([std,pd.Series(n_t)],axis=1)
        plt.plot([*range(52)],std.mean(axis=1)[:52],color='blue')
        plt.plot([*range(51,61)],std.mean(axis=1)[51:61],color='red')
        plt.fill_between([*range(52)],std.mean(axis=1)[:52]-std.std(axis=1).iloc[:52],std.mean(axis=1)[:52]+std.std(axis=1).iloc[:52],color='blue',alpha=0.2)      
        plt.fill_between([*range(51,61)],std.mean(axis=1)[51:61]-std.std(axis=1).iloc[51:61],std.mean(axis=1)[51:61]+std.std(axis=1).iloc[51:61],color='red',alpha=0.2) 
        df_mean.append(std.mean(axis=1))  
        df_std.append(std.std(axis=1)) 
        plt.title('Motif '+str(mot))
        plt.show()
        
df_std=pd.DataFrame(df_std)    
df_mean=pd.DataFrame(df_mean) 

