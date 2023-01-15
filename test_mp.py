# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 16:46:07 2022

@author: thoma
"""

import matrixprofile as mp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tslearn.matrix_profile import MatrixProfile
import matplotlib.transforms as mtransforms
#df = pd.read_stata('C:/Users/thoma/Downloads/S0020818316000059sup001.dta')
#df1 = pd.read_stata('C:/Users/thoma/Downloads/S0020818316000059sup002.dta')
#df2 = pd.read_stata('C:/Users/thoma/Downloads/S0020818316000059sup003.dta')

#for i in range(len(df['warname'].unique())):
#    l = df[df['warname']==df['warname'].unique()[i]]['lndeadpop']
#    plt.plot(np.exp(l))
#    plt.title(df['warname'].unique()[i])
#    plt.show()
    
 
df = pd.read_csv("https://ucdp.uu.se/downloads/ged/ged221-csv.zip",
                 parse_dates=['date_start',
                              'date_end'],
                 low_memory=False)
df_tot = pd.DataFrame(columns=df.country.unique(),
                      index=pd.date_range(df.date_start.min(),
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


test_2=df_tot_m[df_tot_m!=0].dropna(axis=1,thresh=350)
test_2[test_2.isna()]=0
n_test=(test_2-test_2.mean())/test_2.std()
n_test=pd.DataFrame(n_test)
n_test=np.array(n_test).reshape((n_test.shape[0]*n_test.shape[1],))
prof = mp.compute(n_test,windows=[12,200])
profile = mp.discover.motifs(prof,k=3)
mp.visualize(profile)

plt.figure(figsize=(20,5))
plt.plot(n_test)
plt.title('Normalized Monthly Fatalities TS')
plt.xticks([*range(175, 4150, 400)],test_2.columns)
for i in range(9):
    plt.vlines(397*(i+1),-2,17,linestyles='--',color='black')
plt.ylim(-2,17)    
plt.show()

pmp= profile['pmp'][0,:]
plt.figure(figsize=(20,5))
plt.plot(prof['pmp'][0,:])
plt.xticks([*range(175, 4150, 400)],test_2.columns)
for i in range(9):
    plt.vlines(397*(i+1),0.25,2.5,linestyles='--',color='black')
plt.ylim(0.25,2.5)    
plt.title('Matrix Profile - Window Length = 12 months')
plt.show()

#### Catching motifs 
motifs = prof['motifs'][2]['motifs']
neigh = prof['motifs'][2]['neighbors']
comb=motifs+neigh
ind_m=[]
for i in comb:
    ind_m.append(i[1])
pmp_m = pmp[ind_m]   
 
plt.figure(figsize=(20,5))
plt.plot(prof['pmp'][0,:])
plt.xticks([*range(175, 4150, 400)],test_2.columns)
for i in range(9):
    plt.vlines(397*(i+1),0.25,2.5,linestyles='--',color='black')
plt.ylim(0.25,2.5)  
plt.plot(ind_m,pmp_m,marker='*',color='r',linewidth=0,markersize=15)  
plt.title('Matrix Profile - Motif 3')
plt.show()


plt.figure(figsize=(20,5))
plt.plot(n_test)
plt.title('Normalized Monthly Fatalities TS')
plt.xticks([*range(175, 4150, 400)],test_2.columns)
for i in range(9):
    plt.vlines(397*(i+1),-2,17,linestyles='--',color='black')
for i in ind_m:
    plt.plot([*range(i,i+12)],n_test[i:i+12],color='red')
plt.ylim(-2,17)    
plt.show()

str_motifs=['Colombia 1','Colombia 2','Colombia 3',
            'DR Congo 1','DR Congo 2','Etiopia 1',
            'Etiopia 2','Etiopia 3','Etiopia 4','India 1',
            'Sudan 1','Sudan 2']
            
fig, axs = plt.subplots(4, 3,figsize=(20,15))
r=0
c=0
l=0
for i in ind_m:
    axs[c, r].plot([*range(i,i+12)],n_test[i:i+12],color='red')
    axs[c, r].set_title(str_motifs[l])
    r=r+1
    l=l+1
    if r==3:
        r=0
        c=c+1
        
fig.tight_layout()        
plt.show()        
