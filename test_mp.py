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

prof = mp.compute(np.array(df_tot_m['Sudan']),windows=[12])
profile = mp.discover.motifs(prof,k=1,radius=10)
mp.visualize(profile)

# With NA
test = np.concatenate([np.array(df_tot_m['Sudan']),np.array([float('NaN')]),np.array(df_tot_m['Afghanistan'])])
plt.plot(test)
prof = mp.compute(test)
profile = mp.discover.motifs(prof,k=1)
mp.visualize(profile)

# Without NA
test = np.concatenate([np.array(df_tot_m['Sudan']),np.array(df_tot_m['Afghanistan'])])
prof = mp.compute(test,windows=[12,200])
profile = mp.discover.motifs(prof,k=5)
mp.visualize(profile)

test_2=df_tot_m[df_tot_m!=0].dropna(axis=1,thresh=350)
test_2[test_2.isna()]=0
#n_test=(test_2-test_2.min())/(test_2.max()-test_2.min())
n_test=(test_2-test_2.mean())/test_2.std()
n_test=pd.DataFrame(n_test)
n_test=np.array(n_test).reshape((n_test.shape[0]*n_test.shape[1],))
prof = mp.compute(n_test,windows=[12,200])
mp.visualize(prof)
profile = mp.discover.motifs(prof,k=3)
h=mp.visualize(profile)

pmp= profile['pmp'][0,:]
