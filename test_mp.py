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
from sklearn.preprocessing import MinMaxScaler 
 
### Data Preparation 

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

##### Matrix profile analysis

prof = mp.compute(n_test,windows=[12,200])
profile = mp.discover.motifs(prof,k=15,max_neighbors=18)
profile = mp.discover.motifs(prof,k=15,max_neighbors=18,radius=3)
mp.visualize(profile)

plt.figure(figsize=(20,5))
plt.plot(n_test)
plt.title('Normalized Monthly Fatalities TS')
plt.xticks([*range(175, 4150, 400)],test_2.columns)
for i in range(9):
    plt.vlines(397*(i+1),-2,17,linestyles='--',color='black')
plt.ylim(-2,17)    
#plt.savefig('Figures_Matrix_analysis/Normalized_Monthly_Fatalities_TS.png')
plt.show()

pmp= profile['pmp'][0,:]
plt.figure(figsize=(20,5))
plt.plot(prof['pmp'][0,:])
plt.xticks([*range(175, 4150, 400)],test_2.columns)
for i in range(9):
    plt.vlines(397*(i+1),0.25,2.5,linestyles='--',color='black')
plt.ylim(0.25,2.5)    
plt.title('Matrix Profile - Window Length = 12 months')
#plt.savefig('Figures_Matrix_analysis/MP_12m.png')
plt.show()

#### Catching motifs 3
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
#plt.savefig('Figures_Matrix_analysis/MP_motif_3.png')
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
#plt.savefig('Figures_Matrix_analysis/TS_motif_3.png')
plt.show()
    

###### All Motifs
for mot in range(len(prof['motifs'])):
    motifs = prof['motifs'][mot]['motifs']
    neigh = prof['motifs'][mot]['neighbors']
    comb=motifs+neigh
    ind_m=[]
    for i in comb:
        ind_m.append(i[1])
    pmp_m = pmp[ind_m]   
    fig, axs = plt.subplots((len(comb)//3)+1, 3,figsize=(20,15))
    fig.suptitle('Motifs '+str(mot+1)+' (blue) and following values (red)',fontsize=20)
    r=0
    c=0
    l=0
    for i in ind_m:
        try:
            axs[c, r].plot(test_2.index[(i%397)+11:(i%397)+15],n_test[i+11:i+15],color='red',marker='o')
            axs[c, r].plot(test_2.index[i%397:(i%397)+12],n_test[i:i+12],color='blue')
            axs[c, r].set_title(test_2.columns[i//397])
        except:
            r=r-1
            l=l-1
        r=r+1
        l=l+1
        if r==3:
            r=0
            c=c+1    
    fig.tight_layout()        
    #plt.savefig('Figures_specif_motifs/Motif '+str(mot+1)+'.png')
    plt.show()   
    
    tab_mot = n_test[ind_m[0]:ind_m[0]+15].reshape((15,1))
    for i in ind_m[1:]:
        if len(test_2.index[(i%397)+11:(i%397)+15])>2:
            tab_mot=np.concatenate([tab_mot,n_test[i:i+15].reshape((15,1))],axis=1)
    scaler = MinMaxScaler()
    tab_mot = scaler.fit_transform(tab_mot)
    tab_mot=pd.DataFrame(tab_mot)
    m_tab = tab_mot.mean(axis=1)
    std_tab = tab_mot.std(axis=1)
    
    plt.figure(figsize=(8,6))
    for i in range(len(tab_mot.columns)):
        plt.plot(tab_mot.iloc[0:12,i],color='blue',alpha=0.1)
        plt.plot(tab_mot.iloc[11:15,i],color='red',alpha=0.1)
    plt.plot(m_tab.iloc[0:12],color='blue',linewidth=2)    
    plt.plot(m_tab.iloc[11:15],color='red',linewidth=2) 
    plt.fill_between([*range(12)],m_tab.iloc[0:12]-std_tab.iloc[0:12],m_tab.iloc[0:12]+std_tab.iloc[0:12],color='blue',alpha=0.2)    
    plt.plot(m_tab.iloc[11:15],color='red') 
    plt.fill_between([*range(11,15)],m_tab.iloc[11:15]-std_tab.iloc[11:15],m_tab.iloc[11:15]+std_tab.iloc[11:15],color='red',alpha=0.2) 
    plt.title('Motifs '+str(mot+1)+' (CI +/- Std) - N = '+str(len(comb)))
    #plt.savefig('Figures_motifs_mean_std/Global Motif '+str(mot+1)+'.png')
    plt.show()
    
    