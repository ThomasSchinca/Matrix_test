# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 17:43:20 2023

@author: thoma
"""


import matrixprofile as mp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tslearn.matrix_profile import MatrixProfile
import matplotlib.transforms as mtransforms
from sklearn.preprocessing import MinMaxScaler 
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
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

df_tot_m=df_tot.resample('M').sum() 
df_tot_m=(df_tot_m-df_tot_m.mean())/df_tot_m.std()

# =============================================================================
# Monthly 
# =============================================================================

h_max=3
win=12

#### extract series
df_ind = df_tot_m.reset_index(drop=True)
seq=[]
for i in df_tot_m.columns:
    for j in range(len(df_ind.loc[:,i][df_ind.loc[:,i]==df_ind.loc[:,i].min()].index[:-1])):
        if df_ind.loc[:,i][df_ind.loc[:,i]==df_ind.loc[:,i].min()].index[j+1]-df_ind.loc[:,i][df_ind.loc[:,i]==df_ind.loc[:,i].min()].index[j] > 50:#h_max+win:
            seq.append(df_tot_m.loc[df_tot_m.loc[:,i][df_tot_m.loc[:,i]==df_ind.loc[:,i].min()].index[j]:df_tot_m.loc[:,i][df_tot_m.loc[:,i]==df_ind.loc[:,i].min()].index[j+1],i].iloc[1:])
n_test=[]
to=0
exclude=[]
for i in seq:
    n_test=np.concatenate([n_test,i])
    to=to+len(i)
    exclude=exclude+[*range(to-(h_max+win),to)]

              
##### Matrix profile analysis
glob_int=pd.DataFrame()
for rad in [0.5,0.7,1,1.2,1.4,1.6,1.8,2]:
    prof = mp.compute(n_test,windows=[12,200])
    profile = mp.discover.motifs(prof,k=50,max_neighbors=200,radius=rad)
    #mp.visualize(profile)
    
    l_mot=[]
    df_std=[]
    df_mean=[]
    for mot in range(len(profile['motifs'])):
        motifs = profile['motifs'][mot]['motifs']
        neigh = profile['motifs'][mot]['neighbors']
        comb=motifs+neigh
        l=[]
        for el in comb:
            if el[1] not in exclude:
                l.append(el[1])
        l_mot.append(l)        
        #plt.figure(figsize=(8,6))
        std=pd.DataFrame()
        for i in l:
        #    plt.plot([*range(12)],n_test[i:i+12],color='blue',alpha=0.05)
        #    plt.plot([*range(11,15)],n_test[i+11:i+15],color='red',alpha=0.05)
            n_t=n_test[i:i+15]
            std=pd.concat([std,pd.Series(n_t)],axis=1)
        #plt.plot([*range(12)],std.mean(axis=1)[:12],color='blue',linewidth=2)
        #plt.plot([*range(11,15)],std.mean(axis=1)[11:15],color='red',linewidth=2)
        #plt.fill_between([*range(12)],std.mean(axis=1)[:12]-std.std(axis=1).iloc[:12],std.mean(axis=1)[:12]+std.std(axis=1).iloc[:12],color='blue',alpha=0.1)      
        #plt.fill_between([*range(11,15)],std.mean(axis=1)[11:15]-std.std(axis=1).iloc[11:15],std.mean(axis=1)[11:15]+std.std(axis=1).iloc[11:15],color='red',alpha=0.1) 
        df_mean.append(std.mean(axis=1))  
        df_std.append(std.std(axis=1)) 
        #plt.title('Motif '+str(mot+1))
        #plt.show()
        
    df_std=pd.DataFrame(df_std)    
    df_mean=pd.DataFrame(df_mean) 
    
    len_c=[]
    for i in l_mot:
        len_c.append(len(i))
    # =============================================================================
    # Arima comparison
    # =============================================================================
    df_seq=pd.DataFrame()
    for j in seq:
        df_seq=pd.concat([df_seq,pd.DataFrame([j.values,j.index,[j.name]*len(j.index)])],axis=1)
    df_seq=df_seq.T    
    df_seq.index=range(len(df_seq))
    
    s_p = 1
    res_tot=[]
    for mot in l_mot:
        mot_d = df_seq.iloc[mot,:].sort_values([1])
        mot_d_train = mot_d[mot_d[1] <= mot_d[1].iloc[s_p-1] + timedelta(days=100)]    #for three months horizon 
        mot_d_test = mot_d[mot_d[1] > mot_d[1].iloc[s_p-1] + timedelta(days=100)]
        seq_train = pd.DataFrame()
        for j in mot_d_train.index:
            seq_train=pd.concat([seq_train,pd.Series(n_test[j+12:j+15])],axis=1)
        res=pd.DataFrame()    
        for i in range(len(mot_d_test)):
            train = df_tot_m.loc[:,mot_d_test.iloc[i,2]].loc[:mot_d_test.iloc[i,1]+ timedelta(days=365)]
            model = auto_arima(train)
            pred_ar = model.predict(3)
            pred_mot = seq_train.mean(axis=1)
            obs = n_test[mot_d_test.index[i]+12:mot_d_test.index[i]+15]
            res = pd.concat([res,pd.DataFrame([obs,pred_ar,pred_mot]).T])
            seq_train=pd.concat([seq_train,pd.Series(n_test[mot_d_test.index[i]+12:mot_d_test.index[i]+15])],axis=1)
        res_tot.append(res)
    
    df_err=pd.DataFrame()
    for i in range(len(res_tot)):
        try:
            err_ar_1 = abs(res_tot[i].loc[0,1]-res_tot[i].loc[0,0]).mean()
            err_mot_1 = abs(res_tot[i].loc[0,2]-res_tot[i].loc[0,0]).mean() 
            err_ar_2 = abs(res_tot[i].loc[1,1]-res_tot[i].loc[1,0]).mean()
            err_mot_2 = abs(res_tot[i].loc[1,2]-res_tot[i].loc[1,0]).mean()
            err_ar_3 = abs(res_tot[i].loc[2,1]-res_tot[i].loc[2,0]).mean()
            err_mot_3 = abs(res_tot[i].loc[2,2]-res_tot[i].loc[2,0]).mean()
            err_ar_tot = abs(res_tot[i].loc[:,1]-res_tot[i].loc[:,0]).mean()
            err_mot_tot = abs(res_tot[i].loc[:,2]-res_tot[i].loc[:,0]).mean()
            df_err = pd.concat([df_err,pd.DataFrame([err_ar_1,err_ar_2,err_ar_3,err_mot_1,err_mot_2,err_mot_3,err_ar_tot,err_mot_tot])],axis=1)
        except:
            df_err = pd.concat([df_err,pd.DataFrame([np.zeros((8))],index=range(8)).iloc[:,0]],axis=1)    
    
    df_err = df_err.T
    df_err.columns = ['AR-H1','AR-H2','AR-H3','MOT-H1','MOT-H2','MOT-H3','AR-TOT','MOT-TOT']
    df_err.index = range(0,50)
    
    df_err_tot = pd.DataFrame(df_err.iloc[:,6]-df_err.iloc[:,7],columns=['I_tot'])
    for i in range(3):
        df_err_tot=pd.concat([df_err_tot,pd.DataFrame(df_err.iloc[:,0+i]-df_err.iloc[:,3+i],columns=['I_h-'+str(i+1)])],axis=1)
    df_err_tot=pd.concat([df_err_tot,df_std.iloc[:,-3:],pd.Series(len_c,name='N_seq_inside'),pd.Series([*range(1,51)],name='N_motif'),pd.Series([rad]*50,name='rad')],axis=1)
    
    
    interest_df = df_err_tot[(df_err_tot.iloc[:,0]>0) & (df_err_tot.iloc[:,1]>0) & (df_err_tot.iloc[:,2]>0) & (df_err_tot.iloc[:,3]>0) & (df_err_tot.iloc[:,7]>5)]
    for mot in interest_df.iloc[:,8]:
        motifs = profile['motifs'][mot]['motifs']
        neigh = profile['motifs'][mot]['neighbors']
        comb=motifs+neigh
        l=[]
        for el in comb:
            if el[1] not in exclude:
                l.append(el[1])     
        plt.figure(figsize=(8,6))
        std=pd.DataFrame()
        for i in l:
            plt.plot([*range(12)],n_test[i:i+12],color='blue',alpha=0.05)
            plt.plot([*range(11,15)],n_test[i+11:i+15],color='red',alpha=0.05)
            n_t=n_test[i:i+15]
            std=pd.concat([std,pd.Series(n_t)],axis=1)
        plt.plot([*range(12)],std.mean(axis=1)[:12],color='blue',linewidth=2)
        plt.plot([*range(11,15)],std.mean(axis=1)[11:15],color='red',linewidth=2)
        plt.fill_between([*range(12)],std.mean(axis=1)[:12]-std.std(axis=1).iloc[:12],std.mean(axis=1)[:12]+std.std(axis=1).iloc[:12],color='blue',alpha=0.1)      
        plt.fill_between([*range(11,15)],std.mean(axis=1)[11:15]-std.std(axis=1).iloc[11:15],std.mean(axis=1)[11:15]+std.std(axis=1).iloc[11:15],color='red',alpha=0.1) 
        plt.title('Motif '+str(mot)+' with rad='+str(rad))
        plt.show()
    glob_int = pd.concat([glob_int,interest_df],axis=0)    