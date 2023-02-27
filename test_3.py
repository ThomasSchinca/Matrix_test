# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 18:36:52 2023

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
import bisect
from pmdarima.arima import auto_arima
from dateutil.relativedelta import relativedelta

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
df_tot=df_tot.resample('W').sum()   
df_tot_d= df_tot_m.diff()

h_max=5
win=12

#### extract series
df_ind = df_tot_d.reset_index(drop=True)
seq=[]
for i in range(len(df_tot_m.columns)): 
    if len(df_ind.iloc[:,i][df_ind.iloc[:,i]==0].index[:-1])!=0:
        for j in range(len(df_ind.iloc[:,i][df_ind.iloc[:,i]==0].index[:-1])):
            if df_ind.iloc[:,i][df_ind.iloc[:,i]==0].index[j+1]-df_ind.iloc[:,i][df_ind.iloc[:,i]==0].index[j] > 60:    #min 5 years
                seq.append(df_tot_m.iloc[df_ind.iloc[:,i][df_ind.iloc[:,i]==0].index[j]+1:df_ind.iloc[:,i][df_ind.iloc[:,i]==0].index[j+1],i])
    else : 
        seq.append(df_tot_m.iloc[:,i])

seq_n=[]
for i in seq:
    seq_n.append((i-i.mean())/i.std())

n_test=[]
to=0
exclude=[]
interv=[0]
for i in seq_n:
    n_test=np.concatenate([n_test,i])
    to=to+len(i)
    exclude=exclude+[*range(to-(h_max+24),to)]
    interv.append(to)

# =============================================================================
# General test       
# =============================================================================
res_tot=pd.DataFrame()
for rad in [1,1.2,1.4,1.6,1.8,2,2.5]:        
    for win in [12,14,16,18,10,22,24,30,36]:
        prof = mp.compute(n_test,windows=[win,200])
        profile = mp.discover.motifs(prof,k=20,max_neighbors=100,radius=rad)
        pmp= profile['pmp'][0,:]
        to=0
        exclude=[]
        for i in seq_n:
            to=to+len(i)
            exclude=exclude+[*range(to-(h_max+win),to)]
        
        for mot in range(len(profile['motifs'])):
            try:
                motifs = profile['motifs'][mot]['motifs']
                neigh = profile['motifs'][mot]['neighbors']
                comb=motifs+neigh
                ind_m=[]
                for i in comb:
                    if i[1] not in exclude:
                        ind_m.append(i[1])
                pmp_m = pmp[ind_m]   
             
                tab_mot = n_test[ind_m[0]:ind_m[0]+h_max+win].reshape((h_max+win,1))
                for i in ind_m[1:]:
                    try:
                        tab_mot=np.concatenate([tab_mot,n_test[i:i+h_max+win].reshape((h_max+win,1))],axis=1)
                    except:
                        1
                scaler = MinMaxScaler()
                
                tab_mot = scaler.fit_transform(tab_mot)
                tab_mot=pd.DataFrame(tab_mot)
                m_tab = tab_mot.mean(axis=1)
                std_tab = tab_mot.std(axis=1)
                '''
                plt.figure(figsize=(8,6))
                for i in range(len(tab_mot.columns)):
                    plt.plot(tab_mot.iloc[0:win,i],color='blue',alpha=0.1)
                    plt.plot(tab_mot.iloc[win-1:h_max+win,i],color='red',alpha=0.1)
                plt.plot(m_tab.iloc[0:win],color='blue',linewidth=2)    
                plt.plot(m_tab.iloc[win-1:h_max+win],color='red',linewidth=2) 
                plt.fill_between([*range(win)],m_tab.iloc[0:win]-std_tab.iloc[0:win],m_tab.iloc[0:win]+std_tab.iloc[0:win],color='blue',alpha=0.2)    
                plt.plot(m_tab.iloc[win-1:h_max+win],color='red') 
                plt.fill_between([*range(win-1,h_max+win)],m_tab.iloc[win-1:h_max+win]-std_tab.iloc[win-1:h_max+win],m_tab.iloc[win-1:h_max+win]+std_tab.iloc[win-1:h_max+win],color='red',alpha=0.2) 
                plt.title('Motifs '+str(mot+1)+' (CI +/- Std) - N = '+str(len(comb)))
                #plt.savefig('Figures_motifs_mean_std/Global Motif '+str(mot+1)+'.png')
                plt.show()
                '''
                #### ARIMA forecast comparison 
                res=pd.DataFrame()
                for i in ind_m:
                    try:
                        col = seq[bisect.bisect_right(interv, i)-1].name
                        index = seq[bisect.bisect_right(interv, i)-1].index[i-interv[bisect.bisect_right(interv, i)-1]+win-1]
                        index_obs = seq[bisect.bisect_right(interv, i)-1].index[i-interv[bisect.bisect_right(interv, i)-1]+win-1+h_max]
                        ar_train = df_tot_m.loc[:index,col]
                        arima = auto_arima(ar_train)
                        pred_ar = arima.predict(n_periods=h_max)  
                        
                        tab_mot = n_test[ind_m[0]:ind_m[0]+h_max+win].reshape((h_max+win,1))
                        n_mot=0
                        for k in ind_m[1:]:
                            try:
                                if i != k:
                                    tab_mot=np.concatenate([tab_mot,n_test[k:k+h_max+win].reshape((h_max+win,1))],axis=1)    
                                    n_mot=n_mot+1
                            except:
                                1
                        scaler = MinMaxScaler()
                        tab_mot = scaler.fit_transform(tab_mot)
                        tab_mot=pd.DataFrame(tab_mot)
                        m_tab = tab_mot.mean(axis=1)
                        std_tab = tab_mot.std(axis=1)
                        pred_mp =  m_tab.iloc[-h_max:]*(df_tot_m.loc[:index,col].iloc[-win:].max()/m_tab.iloc[:-h_max].max())
                        pred_mp.index= pred_ar.index
                        obs = df_tot_m.loc[:index_obs,col].iloc[-h_max+win:]
                        '''
                        plt.figure(figsize=(12,8))
                        plt.plot(obs,label='Observed value',marker='o')
                        plt.plot(pred_mp,label='MP predict',marker='o')
                        plt.plot(pred_ar,label='AR predict',marker='o')
                        plt.vlines(obs.index[win],0,df_tot_m.loc[:index_obs,col].iloc[-h_max+win:].max()+0.1*df_tot_m.loc[:index_obs,col].iloc[-h_max+win:].max(),colors='red',linestyles='--')
                        plt.legend()
                        plt.title(col)
                        plt.show()
                        '''
                        df_res=pd.DataFrame([obs.iloc[-h_max:].reset_index(drop=True),pred_ar.reset_index(drop=True),pred_mp.reset_index(drop=True)]).T
                        df_res.columns = ['obs','ar','mp']
                        res=pd.concat([res,df_res],axis=0)
                        
                    except:
                        1#print('This sequence is not doable : '+str(col)+'  '+str(index))
                '''
                plt.figure(figsize=(12,8))        
                for i in range(h_max):
                    k=res.loc[i,:]
                    plt.boxplot(abs(k.iloc[:,0]-k.iloc[:,1])-abs(k.iloc[:,0]-k.iloc[:,2]),positions=[i],notch=True, showfliers=False)
                plt.xticks([*range(h_max)],[*range(1,h_max+1)])
                plt.hlines(0, -0.25, h_max-0.75, colors='red',linestyles='--')
                plt.title('Motifs '+str(mot+1))
                '''
                res_l=[]
                for i in range(h_max):
                    k=res.loc[i,:]
                    res_l.append((abs(k.iloc[:,0]-k.iloc[:,1])-abs(k.iloc[:,0]-k.iloc[:,2])).median())
                res_l.append(rad) 
                res_l.append(win) 
                res_l.append(mot)
                res_l.append(n_mot)
                res=pd.DataFrame(res_l,index=['H1','H2','H3','H4','H5','rad','win','mot','n_seq']).T
                res_tot=pd.concat([res_tot,res])
            except:
                1
            
plt.scatter(res_tot.n_seq,res_tot.iloc[:,:5].mean(axis=1))
plt.ylim(-500,800)
plt.xlim(0,20)
plt.hlines(0,0,20,color='red')
plt.ylabel('MAE forecasting improv')
plt.xlabel('Number of seq in motif')
plt.show()

good = res_tot[(res_tot.H1>0) & (res_tot.H2>0) & (res_tot.H3>0) & (res_tot.H4>0) & (res_tot.H5>0) & (res_tot.n_seq>=12)]
good.index = range(len(good))
good_top= good.loc[good.iloc[:,:5].mean(axis=1).sort_values().iloc[-5:].index]


# =============================================================================
# The best motifs 
# =============================================================================

win_sel = [16,18,18]
mot_sel = [13,16,17]
rad_sel = [2,2,2]

for row in range(len(win_sel)):
    win=int(win_sel[row])
    h_max=5
    prof = mp.compute(n_test,windows=[win,200])
    profile = mp.discover.motifs(prof,k=20,max_neighbors=100,radius=int(rad_sel[row]))
    pmp= profile['pmp'][0,:]
    to=0
    exclude=[]
    for i in seq_n:
        to=to+len(i)
        exclude=exclude+[*range(to-win,to)]
    
    mot = int(mot_sel[row])
    motifs = profile['motifs'][mot]['motifs']
    neigh = profile['motifs'][mot]['neighbors']
    comb=motifs+neigh
    ind_m=[]
    for i in comb:
        if i[1] not in exclude:
            ind_m.append(i[1])
    pmp_m = pmp[ind_m]   
 
    tab_mot = n_test[ind_m[0]:ind_m[0]+h_max+win].reshape((h_max+win,1))
    for i in ind_m[1:]:
        try:
            tab_mot=np.concatenate([tab_mot,n_test[i:i+h_max+win].reshape((h_max+win,1))],axis=1)
        except:
            1
    scaler = MinMaxScaler()
    
    tab_mot = scaler.fit_transform(tab_mot)
    tab_mot=pd.DataFrame(tab_mot)
    m_tab = tab_mot.mean(axis=1)
    std_tab = tab_mot.std(axis=1)
    
    plt.figure(figsize=(8,6))
    for i in range(len(tab_mot.columns)):
        plt.plot(tab_mot.iloc[0:win,i],color='blue',alpha=0.1)
        plt.plot(tab_mot.iloc[win-1:h_max+win,i],color='red',alpha=0.1)
    plt.plot(m_tab.iloc[0:win],color='blue',linewidth=2)    
    plt.plot(m_tab.iloc[win-1:h_max+win],color='red',linewidth=2) 
    plt.fill_between([*range(win)],m_tab.iloc[0:win]-std_tab.iloc[0:win],m_tab.iloc[0:win]+std_tab.iloc[0:win],color='blue',alpha=0.2)    
    plt.plot(m_tab.iloc[win-1:h_max+win],color='red') 
    plt.fill_between([*range(win-1,h_max+win)],m_tab.iloc[win-1:h_max+win]-std_tab.iloc[win-1:h_max+win],m_tab.iloc[win-1:h_max+win]+std_tab.iloc[win-1:h_max+win],color='red',alpha=0.2) 
    plt.title('Motifs '+str(mot+1)+' (CI +/- Std)')
    #plt.savefig('Figures_motifs_mean_std/Global Motif '+str(mot+1)+'.png')
    plt.show()
    
    #### ARIMA forecast comparison 
    res=pd.DataFrame()
    if len(ind_m)%3==0:
        fig,ax = plt.subplots(len(ind_m)//3,3, figsize=(40,len(ind_m)//3*5))
    else:    
        fig,ax = plt.subplots((len(ind_m)//3+1),3, figsize=(40,(len(ind_m)//3+1)*5))
    c_l=0
    c_r=0
    for i in ind_m:
        try:
            col = seq[bisect.bisect_right(interv, i)-1].name
            index = seq[bisect.bisect_right(interv, i)-1].index[i-interv[bisect.bisect_right(interv, i)-1]+win-1]
            index_obs = seq[bisect.bisect_right(interv, i)-1].index[i-interv[bisect.bisect_right(interv, i)-1]+win-1+h_max]
            ar_train = df_tot_m.loc[:index,col]
            arima = auto_arima(ar_train)
            pred_ar = arima.predict(n_periods=h_max)  
            
            tab_mot = n_test[ind_m[0]:ind_m[0]+h_max+win].reshape((h_max+win,1))
            for k in ind_m[1:]:
                try:
                    if i != k:
                        tab_mot=np.concatenate([tab_mot,n_test[k:k+h_max+win].reshape((h_max+win,1))],axis=1)    
                except:
                    1
            scaler = MinMaxScaler()
            tab_mot = scaler.fit_transform(tab_mot)
            tab_mot=pd.DataFrame(tab_mot)
            m_tab = tab_mot.mean(axis=1)
            std_tab = tab_mot.std(axis=1)
            pred_mp =  m_tab.iloc[-h_max:]*(df_tot_m.loc[:index,col].iloc[-win:].max()/m_tab.iloc[:-h_max].max())
            pred_mp.index= pred_ar.index
            obs = df_tot_m.loc[:index_obs,col].iloc[-h_max-win:]
            
            pred_mp_p=pd.concat([obs.iloc[-h_max-1:-h_max],pred_mp])
            pred_ar_p=pd.concat([obs.iloc[-h_max-1:-h_max],pred_ar])
            
            
            ax[c_r,c_l].plot(obs,label='Observed value',marker='o')
            ax[c_r,c_l].plot(pred_mp_p,label='MP predict',marker='o')
            ax[c_r,c_l].plot(pred_ar_p,label='AR predict',marker='o')
            ax[c_r,c_l].vlines(obs.index[win-1],0,obs.max()+0.1*obs.max(),colors='red',linestyles='--')
            ax[c_r,c_l].legend()
            ax[c_r,c_l].set_title(col)
            
            df_res=pd.DataFrame([obs.iloc[-h_max:].reset_index(drop=True),pred_ar.reset_index(drop=True),pred_mp.reset_index(drop=True)]).T
            df_res.columns = ['obs','ar','mp']
            res=pd.concat([res,df_res],axis=0)
        except:
            if i%397+win+h_max<397:
                col = seq[bisect.bisect_right(interv, i)-1].name
                index_obs = seq[bisect.bisect_right(interv, i)-1].index[i-interv[bisect.bisect_right(interv, i)-1]]
                index_obs_ar = index_obs + relativedelta(months=+win)
                index_obs_2 = index_obs + relativedelta(months=+(win+h_max))
                obs = df_tot_m.loc[:index_obs_2,col].iloc[-h_max-win:]
                
                ar_train = df_tot_m.loc[:index_obs_ar,col]
                arima = auto_arima(ar_train)
                pred_ar = arima.predict(n_periods=h_max)  
                
                tab_mot = n_test[ind_m[0]:ind_m[0]+h_max+win].reshape((h_max+win,1))
                for k in ind_m[1:]:
                    try:
                        if i != k:
                            tab_mot=np.concatenate([tab_mot,n_test[k:k+h_max+win].reshape((h_max+win,1))],axis=1)    
                    except:
                        1
                scaler = MinMaxScaler()
                tab_mot = scaler.fit_transform(tab_mot)
                tab_mot=pd.DataFrame(tab_mot)
                m_tab = tab_mot.mean(axis=1)
                std_tab = tab_mot.std(axis=1)
                
                pred_mp =  m_tab.iloc[-h_max:]*(df_tot_m.loc[:index_obs_ar,col].iloc[-win:].max()/m_tab.iloc[:-h_max].max())
                pred_mp.index= pred_ar.index
                
                pred_mp_p=pd.concat([obs.iloc[-h_max-1:-h_max],pred_mp])
                pred_ar_p=pd.concat([obs.iloc[-h_max-1:-h_max],pred_ar])
                
                ax[c_r,c_l].plot(obs,label='Observed value',marker='o')     
                ax[c_r,c_l].plot(pred_mp_p,label='MP predict',marker='o')
                ax[c_r,c_l].plot(pred_ar_p,label='AR predict',marker='o')
                ax[c_r,c_l].vlines(obs.index[win],0,obs.max()+0.1*obs.max(),colors='red',linestyles='--')
                ax[c_r,c_l].legend()
                ax[c_r,c_l].set_title(col)
                
                df_res=pd.DataFrame([obs.iloc[-h_max:].reset_index(drop=True),pred_ar.reset_index(drop=True),pred_mp.reset_index(drop=True)]).T
                df_res.columns = ['obs','ar','mp']
                res=pd.concat([res,df_res],axis=0)
            else: 
                print('Overlaping seq')
        c_l=c_l+1
        if c_l==3:
            c_r=c_r+1
            c_l=0
    plt.show()  
      
    plt.figure(figsize=(12,8))        
    for i in range(h_max):
        k=res.loc[i,:]
        plt.boxplot(abs(k.iloc[:,0]-k.iloc[:,1])-abs(k.iloc[:,0]-k.iloc[:,2]),positions=[i], showfliers=False,showmeans=True)
    plt.xticks([*range(h_max)],[*range(1,h_max+1)])
    plt.hlines(0, -0.25, h_max-0.75, colors='red',linestyles='--')
    plt.title('Motifs '+str(mot+1)+' with N = '+str(len(k)))