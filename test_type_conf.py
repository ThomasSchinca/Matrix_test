# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:35:05 2023

@author: thoma
"""

import pandas as pd 
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from dtaidistance import ed
import random 
from sklearn.metrics import mean_squared_error

def int_exc(win_l,seq_n):
    n_test=[]
    to=0
    exclude=[]
    interv=[0]
    for i in seq_n:
        n_test=np.concatenate([n_test,i])
        to=to+len(i)
        exclude=exclude+[*range(to-win_l,to)]
        interv.append(to)
    return exclude,interv,n_test

df = pd.read_csv("https://ucdp.uu.se/downloads/ged/ged221-csv.zip",
                 parse_dates=['date_start',
                              'date_end'],
                 low_memory=False)
df_l=[]
df_l_w=[]
for num in df.type_of_violence.unique():
    df_tot = pd.DataFrame(columns=df.country.unique(),
                          index=pd.date_range(df.date_start.min(),
                                              df.date_end.max()))
    df_tot=df_tot.fillna(0)
    
    for i in df.country.unique():
        df_sub=df[(df.country==i) & (df.type_of_violence==num)]
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
    df_l_w.append(df_tot.resample('W').sum())
    df_l.append(df_tot.resample('M').sum())



seq_tot=[]
seq_t=[]
win_l=12
for df_confl in df_l:
    df_tot_m=df_confl
    seq=[]
    for i in range(len(df_tot_m.columns)): 
        seq.append(df_tot_m.iloc[:,i])
    seq_t=seq_t+seq
    seq_tot.append(seq)
    
exclude,interv,n_test = int_exc(win_l,seq_t)
l=random.sample(range(len(n_test[:-win_l])), len(n_test[:-win_l]))    

c_tot=[]
for cou in range(10000,len(l)):
    i=l[cou]
    if i not in exclude: 
        seq1 = np.array(n_test[i:i+win_l])
        if seq1.sum() != 0:
            if seq1.var()!=0.0:
                seq1 = (seq1 - seq1.min())/(seq1.max() - seq1.min())
            c_th=[]
            for st in seq_tot:
                num=0
                exclude_sub,interv,n_test_sub = int_exc(win_l,st)
                for j in range(len(n_test_sub[:-win_l])):
                    if j not in exclude_sub:
                        seq2 = n_test_sub[j:j+win_l]
                        seq2 = seq2 = (seq2 - seq2.min())/(seq2.max() - seq2.min())       
                        dist = ed.distance(seq1,seq2)
                        if dist<0.5:        
                            num = num+1
                c_th.append(num)
            c_tot.append([c_th,seq1])
        
c_tot_df = np.array(c_tot)
rep = pd.DataFrame(c_tot_df[:,0].tolist())

df_1 = rep[(rep[0]>8) & ((rep[1]+rep[2])<3)]
df_1=df_1.sort_values([0],ascending=False)
df_2 = rep[(rep[1]>5) & ((rep[0]+rep[2])<3)]
df_2=df_2.sort_values([1],ascending=False)
df_3 = rep[(3>(rep[0]+rep[1])) & (rep[2] >4)]
df_3=df_3.sort_values([2],ascending=False)    

for i in df_1.index[:1]:
    plt.plot(c_tot_df[i,1],marker='o')
    plt.title('Shape State-based conflict')
    plt.show()
    
for i in df_2.index[:1]:
    plt.plot(c_tot_df[i,1],marker='o')
    plt.title('Shape One-sided violence')
    plt.show()
    
for i in df_3.index:
    plt.plot(c_tot_df[i,1],marker='o')
    plt.title('Shape Non-state conflict')
    plt.show()
    
    
names=['State-Based conflict','One-sided conflict','Non-State conflict']
    
### Data 
l_mot = ['State_R.csv','One_R.csv','Non_R.csv']
l_other = [['State_O.csv'],['One_S.csv'],['Non_S.csv','Non_O.csv']]
### Matrix profile model
for k in range(3):
    file = l_mot[k]
    df=pd.read_csv(file,index_col=0)
    df=df.iloc[:,1:]
    len_df=len(df.columns)
    
    for side in range(len(l_other[k])):
        df_s=pd.read_csv(l_other[k][side],index_col=0)
        df_s=df_s.iloc[:,1:]
        for col in df_s:
            df_s=df_s.rename(columns={col:col+'~'+l_other[k][side].split('.')[0][-1]})
        df=pd.concat([df,df_s],axis=1)
    
    num_plots = len(df.columns)  # Number of subplots
    
    fig, axes = plt.subplots(int(len(df.columns)/3+1), 3, figsize=(15, 6+2*int(len(df.columns)/3+1)))  # Adjust the figure size as needed
    
    # Flatten the axes array to simplify indexing
    axes = axes.flatten()
    for idx, i in enumerate(df.columns):
        ax = axes[idx]  # Select the current subplot
        ax.plot(df.loc[:, i].iloc[:-6], marker='o')
        ax.set_ylim([0, 1.2*max(df.loc[:, i].iloc[:-6])])
        
        if idx <= (len_df-1):
            ax.set_title(i,color='green')
        else:
            if i.split('~')[1] == 'O':
                ax.set_title(i.split('~')[0]+' One-sided',color='red')    
            elif i.split('~')[1] == 'S':
                    ax.set_title(i.split('~')[0]+' State-based',color='red')   
            elif i.split('~')[1] == 'N':
                ax.set_title(i.split('~')[0]+' Non-State',color='red')            
    
    # Remove empty subplots, if any
    if len(df.columns) < len(axes):
        for idx in range(len(df.columns), len(axes)):
            fig.delaxes(axes[idx])
    
    plt.tight_layout()  # Adjust the spacing between subplots
    plt.suptitle('Most representative pattern of '+names[k],fontsize=15)
    fig.subplots_adjust(top=0.92)
    plt.show()