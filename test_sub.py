# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:22:50 2023

@author: thoma
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
### Data 
df_tot = pd.read_csv('C:/Users/thoma/Desktop/df_sub_m.csv',index_col=0,parse_dates=True)

### Matrix profile model
for file in ['Motif_1.csv','Motif_2.csv','Motif_3.csv']:
    df=pd.read_csv(file,index_col=0)
    
    df=df.iloc[:,1:]
    df_norm = (df-df.iloc[:-6,:].min())/(df.iloc[:-6,:].max()-df.iloc[:-6,:].min())
    df_pred = df_norm.mean(axis=1).iloc[-6:]
    
    k=[]
    for i in range(len(df_norm.columns)):
        df_wc = df_norm.drop(df_norm.columns[i], axis=1)
        df_pred = df_wc.mean(axis=1).iloc[-6:]
        df_pred = df_pred*(df.iloc[:-6,i].max()-df.iloc[:-6,i].min())+df.iloc[:-6,i].min()
        k.append(df_pred)
    k=pd.DataFrame(k)
    k=k.T
    k.columns = df_norm.columns
    k.loc[len(df.iloc[:-6,0])-1] = df.loc[len(df.iloc[:-6,0])-1]
    k=k.sort_index()
    
    
    pred_ar=[]
    ### Arima comparison 
    for i in df.columns:
        txt = i.split('-')[1]
        dat = pd.to_datetime(txt, format="%Y/%m")+pd.DateOffset(months=len(df.iloc[:-6,0]))
        ar_train = df_tot.loc[:dat,i.split('-')[0]]
        arima = auto_arima(ar_train)
        pred_ar.append(arima.predict(n_periods=6).reset_index(drop=True))
    pred_ar=pd.DataFrame(pred_ar)
    pred_ar=pred_ar.T 
    pred_ar.columns = df.columns
    pred_ar.index = df_pred.index   
    pred_ar.loc[len(df.iloc[:-6,0])-1] = df.loc[len(df.iloc[:-6,0])-1]
    pred_ar=pred_ar.sort_index()
    
    num_plots = len(df.columns)  # Number of subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # Adjust the figure size as needed
    
    # Flatten the axes array to simplify indexing
    axes = axes.flatten()
    mse_ar_t=0
    mse_mp_t=0
    for idx, i in enumerate(df.columns):
        mse_mp = mean_squared_error(df.loc[:, i].iloc[-6:],k.loc[:, i].iloc[1:])
        mse_ar = mean_squared_error(df.loc[:, i].iloc[-6:],pred_ar.loc[:, i].iloc[1:])
        mse_ar_t = mse_ar_t + mse_ar
        mse_mp_t =mse_mp_t + mse_mp
        ax = axes[idx]  # Select the current subplot
        ax.plot(df.loc[:, i].iloc[-7:], marker='o', color='r', label='Obs')
        ax.plot(k.loc[:, i], marker='o', color='green', label='Pred MP')
        ax.plot(pred_ar.loc[:, i], marker='o', color='orange', label='Pred Arima')
        ax.plot(df.loc[:, i].iloc[:-6], marker='o')
        ax.set_ylim([0.8*min(df.loc[:, i]), 1.2*max(df.loc[:, i])])
        if (mse_ar-mse_mp)/mse_ar*100 >0:
            ax.set_title(i+' MSE : - {0: .1f} %'.format(abs((mse_ar-mse_mp)/mse_ar*100)),color='green')
        else:
            ax.set_title(i+' MSE : + {0: .1f} %'.format(abs((mse_ar-mse_mp)/mse_ar*100)),color='red')    
        ax.legend()
    
    # Remove empty subplots, if any
    if len(df.columns) < len(axes):
        for idx in range(len(df.columns), len(axes)):
            fig.delaxes(axes[idx])
    
    plt.tight_layout()  # Adjust the spacing between subplots
    plt.suptitle(file.split('.')[0]+' / MSE : -{0: .1f} %'.format((mse_ar_t-mse_mp_t)/mse_ar_t*100))
    fig.subplots_adjust(top=0.9)
    plt.show()
