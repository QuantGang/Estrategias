# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 20:51:42 2020

@author: Quintero
"""

#Momentum with filter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data
from datetime import datetime
def odata(ticker, start, end):
     return data.DataReader(ticker, 'yahoo', start, end)
def oget(tickers, start, end):
    Alldata = []
    for t in tickers:
        rdata = odata(t,start,end)
        Alldata.append(rdata)
    return pd.concat(Alldata, keys=tickers, names=['Ticker','Date'])
tck_data = odata('TSLA', datetime(2020, 2, 14),datetime(2020, 4, 20))
clspx = tck_data[['Adj Close']]
l=len(clspx)
clspx['Close']=tck_data[['Close']]
clspx['MA21'] = clspx['Close'].rolling(window=21).mean()
clspx['MA100'] = clspx['Close'].rolling(window=100).mean()
clspx['MA200'] = clspx['Close'].rolling(window=200).mean()
clspx['EWMA9'] = clspx['Close'].ewm(ignore_na=False,span=9,min_periods=0,adjust=True).mean()
clspx['EWMA120'] = clspx['Close'].ewm(ignore_na=False,span=120,min_periods=0,adjust=True).mean()
clspx['EWMA21'] = clspx['Close'].ewm(ignore_na=False,span=21,min_periods=0,adjust=True).mean()
clspx['EWMA100'] = clspx['Close'].ewm(ignore_na=False,span=100,min_periods=0,adjust=True).mean()
clspx['EWMA200'] = clspx['Close'].ewm(ignore_na=False,span=200,min_periods=0,adjust=True).mean()
clspx['EWMA50'] = clspx['Close'].ewm(ignore_na=False,span=50,min_periods=0,adjust=True).mean()
clspx['Volume']= tck_data[['Volume']]
clspx['High']=tck_data[['High']]
clspx['Low']=tck_data[['Low']]
clspx['Open']=tck_data[['Open']]
#clspx.plot(figsize=(10, 5))
#para recortar los datos y ver el periodo cercano
#clspx=clspx.iloc[2700:]
#si lo quieren ver graficamente quiten el # de las siguientes 2 lineas
clspx[['Close', 'MA21', 'MA100', 'MA200']].plot(figsize=(10, 5))
#clspx[['Close', 'EWMA21', 'EWMA100', 'EWMA200']].plot(figsize=(10, 5))
plt.show()
#señales de entrada
#variables
close_px = np.array(clspx['Close'])
low_px= np.array(clspx['Low'])
high_px= np.array(clspx['High'])
open_px= np.array(clspx['Open'])
ewma_px= np.array(clspx['MA21'])
ewma100_px= np.array(clspx['MA100'])
ewma200_px= np.array(clspx['MA200'])
cloewma = np.greater(close_px, ewma_px)
clbewma=np.less(close_px,ewma_px)
e21o100=np.greater(ewma_px,ewma100_px)
e21o200=np.greater(ewma_px,ewma200_px)
e100o200=np.greater(ewma100_px,ewma200_px)
#precio rompe media de 21, y esta encima de la media de 100 y 200 usd, compro cierro a precio de cierre
#precio rompe media de 21, y esta debajo de la media de 100 y 200 usd, compro cierro a precio de cierre
signal=[0]*(l)
for i  in range(len((e100o200))):    
    if cloewma[i] == True and e21o100[i] == True and e21o100[i] == True and e100o200 [i] == True:
        signal[i]=1
    elif cloewma[i] == False and e21o100[i] == False and e21o100[i] == False and e100o200 [i] == False:
        signal[i]=-1
    else:
        signal[i]=0
signal=pd.DataFrame(signal)
signal=signal.set_index(clspx.index)
fig = plt.figure()
close_px1=clspx['Close']
log_ret = np.log(close_px1).diff()
log_ret=pd.DataFrame(log_ret)
log_ret=np.array(log_ret)
signal=np.array(signal)
r_s = signal * log_ret
## Calculate the cumulative log returns
r_s=pd.DataFrame(r_s)
signal=pd.DataFrame(signal)
log_ret=pd.DataFrame(log_ret)
#clspx.to_excel (r'C:\Users\Quintero\Desktop\TRADING STRATEGY´S\clspx20.xlsx', index = False, header=True)
cum_ret = r_s.cumsum()
nost_cum_ret = log_ret.cumsum()
# And relative returns
cum_rel_ret = np.exp(cum_ret) - 1
nost_cum_rel_ret = np.exp(nost_cum_ret) - 1

fig = plt.figure()
ax = fig.add_subplot(2,1,1)

ax.plot(cum_ret.index, cum_ret, label='Momentum 21-100-200')
ax.plot(nost_cum_ret.index, nost_cum_ret, label='buy and hold')

ax.set_ylabel('Cumulative log-returns')
ax.legend(loc='best')
ax.grid()

ax = fig.add_subplot(2,1,2)
ax.plot(cum_rel_ret.index, 100*cum_rel_ret, label='Momentum 21-100-200')
ax.plot(nost_cum_rel_ret.index, 100*nost_cum_rel_ret, label='buy and hold')

ax.set_ylabel('Total relative returns (%)')
ax.legend(loc='best')
ax.grid()
plt.show()