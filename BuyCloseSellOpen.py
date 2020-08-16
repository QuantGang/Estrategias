# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:49:25 2020

@author: Quintero
"""

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
tck_data = odata('TSLA', datetime(2020, 2, 19),datetime(2020, 4, 20))#YYYY/MM/DD
#2020,3,15
clspx = tck_data[['Adj Close']]
numest=2#num per -1
clspx = tck_data[['Close']]
clspx['Open']=tck_data[['Open']]
#señales de entrada
#variables
close_px = clspx['Close']
open_px= np.array(clspx['Open'])
#Buy close sell open
close_px_lastday=close_px.shift(1)
r_s_lastday= np.log(open_px)-np.log(close_px_lastday)
cum_ret = r_s_lastday.cumsum()
log_ret = np.log(close_px).diff()
#buy open sell close
r_s_day= np.log(open_px)-np.log(close_px)
cum_ret_day = r_s_day.cumsum()
## Calculate the cumulative log returns
cum_ret = r_s_lastday.cumsum()
nost_cum_ret = log_ret.cumsum()
cum_rel_ret = np.exp(cum_ret) - 1
cum_rel_day = np.exp(cum_ret_day) - 1
nost_cum_rel_ret = np.exp(nost_cum_ret) - 1

fig = plt.figure()
ax = fig.add_subplot(2,1,1)

ax.plot(cum_ret.index, cum_ret, label='buy close sell open')
ax.plot(cum_ret_day.index, cum_ret_day, label='buy open sell close')
ax.plot(nost_cum_ret.index, nost_cum_ret, label='buy and hold')
ax.set_ylabel('Cumulative log-returns')
ax.legend(loc='best')
ax.grid()

ax = fig.add_subplot(2,1,2)
ax.plot(cum_rel_ret.index, 100 * cum_rel_ret, label='buy close sell open')
ax.plot(cum_rel_ret.index, 100 * cum_rel_day, label='buy open sell close')
ax.plot(nost_cum_rel_ret.index, 100 * nost_cum_rel_ret, label='buy and hold')

ax.set_ylabel('Total relative returns (%)')
ax.legend(loc='best')
ax.grid()
plt.show()
