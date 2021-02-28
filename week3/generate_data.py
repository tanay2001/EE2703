import datetime
from datetime import date,timedelta
import nsepy
from nsepy import get_history
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from multiprocessing import Pool, Process
options_data=pd.DataFrame()

#Function generates possible expiry dates for the given date
def possible_expiry_date(date):
    dates=[]
    for i in expiry_dates:
        if i>date and (i-date).days<365:
            dates.append(i)
    return dates

#Reading the dates where the options will expire

file1 = open('/home/tanay/Downloads/dates.txt', 'r') 
dates=[]
for lines in file1:
    temp=lines.split()[2]
    if temp[0]=='"':
        dates.append(temp[1:-2])

expiry_dates=[date(int(i.split('-')[2]),int(i.split('-')[1]),int(i.split('-')[0])) for i in dates]
expiry_dates = sorted(expiry_dates,  key = lambda x: x.year)
expiry_dates=expiry_dates[3:]
expiry_dates=list(set(expiry_dates))

#Generating date range
date_range=[]
sdate = date(2015, 7, 24)   # start date
edate = date(2020, 10, 1)   # end date
delta = edate - sdate       # as timedelta
d={}
for i in expiry_dates:
    if i.year in d:
        d[i.year] +=1
    else:
        d[i.year] =1

for i in range(delta.days + 1):
    date_range.append(sdate + timedelta(days=i))   
#Getting data from NSE website
count=0
def f(x):
     return get_history(symbol="NIFTY",start=a,end=a,index=True,option_type='CE',expiry_date=x)

p = Process(target = f, args= ())

for a in tqdm(date_range):
    count+=1
    temp=pd.DataFrame()
    if a.weekday()==5 or a.weekday()==6:
        continue
    possible_expiry_dates=possible_expiry_date(a)
    y = time.time()
    for p in possible_expiry_dates:
        #nifty_opt = 
        ls = p.start()
        nifty_opt["DateOfTrade"]=a
        #print(nifty_opt)
        if ~temp.empty:
            temp=temp.append(nifty_opt)
    options_data=options_data.append(temp)
    print(time.time()-y)
    if count%100==0:
        options_data.to_pickle("options.pkl")
        with open("options_data.csv", 'a') as f:
            options_data.to_csv(f, header=f.tell()==0)
        options_data=pd.DataFrame()
options_data.to_csv("options_data.csv",index=False)