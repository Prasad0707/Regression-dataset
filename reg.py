# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:13:23 2020

@author: Duddupudi
"""


import pandas
import numpy as np

data=pandas.read_csv(r"C:\Users\Duddupudi\Downloads\Metro_Interstate_Traffic_Volume.csv")
print(data.head(5))


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['holiday']=le.fit_transform(data['holiday'])     
data['weather_main']=le.fit_transform(data['weather_main'])   
data['weather_description']=le.fit_transform(data['weather_description'])   


data['date_time']=pd.to_datetime(data.date_time,errors='coerce')


data['year'] = data['date_time'].dt.year 
data['month'] = data['date_time'].dt.month 
data['day'] = data['date_time'].dt.day 
data['hour'] = data['date_time'].dt.hour 
data['minute'] = data['date_time'].dt.minute

data=data.dropna(axis=0)

x=data.iloc[:,[0,1,2,3,4,5,6,7,9,10,11,12,13]].values
y=data.iloc[:,[8]].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=9)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)






from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)*100))
