import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from math import floor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier

#open train & test file
file = "Train.csv"
df = pd.read_csv(file)
df2 = pd.read_csv("Test.csv")
#features to consider 
cols = ["AC", "DATOP", "DEPSTN", "ARRSTN", "STATUS", "STD", "STA"]

labels = df["target"]


#encode strings
l_ac = preprocessing.LabelEncoder()
ALL = pd.concat([df["AC"], df2["AC"]], ignore_index=True)
l_ac.fit(ALL)

l_stat = preprocessing.LabelEncoder()
l_stat.fit(df["STATUS"])

l_dep = preprocessing.LabelEncoder()
country = pd.concat([df["DEPSTN"], df["ARRSTN"], df2["DEPSTN"], df2["ARRSTN"]], ignore_index=True)
l_dep.fit(country)

l_id = preprocessing.LabelEncoder()
ID = pd.concat([df["FLTID"], df2["FLTID"]], ignore_index=True)
l_id.fit(ID)

def process_data(df):
    features = df[cols]
    #Label Encoding
    features["AC"] = l_ac.transform(features["AC"])
    features["STATUS"] = l_stat.transform(features["STATUS"])
    features["DEPSTN"] = l_dep.transform(features["DEPSTN"])
    features["ARRSTN"] = l_dep.transform(features["ARRSTN"])

    #Encode Date
    def encode_date(date):
        a = list(date.split('-'))
        return (int(a[1]) - 1) * 31 + int(a[2])
    def encode_date_year(date):
        a = list(date.split('-'))
        return int(a[0]) - 2015
    
    features['day_op'] = features['DATOP'].map(encode_date)
    features["year_op"] = features["DATOP"].map(encode_date_year)

    def encode_time_std_year(date):
        A = list(date.split(' '))
        a = list(A[0].split('-'))
        b = list(A[1].split(':'))
        return int(a[0]) - 2015
    
    def encode_time_std_day(date):
        A = list(date.split(' '))
        a = list(A[0].split('-'))
        b = list(A[1].split(':'))
        return (int(a[1]) - 1) * 31 + int(a[2])
    
    def encode_time_std_time(date):
        A = list(date.split(' '))
        a = list(A[0].split('-'))
        b = list(A[1].split(':'))
        return int(b[0]) * 60 + int(b[1])
    
    def encode_time_sta_year(date):
        A = list(date.split(' '))
        a = list(A[0].split('-'))
        b = list(A[1].split('.'))
        return int(a[0]) - 2015
    
    def encode_time_sta_day(date):
        A = list(date.split(' '))
        a = list(A[0].split('-'))
        b = list(A[1].split('.'))
        return (int(a[1]) - 1) * 31 + int(a[2])
    
    def encode_time_sta_time(date):
        A = list(date.split(' '))
        a = list(A[0].split('-'))
        b = list(A[1].split('.'))
        return int(b[0]) * 60 + int(b[1])
    
    features["a3"] = features["STD"].map(encode_time_std_time)
    features["b3"] = features["STA"].map(encode_time_sta_time)
    features.drop(columns =["STD", "STA", "DATOP"], inplace = True) 
    #scale data
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)

    return features

ftrs = process_data(df)

#split data
x_train, x_test, y_train, y_test = train_test_split(ftrs, labels, test_size = 0.1)
clf = LinearRegression()

#other algorithms
#clf = svm.SVR(kernel = "rbf", gamma = 10.0, C = 100.0)
#clf = GaussianNB()
#clf  = RandomForestRegressor(max_depth=3, random_state=42, n_estimators=10000, min_samples_split = 150)
#clf = svm.SVR(kernel = 'linear')
#clf = linear_model.SGDRegressor(max_iter = 2000,eta0=0.00001, power_t=0.25)

clf = clf.fit(x_train, y_train)

X = df2[cols]
X["DATOP"] = X["DATOP"].map(str)
X = process_data(df2)
ans = clf.predict(X)
df2.drop(columns =['DATOP', 'FLTID', 'DEPSTN', 'ARRSTN', 'STD', 'STA', 'STATUS', 'AC'], inplace = True) 
df2["target"] = ans
df2['target'] = df2['target'].map(floor)

#export results
export_csv = df2.to_csv (r'export_dataframe.csv',index = None, header=True)
