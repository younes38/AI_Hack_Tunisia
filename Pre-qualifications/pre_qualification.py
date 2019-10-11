import numpy as np
import pandas as pd
from itertools import permutations
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree

np.random.seed(0)
df = pd.read_csv("Train_v2.csv")
#columns to be considered 
cols =['age_of_respondent','education_level','cellphone_access','job_type','location_type', 'relationship_with_head']

def to_nb(df): #convert strings to numbers    
    def cv2(s):
        if s == "Yes" :
            return 1
        return 0
    try:
        col = "bank_account"
        df[col] = list(map(cv2, df[col]))
    except: 
        print("err")
    for col in cols:
        mp = dict()
        def convert(s) : 
            return mp[s]
        if df[col].dtype != np.int64  and col != "uniqueid" : 
            col_content = df[col].values.tolist()
            unique = set(col_content)
            x = 0
            for u in unique:
                mp[u] = x
                x += 1
            df[col] = list(map(convert, df[col]))
    return df
df = to_nb(df)  

features = df[cols].values
labels = df['bank_account'].values
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2)

#scale data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
df["age_of_respondent"].apply(np.log)
df["household_size"].apply(np.log)

clf = RandomForestClassifier(n_estimators = 1000, n_jobs = -1, max_depth = 10, min_samples_split = 200)
clf = clf.fit(x_train, y_train)
res = clf.score(x_test, y_test)
print(res)

#test file
df2 = pd.read_csv("Test_v2.csv")
df3 = pd.read_csv("Test_v2.csv")

df2 = to_nb(df2)
problem_test = df2[cols]

problem_test = scaler.transform(problem_test)
df["age_of_respondent"].apply(np.log)
df["household_size"].apply(np.log)

res = clf.predict(problem_test)
print(res)

df2["uniqueid"] = df2["uniqueid"].map(str) +" x " + df3["country"].map(str)
print(df2["uniqueid"])
df2["bank_account"] = list(res)
dfs =  df2[["uniqueid", "bank_account"]]
export_csv = dfs.to_csv (r'export_dataframe.csv',index = None, header=True)
