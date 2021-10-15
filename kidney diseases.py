import numpy as np
import pandas as pd  
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
import prediction as fp 


train_data = pd.read_csv('C:\Program Files\Python39/kidney_disease.csv')

col = []
for i in train_data.columns:
    if train_data[i].isnull().sum() != 0 :
        col.append(i)

for i in col :
    fp.Random_Value_Implementation(train_data,i)


dict = {'yes' : 1,'\tyes' : 1,'no' : 0,'\tno' : 0}
train_data['htn'] = train_data['htn'].map(dict)
train_data['dm'] = train_data['dm'].map(dict)
train_data['cad'] = train_data['cad'].map(dict)
train_data['pe'] = train_data['pe'].map(dict)
train_data['ane'] = train_data['ane'].map(dict)


#print(train_data.head())
#print(train_data.dtypes)
dict2 = {'normal' : 0, 'abnormal' : 1}
train_data['rbc'] = train_data['rbc'].map(dict2)
train_data['pc'] = train_data['pc'].map(dict2)

dict3 = {'present' : 1, 'notpresent' : 0}


train_data['pcc'] = train_data['pcc'].map(dict3)
train_data['ba'] = train_data['ba'].map(dict3)

di = {'\t?' : '0', '\t6200' : '6200', '\t8400' : '8400','\t43' : '43'}
#train_data['pcv'] = train_data['pcv'].replace(to_replace={'\t?' : '0', '\t43' : '43'},inplace=True)
#train_data['wc'] = train_data['wc'].replace(to_replace={'\t?' : '0', '\t6200' : '6200', '\t8400' : '8400'},inplace=True)
#train_data['rc'] = train_data['rc'].replace(to_replace={'\t?' : '0'},inplace=True)

#print(train_data.isnull().sum())
#for i in train_data.columns :
#    print('{} are {}'.format(i,train_data[i].unique()))
for i in ['pcv','wc','rc'] :
    l = list(train_data[i])
    for j in range(len(l)) :
        if l[j] in di :
            l[j] = di[l[j]]
    train_data[i] = pd.DataFrame(l)


train_data['pcv'] = train_data['pcv'].astype(int)
train_data['wc'] = train_data['wc'].astype(int)
train_data['rc'] = train_data['rc'].astype(float)


dict4 = {'poor' : 0, 'good' : 1}
train_data['appet'] = train_data['appet'].map(dict4)

#print(train_data['classification'].unique())
dict5 = {'ckd' : 1, 'notckd' : 0, 'ckd\t' : 1}
train_data['classification'] = train_data['classification'].map(dict5)
train_data.drop('id', axis=1,inplace=True)


train_data.dropna(axis=1,inplace=True)
#print(train_data.isnull().sum())
#print(train_data.head())
#print(train_data.shape)


X = train_data.drop('classification', axis=1)
Y = train_data['classification']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
fp.predict(RandomForestRegressor(),X_train,Y_train,X_test,Y_test)

