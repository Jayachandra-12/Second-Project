from math import e
import numpy as np
import pandas as pd  
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

def Random_Value_Implementation(data,feature) :
    random_sample = data[feature].dropna().sample(data[feature].isnull().sum())
    random_sample.index = data[data[feature].isnull()].index
    data.loc[data[feature].isnull(),feature] = random_sample

def predict(ml_model,X_train,Y_train,X_test,Y_test) :
    model = ml_model.fit(X_train,Y_train)
    print("Training score : {}".format(model.score(X_train,Y_train)))
    Y_prediction = model.predict(X_test)
    #print(Y_prediction)
    for i in range(len(Y_prediction)) :
        if Y_prediction[i] >= 0.5 :
            Y_prediction[i] = 1
        else :
            Y_prediction[i] = 0
    
    print("Predictions are : \n{}".format(Y_prediction))
    print('\n')


    r2_score = metrics.r2_score(Y_test,Y_prediction)
    print('r2_score is {}'.format(r2_score))
    print('MAE is {}'.format(metrics.mean_absolute_error(Y_test,Y_prediction)))
    print('MSE is {}'.format(metrics.mean_squared_error(Y_test,Y_prediction)))
    print('RMSE is {}'.format(np.sqrt(metrics.mean_squared_error(Y_test,Y_prediction))))
    sns.distplot(Y_test-Y_prediction)
    plt.show()
