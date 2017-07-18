#import libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import preprocessing

#load dataset
def load_dataset(filename):
    dataset=pd.read_csv(filename,sep=";")
    return dataset
dataset = load_dataset("/home/anushi/MLProjects/winequality-red.csv")

#creating matrices for output and input variables
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,11].values

#splitting dataset
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#feature selection
from sklearn.feature_selection import RFE
rfe = RFE(regressor,7)
fit = rfe.fit(x_train,y_train)

#Fitting variables to model
x_train = fit.transform(x_train)
x_test = fit.transform(x_test)
regressor.fit(x_train,y_train)

#predicting the result
y_pred = regressor.predict(x_test)

from sklearn.metrics import median_absolute_error
from sklearn.metrics import accuracy_score
print ("r squared=",regressor.score(x_test,y_test))


print ("accuracy score=",accuracy_score(y_pred,y_test))
print (median_absolute_errro(y_test,y_pred))


print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
Z=(" %s") % fit.ranking_
print Z

        
