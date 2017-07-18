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

#fitting RandomForestClassification to the training set
from sklearn.ensemble import RandomForestClassifier
classifier =  RandomForestClassifier(n_estimators=20,criterion='entropy',
                                     random_state = 0)
classifier.fit(x_train,y_train)

#Predicting results
y_pred = classifier.predict(x_test)

preds = classifier.predict_proba(x_test)[0:10]

scr = classifier.score(x_test,y_pred)

print preds

print scr
