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
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

#fitting KNN to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric='minkowski',p=2)
classifier.fit(x_train,y_train)

# predicting results
y_pred = classifier.predict(x_test)

y_proba = classifier.predict_proba(x_test)[0:10]

src = classifier.score(x_test,y_test)
srcm = classifier.score(x_test,y_pred)

print y_proba
print src
print srcm
