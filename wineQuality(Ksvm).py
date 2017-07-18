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

#fitting kernal SVM classifier to the dataset
from  sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(x_train,y_train)

#predicting results
y_pred = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)

print "Accuracy = ",acc
