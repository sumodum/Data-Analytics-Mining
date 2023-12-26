import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import config

df = pd.read_csv(config.DATAPATH) #add csv name
Y = df[df.columns[0]]
X = df[df.columns[1:]]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state = 0) # 70% training and 30% test

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100, criterion='entropy')

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,Y_train)

Y_pred=clf.predict(X_test)


print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
print("F1 Score:", metrics.f1_score(Y_test, Y_pred))
print("Corr Matrix: \n", metrics.confusion_matrix(Y_test, Y_pred))
