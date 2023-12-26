import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
import config

data = pd.read_csv(config.DATAPATH)

#create X feature variables and y outcome variable
X=data.iloc[: , 1:].values
y=data.iloc[:, 0].values

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#train svm classifier
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

#predict for test set
y_pred = clf.predict(X_test)

#evaluate accuracy
print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print("F Score:",metrics.f1_score(y_test, y_pred))
print("Accuracy Score:",metrics.accuracy_score(y_test, y_pred))

