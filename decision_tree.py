# Decision Tree CLassifier

# Importing the libraries
import pandas as pd
import config

# Importing the datasets
datasets = pd.read_csv(config.DATAPATH)

X = datasets.iloc[:, 1:].values
Y = datasets.iloc[:, 0].values


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Fitting the classifier into the Training set

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier = classifier.fit(X_Train,Y_Train)
# Predicting the test set results

Y_Pred = classifier.predict(X_Test)

# Metric for evaluating decision tree
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)
fscore = f1_score(Y_Test,Y_Pred)
ascore = accuracy_score(Y_Test,Y_Pred)
print("Confusion Matrix: ")
print(cm)
print("F Score: ",fscore)
print("Accuracy Score: ", ascore)
