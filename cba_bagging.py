from pprint import pprint
from binning import binning, bin_dataframe,append_col_name_to_df
from apriori import get_assoc_rules,get_freq_itemsets
from util import checkMajority,readDataset
import time
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import config
from m1 import M1,applyDefaultClass

# To suppress warnings
pd.options.mode.chained_assignment = None


DELIMITER = ','
MINSUPPORT = config.MINSUP
DATA_SOURCE = config.DATAPATH
# Read in data and extract columns as attributes 
# Our first column is the attribute we want to classify
data = readDataset(DATA_SOURCE, DELIMITER)
attributes = list(data.columns.values)
attribute_to_classify = attributes[0]


train,test = train_test_split(data,test_size=0.25,random_state=47)



start = time.process_time()
binning_references = binning(data)

### Bagging Procedure 
### Run a total of T=11 times, each time we generate a dataset by randomly selecting instances
### from our training data. We use that dataset to perform CBA
### At the end, we use majority of the results from all 11 random instances
T = 11

# Discretisation of dataset first 
# Create new DF to hold all lists of guesses 
guessDF = pd.DataFrame()
start = time.process_time()
for x in range(T): 
    train_copy = train.copy()
    bin_dataframe(train_copy, binning_references)
    bagging = pd.DataFrame()
    # Randomly select N instances to put into bagging with replacement

    for i in range(len(train_copy)): 
        bagging = pd.concat([bagging,train_copy.sample(1)])
    finalDF = append_col_name_to_df(bagging)

    try:

        frequentItems = get_freq_itemsets(finalDF, min_support=MINSUPPORT)
        classifiers = get_assoc_rules(frequentItems,attribute_to_classify)
        defaultClass,Majority = M1(train_copy,classifiers,attribute_to_classify)

        test_set = test.copy() 
        bin_dataframe(test_set, binning_references)
        test_set['classifyGuess'] = None 
        d = applyDefaultClass(defaultClass,Majority,test_set)
        toAppend = test_set['classifyGuess'].copy()
        guessDF = pd.concat([guessDF,toAppend],axis=1)

    except: 
        print("There are no available classifiers generated from M1.")


print(">>>>> All classifcation results from bagging <<<<< ")
print(guessDF)

majority = []

# Iterate over rows to find majority and append to a list
for x in range(len(guessDF)):
    y = guessDF.iloc[x].values.tolist()
    majority.append(checkMajority(y))
correct = 0 

test_set = test.copy() 
bin_dataframe(test_set, binning_references)
testClass = test_set[attribute_to_classify].values.tolist()
for x in range(len(majority)): 
    if str(majority[x]) == str(testClass[x]):
        correct += 1 

print(test_set)
print("Performing CBA-M1 with Bagging")
print(">>>>> Reading dataset from "+ str(DATA_SOURCE) + " <<<<<")
print("Correct number of classification on test set: " + str(correct))
print("Total number of rows in test set: "+ str(len(majority)))
print("Accuracy = "+ str(correct/len(majority)))


