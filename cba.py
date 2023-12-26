from pprint import pprint
from binning import binning, bin_dataframe, append_col_name_to_df
from util import dfMemorySize,checkAccuracy,readDataset
from apriori import get_assoc_rules,get_freq_itemsets
import time
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from m1 import M1,applyDefaultClass
import config


# To suppress warnings
pd.options.mode.chained_assignment = None

# Import constants from config
DELIMITER = ','
MINSUPPORT = config.MINSUP
DATA_SOURCE = config.DATAPATH

# Read in data and extract columns as attributes 
# Our first column is the attribute we want to classify! 
data = readDataset(DATA_SOURCE, DELIMITER)
attributes = list(data.columns.values)
attribute_to_classify = attributes[0]

# Split our data into training and test dataset 
train,test = train_test_split(data,test_size=0.25,random_state=47)



start = time.process_time()

# Discretization of our continous attributes
binning_references = binning(data)
train_copy = train.copy()
bin_dataframe(train_copy, binning_references)
finalDF = append_col_name_to_df(train_copy)

# Apply apriori algorithm to get a set of rules whose consequent = attribute to classify

try:
       frequentItems = get_freq_itemsets(finalDF, min_support=MINSUPPORT)
       classifiers = get_assoc_rules(frequentItems,attribute_to_classify)

       # Apply our rules on train dataset to build classifiers 
       bin_dataframe(train,binning_references)
       defaultClass,Majority = M1(train,classifiers,attribute_to_classify)

       test_set = test.copy()
       bin_dataframe(test_set, binning_references)

       # Apply classifiers and default class to test set and measure accuracy 
       test_set['classifyGuess'] = None 
       d = applyDefaultClass(defaultClass,Majority,test_set)
       print(" >>>> Applying default class to test set : ")
       elapsed_time = time.process_time() - start

       print(d)
       a = checkAccuracy(d,attribute_to_classify)

       print(">>>>> Reading dataset from "+ str(DATA_SOURCE) + " <<<<<")
       print(">>>>> Rows in dataset: "+ str(len(data))+ " rows <<<<<")
       print(">>>>> Rows in Training Set: "+ str(len(finalDF))+ " rows <<<<<")
       print(">>>>> Rows in Test Set: "+ str(len(test))+ " rows <<<<<")
       print(">>>>> Number of rules in defaultClass : " + str(len(defaultClass)) + " <<<<<")
       print(">>>>> Accuracy of M1 Algorithm : " + str(a) + " <<<<<")
       print(">>>>> Time taken to train classifer and it on test data :  " + str(elapsed_time) + " secs" + "<<<<<")
except:
       print("There are no available classifiers generated from M1.")

