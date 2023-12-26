import pandas as pd

def readDataset(filename, delimiter):
  return pd.read_csv(filename, index_col=False, delimiter=delimiter)
  
# Compute Memory usage by the database
def dfMemorySize(df):
  mem_in_byte = df.memory_usage(deep=True).sum()
  mem_in_mb = float(mem_in_byte) / float(1000000)
  return " Memory used for this dataframe is: " + str(mem_in_mb) + "MB"

def checkMajority(arr):
  if not arr: 
    return
  counter = {} 
  for element in arr: 
      if element not in counter:
        counter[element] = 1
      else: 
        counter[element] += 1 
  
  return max(counter,key=counter.get)
      

def checkAccuracy(dataset,attr): 
  # correct = dataset[(str(dataset[attr]) == str(dataset['classifyGuess']))]
  # return len(correct) / len(dataset)
  correct = 0
  testClass = dataset[attr].values.tolist()
  guess = dataset['classifyGuess'].values.tolist()
  for x in range(len(dataset)): 
    if str(testClass[x]) == str(guess[x]):
        correct += 1 
  
  return correct / len(dataset)



