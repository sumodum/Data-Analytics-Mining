import pandas as pd
import numpy as np 
from util import checkMajority

# Implement CBA-CB M1 Algorithm 
def M1(dataset,cars,attr): 
    RuleList = []
    totalWrongCovered = 0 
    for i in range(len(cars)): 
        car = cars[i]
        if (dataset.empty):
            break 
        else: 
            coveredIndexes = coveredByRule(dataset,car['antecedents'])
            if len(coveredIndexes) == 0: 
                continue 
    
            else:
                print(car)
                m,e,w = calculateMajorityandError(dataset,coveredIndexes,attr,car['consequents'],totalWrongCovered)
                totalWrongCovered = w
                car['majority'] = m
                car['error'] = e
                RuleList.append(car)
    if len(RuleList) == 0:
        return 
    # Find rule with least error and prune every rule below it 
    lowestError,i,maj = RuleList[0]['error'],RuleList[0]['index'],RuleList[0]['majority']
    for rule in RuleList: 
        a,b,c = rule['error'],rule['index'],rule['majority']
        if a<lowestError: 
            lowestError = a 
            i = b 
            maj=c 
    result = []
    for rule in RuleList: 
        if rule['index'] <= i: 
            result.append(rule)

    return result,maj

# Apply our classifer to test set 
def applyDefaultClass(defaultClass,majority,dataset): 
    for rule in defaultClass:
        coveredIndexes = coveredByRule(dataset,rule['antecedents'])
        dataset.loc[coveredIndexes,'classifyGuess'] = rule['consequents']
    
    dataset.classifyGuess.fillna(value=majority,inplace=True)
    return dataset 

# Figure out which record is covered by the LHS of a rule. 
def coveredByRule(df,LHS): 
    first = True 
    for col,value in LHS.items():
        if first:
            index_list = df[(df[col].isin(value))].index.tolist()
            first = False 
        else: 
            temp = df[(df[col].isin(value))].index.tolist()
            temp2 = [value for value in index_list if value in temp]
            index_list = temp2 
    return index_list

# For each covered rule, we calculate the total errors + majority class left after marking those covered tuples. 
def calculateMajorityandError(df,coveredIndexes,attr,consequent,totalWrongCovered):
    # Calculate wrong covered 
    totalErrors = 0 
    temp = df.loc[coveredIndexes] 
    consequentList = temp[attr].values.tolist()

    errorList = [x for x in consequentList if x != consequent]
    wCov = len(errorList)
    totalWrongCovered += wCov 

    df.drop(coveredIndexes, axis=0, inplace=True)
    majority = checkMajority(consequentList)
    consequentList2 = df[attr].values.tolist()
    if majority: 
        errorList2 = [x for x in consequentList2 if x != majority]
        totalErrors += len(errorList2)

    totalErrors += totalWrongCovered
    # Majority class left 
    return majority,totalErrors,totalWrongCovered