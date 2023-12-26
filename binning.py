import pandas as pd
import numpy as np
import math
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def delimiter():
  return '___'

def binning(df):
  bin = {}
  for column in df.columns: 
    if df[column].dtype in ['int64', 'float64']:
      bin[column] = percentileValues(df[column].tolist())

  print(bin)
  
  return bin


def percentileValues(data):
  results = []

  a = np.array(data)
  for i in range(9):
    results.append(
      np.percentile(a, (i + 1) * 10)
    )
  return results


def bin_dataframe(df, bin):
  # col = numerical column , references = percentiles from 10 to 90th
  for col, references in bin.items():
    mapping_references = {}

    min_value = 0
    for index, item in enumerate(references):
      mapping_references[index] = list(
        df[
          # for each column, find values between min and item, then item = percentile values 
          df[col].between(min_value, item) 
        ].index
      )
      min_value = item


    mapping_references[len(references)] = list(
        df[
          df[col] >= min_value
        ].index
      )

    for i, indexes in mapping_references.items():
      df.loc[df.index.isin(indexes), col] = col +str(i)


def filter_df_columns(df, cols=None):
  filtered_df = df if cols is None else df[cols]
  return filtered_df


def append_col_name_to_df(df):
  for col in list(df.columns):
    df[col] = col + delimiter() + df[col].astype(str)
  
  return df




