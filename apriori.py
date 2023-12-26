import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from binning import delimiter


# mlxtend requires us to use onehot encoding first before generating frequent itemsets 
def encode_df(df):
  te = TransactionEncoder()
  te_arr = te.fit(df).transform(df)
  result = pd.DataFrame(te_arr, columns = te.columns_)
  return result


def get_freq_itemsets(df, min_support=0.6):
  df_list = df.values.tolist()
  encoded_data = encode_df(df_list)
  freq_items = apriori(encoded_data, min_support=min_support, use_colnames=True)
  return freq_items


def get_assoc_rules(freq_items,attr):
  rules = association_rules(freq_items, metric = 'lift', min_threshold = 1)
  rules = rules.sort_values(['confidence', 'lift'], ascending = [False, False])

  rules = filterRules(rules,attr)
  rules = formatRules(rules)


  return rules

# Filter df to only include rules with one consequent type 
def filterRules(rules, attribute):
    copy = rules.copy()
    for index, rule in copy.iterrows():

      if len(rule['consequents']) > 1:
        copy.drop(index, inplace=True)
        continue

      consequent_attr = list(rule['consequents'])[0].split(delimiter())[0]
      if consequent_attr != attribute:
        copy.drop(index, inplace=True)
        continue

    return copy


def formatRules(rules):
    c = []

    index = 0
    for _, rule in rules.iterrows():
      c.append(
        {
          "index": index,
          "antecedents": generate_antecedents_dict(list(rule['antecedents'])),
          "consequents": list(rule['consequents'])[0].split(delimiter())[1]
        }
      )
      index += 1

    return c


def generate_antecedents_dict(antecedents):
    result = {}

    for antecedent in antecedents:
      key, value = antecedent.split(delimiter())
      if key in result:
        result[key].append(value)
      else:
        result[key] = [value]

    return result
