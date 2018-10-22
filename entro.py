# Imtiaz Ahmed
import pandas as pd
import math

def measure(dictionary, method):
    # method is of type String, either "entropy" or "gini"
    if (method != "entropy" and method != "gini"):
        print("param method is of type String, either \"entropy\" or \"gini\"")
        return;
    
    # Get total number of values in dictionary
    total = 0;
    for d in dictionary:
        total += dictionary[d]
    
    # Calculate probability and value using specified method
    sum = 0;
    for d in dictionary:
        # multiply by 1.0 to preserve float value
        prob = (dictionary[d]*1.0/total)
        # -H = sig P(i) * log2 P(i)
        if (method == "entropy"):
            sum += prob*math.log(prob,2)
        # Gini = 1 -  sig P(i)^2
        elif (method == "gini"):
            sum += math.pow(prob,2)
    
    if (method == "entropy"):
        return (-sum)
    elif (method == "gini"):
        return (1-sum)

# This function takes a  dataframe, a target feature (String) and a method of 
# calculation (String) as parameters and returns the descriptive feature within 
# the dataframe that provides the highest information gain.
def rem(df, target, method):
    select_features = {}
    # Set up dictionary of target feature to find entropy/gini
    target_dict = {}
    for i, v in df[target].value_counts().iteritems():
        target_dict[i] = v
    print(method + " of target feature " + target + ": " + str(measure(target_dict, method)) + "\n")
    
    # Calculate rem, excluding the target feature of course
    for col in df.drop([target], axis=1):
        # Gather all possible values in a column into a list...
        uniqueValues = df[col].unique()
        sum = 0
        # For each possible value in a column
        for u in uniqueValues:
            # Partition data by each value into table; printed with print(temp)
            part = df[df[col] == u]
            print(col + " = " + u)
            
            # Gather count for target feature values within the partition
            part_dict = {}
            for i, v in part[target].value_counts().iteritems():
                part_dict[i] = v
            print(part_dict)
            
            # rem = sig P(i)/n * H(P(i)/Gini(P(i)), target)
            sum += ( part[col].count()*1.0/df[col].count() ) * measure(part_dict,method)
        print("rem of " + col + " using " + method + ": " + str(sum))
        # IG = H/Gini - rem
        print("IG of " + col + " using " + method + ": " + str(measure(target_dict, method)-sum) + "\n")
        select_features[col] = measure(target_dict, method)-sum
    
    print("Feature with highest IG using " + method + ": " + str(max(select_features)) + "\n")
    return str(max(select_features))

df = pd.read_csv("dataset_a.csv", usecols=['Rain','Sprinkler','Grass'])
target = "Grass"

# Split data into subsets based on values of the first attribute with highest
# information gain.
first = rem(df, target, "entropy")
print("------------------------------");
first_unique = df[first].unique()
for f in first_unique:
    print("When " + first + " = " + f)
    branch = df[df[first] == f].drop([first], axis = 1)
    rem(branch, target, "entropy")
    print("------------------------------");
    
print("End program.")