# PROBLEM 5
#(also similar for problems 6,7,8) 
#
# 5. Write a program to do: A dataset collected in a cosmetics shop showing
# details of customers and whether or not they responded to a special offer
# to buy a new lip-stick is shown in table below. (Use library commands)
# According to the decision tree you have made from the previous training
# data set, what is the decision for the test data: [Age < 21, Income = Low,
# Gender = Female, Marital Status = Married]? 


# Using sklearn if available, otherwise fallback to a simple rule.
# Trains a decision tree on the given dataset and predicts for the test instance:
# [Age <21, Income=Low, Gender=Female, Ms=Married]
# This code is written to run in this environment; it will use sklearn if installed,
# otherwise it will use a simple rule-based fallback (derived from the training data).
import sklearn, pandas as pd


# data = [
#     {"Id":1, "Age":"<21",   "Income":"High",   "Gender":"Male",   "Ms":"Single",  "Buys":"No"},
#     {"Id":2, "Age":"<21",   "Income":"High",   "Gender":"Male",   "Ms":"Married", "Buys":"No"},
#     {"Id":3, "Age":"21-35", "Income":"High",   "Gender":"Male",   "Ms":"Single",  "Buys":"Yes"},
#     {"Id":4, "Age":">35",   "Income":"Medium", "Gender":"Male",   "Ms":"Single",  "Buys":"Yes"},
#     {"Id":5, "Age":">35",   "Income":"Low",    "Gender":"Female", "Ms":"Single",  "Buys":"Yes"},
#     {"Id":6, "Age":">35",   "Income":"Low",    "Gender":"Female", "Ms":"Married", "Buys":"No"},
#     {"Id":7, "Age":"21-35", "Income":"Low",    "Gender":"Female", "Ms":"Married", "Buys":"Yes"},
# ]

df = pd.read_csv("./Datasets/Lipstick.csv")

# Test instance
test = {"Age":"<21", "Income":"Low", "Gender":"Female", "Ms":"Married"}

attributes = ["Age","Income","Gender","Ms"]
target = "Buys"

# Prepare X and y
X = df[attributes]
y = df[target]

# Try using sklearn
try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.preprocessing import OrdinalEncoder
    import numpy as np

    enc = OrdinalEncoder()  # encodes categorical values to integers
    X_enc = enc.fit_transform(X)
    clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
    clf.fit(X_enc, y)

    # encode test and predict
    test_enc = enc.transform([[test[a] for a in attributes]])
    pred = clf.predict(test_enc)[0]

    print("Sklearn decision tree used.")
    print("Encoded categories mapping (feature -> categories):")
    for i,feat in enumerate(attributes):
        print(f" {feat}: {list(enc.categories_[i])}")
    print("\nDecision tree (text):")
    print(export_text(clf, feature_names=attributes))
    print("Test instance:", test)
    print("Prediction:", pred)

except Exception as e:
    # Fallback: simple rule derived from training data (no libraries)
    # Observed from the trained tree (and IG calculation): Age is root with mapping:
    #  '<21'   -> No  (both examples were No)
    #  '21-35' -> Yes (both examples were Yes)
    #  '>35'   -> mixed but majority Yes (2 Yes, 1 No) -> predict Yes
    print("sklearn not available or failed; using fallback rule-based predictor.")
    def predict_rule(inst):
        if inst["Age"] == "<21":
            return "No"
        if inst["Age"] == "21-35":
            return "Yes"
        if inst["Age"] == ">35":
            # majority in >35 subset is Yes (2 Yes, 1 No)
            return "Yes"
        return "Unknown"

    pred = predict_rule(test)
    print("Test instance:", test)
    print("Prediction (fallback):", pred)
