# PROBLEM 4
# 4. Write a program to do: A dataset collected in a cosmetics shop showing
# details of customers and whether or not they responded to a special offer
# to buy a new lip-stick is shown in table below. (Implement step by step
# using commands - Dont use library) Use this dataset to build a decision
# tree, with Buys as the target variable, to help in buying lipsticks in the
# future. Find the root node of the decision tree. 

# Simple ID3 root selection (no libraries)
import math
from collections import Counter, defaultdict

data = [
    {"Id":1, "Age":"<21",   "Income":"High",   "Gender":"Male",   "Ms":"Single",  "Buys":"No"},
    {"Id":2, "Age":"<21",   "Income":"High",   "Gender":"Male",   "Ms":"Married", "Buys":"No"},
    {"Id":3, "Age":"21-35", "Income":"High",   "Gender":"Male",   "Ms":"Single",  "Buys":"Yes"},
    {"Id":4, "Age":">35",   "Income":"Medium", "Gender":"Male",   "Ms":"Single",  "Buys":"Yes"},
    {"Id":5, "Age":">35",   "Income":"Low",    "Gender":"Female", "Ms":"Single",  "Buys":"Yes"},
    {"Id":6, "Age":">35",   "Income":"Low",    "Gender":"Female", "Ms":"Married", "Buys":"No"},
    {"Id":7, "Age":"21-35", "Income":"Low",    "Gender":"Female", "Ms":"Married", "Buys":"Yes"},
]

attributes = ["Age", "Income", "Gender", "Ms"]
target = "Buys"

def entropy(rows):
    counts = Counter(r[target] for r in rows)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c/total
        ent -= p * math.log2(p) if p>0 else 0
    return ent

def information_gain(rows, attr):
    base = entropy(rows)
    total = len(rows)
    parts = defaultdict(list)
    for r in rows:
        parts[r[attr]].append(r)
    weighted = 0.0
    for subset in parts.values():
        weighted += (len(subset)/total) * entropy(subset)
    return base - weighted

# compute IG for each attribute and pick the best
gains = {attr: information_gain(data, attr) for attr in attributes}
root = max(gains, key=gains.get)

print("Information gains:")
for a, g in gains.items():
    print(f" {a:7s}: {g:.4f}")
print("\nRoot attribute (highest IG):", root)
