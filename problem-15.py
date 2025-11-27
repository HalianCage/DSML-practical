# titanic_simple.py — minimal EDA with Seaborn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")      # make sure file is in same folder
print(df.head())                     # quick look

# simple cleanup
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['FamilySize'] = df.get('SibSp',0) + df.get('Parch',0) + 1

sns.set(style="whitegrid")

plt.figure(figsize=(5,4))
sns.countplot(data=df, x="Survived").set_title("Survived (0=No,1=Yes)")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Sex", hue="Survived").set_title("Survival by Sex")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Pclass", hue="Survived").set_title("Survival by Class")
plt.show()

plt.figure(figsize=(7,4))
sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True).set_title("Age by Survival")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="Pclass", y="Fare").set_title("Fare by Class")
plt.show()

plt.figure(figsize=(7,4))
sns.countplot(data=df, x="FamilySize", hue="Survived").set_title("Family Size vs Survival")
plt.show()

# quick numeric summaries
print("\nOverall survival rate:", df['Survived'].mean())
print("Survival rate by sex:\n", df.groupby('Sex')['Survived'].mean())
print("Survival rate by class:\n", df.groupby('Pclass')['Survived'].mean())
