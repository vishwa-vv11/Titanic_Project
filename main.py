#Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1 . Import and basic info of the dataset
df = pd.read_csv("Titanic-Dataset.csv")
print("First 5 rows:\n", df.head())
print("Basic Info of the dataset:\n")
df.info()

# 2 . Handle missing values
print("\nMissing values before handling:\n", df.isnull().sum())
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin'] = df['Cabin'].fillna('U')
df['Deck'] = df['Cabin'].apply(lambda x: x[0])
df['Has_Cabin'] = df['Cabin'].apply(lambda x: 0 if x=='U' else 1)
print("\nMissing values after handling:\n")
print(df.isnull().sum())

# 3 .  Convert categorical features into numerical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
deck_mapping = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':0}
df['Deck'] = df['Deck'].map(deck_mapping)
print("\nData after encoding categorical features:\n")
print(df.head())

# 4 .  Normalize / Standardize numerical features
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print("\nData after standardizing numerical features:\n")
print(df.head())


#   5 . Visualize outliers using boxplots and remove them
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
plt.show()

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]
print("\nShape after removing outliers:", df.shape)