import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Check missing values
print("Original dataset shape: ", df.shape)
print("Number of missing values: ", df.isnull().sum())
df = df.dropna(how="any")
print("Shape after removing missing values: ", df.shape)

# Data statistics
print("\nData statistics:")
print(df.describe())

# Preview dataset
print("\nData preview:")
print(df.head())

# Feature matrix and target vector
featuer_names = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
target_name = "target"
X = df.loc[:, feature_names].values
y = df.loc[:, target_name].values

print("\nFeature matrix shape: ", X.shape)
print("Target vector shape: ", y.shape)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
