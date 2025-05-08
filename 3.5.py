import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from wittgenstein import RIPPER
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

data = [
    [1, 'youth', 'high', 'no', 'fair', 'no'],
    [2, 'youth', 'high', 'no', 'excellent', 'no'],
    [3, 'middle-aged', 'high', 'no', 'fair', 'yes'],
    [4, 'senior', 'medium', 'no', 'fair', 'yes'],
    [5, 'senior', 'low', 'yes', 'fair', 'yes'],
    [6, 'senior', 'low', 'yes', 'excellent', 'no'],
    [7, 'middle-aged', 'low', 'yes', 'excellent', 'yes'],
    [8, 'youth', 'medium', 'no', 'fair', 'no'],
    [9, 'youth', 'low', 'yes', 'fair', 'yes'],
    [10, 'senior', 'medium', 'yes', 'fair', 'yes'],
    [11, 'youth', 'medium', 'yes', 'excellent', 'yes'],
    [12, 'middle-aged', 'medium', 'no', 'excellent', 'yes'],
    [13, 'middle-aged', 'high', 'yes', 'fair', 'yes'],
    [14, 'senior', 'medium', 'no', 'excellent', 'no']
]

columns = ['RID', 'age', 'income', 'student', 'credit rating', 'Class: buys_computer']
df = pd.DataFrame(data, columns=columns).drop('RID', axis=1)

X = df.drop('Class: buys_computer', axis=1)
y = df['Class: buys_computer']

y_ripper = y.replace({'no': 0, 'yes': 1})
ripper = RIPPER()
ripper.fit(X, y_ripper)
ripper_pred = ripper.predict(X)
ripper_acc = accuracy_score(y_ripper, ripper_pred)

encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

dt = DecisionTreeClassifier()
dt.fit(X_encoded, y_encoded)
dt_pred = dt.predict(X_encoded)
dt_acc = accuracy_score(y_encoded, dt_pred)

nb = CategoricalNB()
nb.fit(X_encoded, y_encoded)
nb_pred = nb.predict(X_encoded)
nb_acc = accuracy_score(y_encoded, nb_pred)

print(f"RIPPER Training Accuracy: {ripper_acc}")
print(f"Decision Tree Training Accuracy: {dt_acc}")
print(f"Naive Bayes Training Accuracy: {nb_acc}")
