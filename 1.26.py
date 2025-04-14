import numpy as np
from sklearn import datasets, svm
from sklearn.metrics import accuracy_score

X, y = datasets.make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

clf = svm.SVC(kernel='rbf', gamma=2, C=1)
clf.fit(X, y)

predictions = clf.predict(X)
accuracy = accuracy_score(y, predictions)
