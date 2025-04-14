from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = make_classification(
    n_samples = 100,
    n_features = 2,
    n_redundant = 0,
    n_informative = 2,
    n_clusters_per_class = 1,
    class_sep = 2,  
    random_state = 42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

accuracy
