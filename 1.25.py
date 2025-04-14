from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

mnist = datasets.fetch_openml('mnist_784', version = 1, as_frame = False, parser = 'auto')
X, y = mnist.data, mnist.target.astype(int)
X, y = X[:10000], y[:10000]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42, stratify = y
)

pipeline = make_pipeline(
    StandardScaler(),
    SVC(kernel = 'rbf', decision_function_shape = 'ovr')
)

param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv = 3,
    n_jobs = -1,
    verbose = 2
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(grid_search.best_params_)
