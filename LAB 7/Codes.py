from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

# Example: Random Forest hyperparameter tuning
X = ...  # your features
y = ...  # your target

param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
search.fit(X, y)

print("Best parameters:", search.best_params_)
print("Best cross-validation score:", search.best_score_)import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Prepare your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "SVM": SVC(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(),
    "NaiveBayes": GaussianNB(),
    "MLP": MLPClassifier()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    results.append({
        "Model": name,
        "Train Accuracy": accuracy_score(y_train, y_pred_train),
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Train F1": f1_score(y_train, y_pred_train, average='weighted'),
        "Test F1": f1_score(y_test, y_pred_test, average='weighted')
    })

df_results = pd.DataFrame(results)
print(df_results)from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Prepare your regression data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

regressors = {
    "SVR": SVR(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(),
    "CatBoost": CatBoostRegressor(verbose=0),
    "AdaBoost": AdaBoostRegressor(),
    "XGBoost": XGBRegressor(),
    "MLP": MLPRegressor()
}

reg_results = []
for name, reg in regressors.items():
    reg.fit(X_train, y_train)
    y_pred_train = reg.predict(X_train)
    y_pred_test = reg.predict(X_test)
    reg_results.append({
        "Regressor": name,
        "Train MSE": mean_squared_error(y_train, y_pred_train),
        "Test MSE": mean_squared_error(y_test, y_pred_test),
        "Train R2": r2_score(y_train, y_pred_train),
        "Test R2": r2_score(y_test, y_pred_test)
    })

df_reg_results = pd.DataFrame(reg_results)
print(df_reg_results)from sklearn.cluster import AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt

# Example clustering on your features
X = ...  # your features

# Hierarchical Clustering
hier = AgglomerativeClustering(n_clusters=3)
labels_hier = hier.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels_hier)
plt.title("Hierarchical Clustering")
plt.show()

# Density-Based Clustering (DBSCAN)
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_db = dbscan.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels_db)
plt.title("DBSCAN Clustering")
plt.show()