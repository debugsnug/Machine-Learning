import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

file_path = r"D:\\OneDrive - Amrita vishwa vidyapeetham\\SEM 5\\23CSE301 Machine Learning\\LAB\\Machine-Learning\\LAB 3\\20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv"
df = pd.read_csv(file_path)

# Features and Target
X = df[['F1','F2','F3','F4']]
y = df['CallerSex'].astype('category').cat.codes   # Encode target if categorical

# Define parameter grid for KNN
param_dist = {
    'n_neighbors': randint(1, 20),       # Randomly try k from 1 to 20
    'weights': ['uniform', 'distance'],  # Weighting strategy
    'p': [1, 2]                          # 1 = Manhattan, 2 = Euclidean
}

# Initialize model
knn = KNeighborsClassifier()

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=knn,
    param_distributions=param_dist,
    n_iter=20,             # number of random parameter settings
    cv=5,                  # 5-fold cross-validation
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

# Fit search
random_search.fit(X, y)

print("Best Hyperparameters:", random_search.best_params_)
print("Best CV Accuracy:", random_search.best_score_)


print("--------------------------------------------------------------------------------------------")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Classifiers to compare
models = {
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "MLP (Neural Net)": MLPClassifier(max_iter=500, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

# Store results
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Train Accuracy": accuracy_score(y_train, y_train_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred, average='weighted'),
        "Recall": recall_score(y_test, y_test_pred, average='weighted'),
        "F1-Score": f1_score(y_test, y_test_pred, average='weighted')
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(results_df)


# SVM:
    #Train: 94.8%, Test: 94.3% → very consistent → good generalization.
    #Precision slightly lower (0.889) → it sometimes mislabels negatives as positives.
    #F1: 0.916 → reliable.
#Decision Tree
    #Train: 100%, Test: 90.8% → classic overfit (memorized training set, weaker test performance).
    #Precision/Recall balanced but not great.
    #Not the best standalone choice.
#Random Forest
    #Train: 100%, Test: 94.3%.
    #Better generalization than single tree because of bagging.
    #Precision (0.924) & Recall (0.943) → very strong.
#AdaBoost
    #Train: 95.4%, Test: 94.3%.
    #Very stable.
    #Highest F1 (0.934) → excellent balance.
#Naive Bayes
    #Train: 93.3%, Test: 92.2%.
    #Slightly weaker than others, but still good.
    #Assumes feature independence (which may not fully hold for acoustic features).
#MLP (Neural Net)
    #Train: 94.8%, Test: 92.9%.
    #Precision lower (0.889) → struggles with false positives.
    #Still fairly strong.
#XGBoost
    #Train: 100%, Test: 94.3%.
    #Similar to Random Forest but often more efficient.
    #Good balance of metrics.
#CatBoost
    #Train: 97.9%, Test: 94.3%.
    #Avoids complete overfitting (unlike XGBoost/Random Forest).
    #Very stable.