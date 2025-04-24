import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# Step 1: Load and Balance Data
# -------------------------------


data = pd.read_csv('magic04.data', header=None)  

#put names for columns
data.columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10', 'class']

# change the data in column 'classs' into numeric data
data['class'] = data['class'].map({'g': 1, 'h': 0})

#split classes
gamma = data[data['class'] == 1]
hadron = data[data['class'] == 0]

#reduce gamma classes to balance data
gamma_balanced = gamma.sample(n=len(hadron), random_state=42)
balanced_data = pd.concat([gamma_balanced, hadron]).sample(frac=1, random_state=42)

# -------------------------------
# Step 2: Split Data
# -------------------------------

X = balanced_data.drop('class', axis=1)
y = balanced_data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------------
# Step 3: Classification Models
# -------------------------------

# (a) Decision Tree (no tuning)
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# (b) Naive Bayes (no tuning)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

# (c) Random Forest - tune n_estimators
rf_scores = {}
for n in [10, 50, 100]:
    rf_model = RandomForestClassifier(n_estimators=n, random_state=42)
    scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    rf_scores[n] = np.mean(scores)

best_rf_n = max(rf_scores, key=rf_scores.get)
best_rf_model = RandomForestClassifier(n_estimators=best_rf_n, random_state=42)
best_rf_model.fit(X_train, y_train)
rf_pred = best_rf_model.predict(X_test)
# (d) AdaBoost - tune n_estimators
ab_scores = {}
for n in [10, 50, 100]:
    ab_model = AdaBoostClassifier(n_estimators=n, random_state=42)
    scores = cross_val_score(ab_model, X_train, y_train, cv=5)
    ab_scores[n] = np.mean(scores)

best_ab_n = max(ab_scores, key=ab_scores.get)
best_ab_model = AdaBoostClassifier(n_estimators=best_ab_n, random_state=42)
best_ab_model.fit(X_train, y_train)
ab_pred = best_ab_model.predict(X_test)
# -------------------------------
# Step 5: Evaluation
# -------------------------------

def evaluate_model(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=1))
    print("Recall   :", recall_score(y_true, y_pred, zero_division=1))
    print("F1 Score :", f1_score(y_true, y_pred))

evaluate_model("Decision Tree", y_test, dt_pred)
evaluate_model("Naive Bayes", y_test, nb_pred)
evaluate_model(f"Random Forest (n={best_rf_n})", y_test, rf_pred)
evaluate_model(f"AdaBoost (n={best_ab_n})", y_test, ab_pred)