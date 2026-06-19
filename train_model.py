import time
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

data = pd.read_csv("dataset.csv")

X = data[["temperature", "humidity", "wind"]]
y = data["rain"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Decision Tree ----------------
start = time.time()
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
dt_time = time.time() - start

dt_pred = dt.predict(X_test)

# ---------------- Random Forest ----------------
start = time.time()
rf = RandomForestClassifier(n_estimators=50, max_depth=4)
rf.fit(X_train, y_train)
rf_time = time.time() - start

rf_pred = rf.predict(X_test)

# ---------------- Metrics Function ----------------
def get_metrics(y_test, pred):
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, pred),
        "Confusion Matrix": confusion_matrix(y_test, pred)
    }

dt_metrics = get_metrics(y_test, dt_pred)
rf_metrics = get_metrics(y_test, rf_pred)

# ---------------- Save Models ----------------
pickle.dump(dt, open("decision.pkl", "wb"))
pickle.dump(rf, open("random.pkl", "wb"))

# ---------------- Save Scores ----------------
pickle.dump({
    "Decision Tree": {
        "metrics": dt_metrics,
        "time": dt_time
    },
    "Random Forest": {
        "metrics": rf_metrics,
        "time": rf_time
    }
}, open("scores.pkl", "wb"))

print("Training Complete")