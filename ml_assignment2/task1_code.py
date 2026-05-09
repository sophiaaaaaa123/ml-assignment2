# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB



# read csv file for task1
base = "Desktop/ml_assignment2/task1_data/"
train1_meta = pd.read_csv(base + "train_metadata.csv")
test1_meta = pd.read_csv(base + "test_metadata.csv")
color = pd.read_csv(base + "color_histogram.csv")
hog = pd.read_csv(base + "hog_pca.csv")
add = pd.read_csv(base + "additional_features.csv")

features1 = color.merge(hog, on="image_id")
features1 = features1.merge(add, on="image_id")


# task 1 training data
train1_df = train1_meta.merge(features1, on="image_id")

X = train1_df.drop(columns=["image_id", "image_path", "class_id", "class_name"])    # features
y = train1_df["class_id"]   # label

# task 1 test data
test_df = test1_meta.merge(features1, on="image_id")
X_test = test_df.drop(columns=["image_id", "image_path"])

# split data set into 80% training + 20% validation
X_train1, X_val1, y_train1, y_val1 = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# scaler
scaler = StandardScaler()

# learn mean/std from training data, then scale training data
X_train_scaled1 = scaler.fit_transform(X_train1)

# use SAME mean/std to scale validation data
X_val_scaled1 = scaler.transform(X_val1)

# models used in task 1:
models = {
    "Logistic L2 C=1": LogisticRegression(C=1, penalty="l2", max_iter=5000),
    "Logistic L2 C=10": LogisticRegression(C=10, penalty="l2", max_iter=5000),
    "Logistic L1 C=1": LogisticRegression(C=1, penalty="l1", solver="liblinear", max_iter=5000),       # best (1)
    "Logistic L1 C=10": LogisticRegression(C=10, penalty="l1", solver="liblinear", max_iter=5000),      # best (3)

   
    "Random Forest 200 itr": RandomForestClassifier(n_estimators=200, random_state=1),
    "Random Forest 300 itr": RandomForestClassifier(n_estimators=300, random_state=1),      # best (2)
    "Random Forest 400 itr": RandomForestClassifier(n_estimators=400, random_state=1),

    "kNN 5": KNeighborsClassifier(n_neighbors=5),
    "kNN 7": KNeighborsClassifier(n_neighbors=7),

    "SVM Linear 1": SVC(kernel="linear", C=1),
    "SVM Linear 10": SVC(kernel="linear", C=10),
    "SVM RBF 1": SVC(kernel="rbf", C=1),
    "SVM RBF 10": SVC(kernel="rbf", C=10),    
}

# train and predict
for name, model in models.items():

    model.fit(X_train_scaled1, y_train1)
    pred = model.predict(X_val_scaled1)

    accuracy = accuracy_score(y_val1, pred)
    precision = precision_score(y_val1, pred, average="macro")
    recall = recall_score(y_val1, pred, average="macro")
    f1 = f1_score(y_val1, pred, average="macro")

    print(
        name,
        "accuracy:", round(accuracy, 4),
        "precision:", round(precision, 4),
        "recall:", round(recall, 4),
        "f1:", round(f1, 4)
    )

