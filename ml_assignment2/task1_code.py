# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# read csv file for task1
base = "Desktop/ml_assignment2/task1_data/"
train1_meta = pd.read_csv(base + "train_metadata.csv")
test1_meta = pd.read_csv(base + "test_metadata.csv")
color = pd.read_csv(base + "color_histogram.csv")
hog = pd.read_csv(base + "hog_pca.csv")
add = pd.read_csv(base + "additional_features.csv")

features = color.merge(hog, on="image_id")
features = features.merge(add, on="image_id")


# task 1 training data
train_df = train1_meta.merge(features, on="image_id")

X = train_df.drop(columns=["image_id", "image_path", "class_id", "class_name"])    # features
y = train_df["class_id"]   # label

# task 1 test data
test_df = test1_meta.merge(features, on="image_id")
X_test = test_df.drop(columns=["image_id", "image_path"])

# split data set into 80% training + 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)


# cv5 needed for Section 5 
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


# models used in task 1:
models = {
    "Logistic L1 C=1": LogisticRegression(C=1, l1_ratio=1, solver="saga", max_iter=5000), 
    "Logistic L1 C=10": LogisticRegression(C=10, l1_ratio=1, solver="saga", max_iter=5000),
    "SVM RBF C=10": SVC(kernel="rbf", C=10),
    "SVM Linear C=1": SVC(kernel="linear", C=1),
    "Random Forest 300": RandomForestClassifier(n_estimators=300, random_state=1),
    "kNN 7": KNeighborsClassifier(n_neighbors=7),
}


for name, model in models.items():

    pipe = Pipeline([
        ("select", SelectKBest(mutual_info_classif, k=200)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=150)),
        ("model", model)
    ])

    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv5,
        scoring={
            "accuracy": "accuracy",
            "f1": "f1_macro",
            "precision": "precision_macro",
            "recall": "recall_macro"
        }
    )

    print(
        name,
        "accuracy:", round(scores["test_accuracy"].mean(), 4),
        "f1:", round(scores["test_f1"].mean(), 4),
        "precision:", round(scores["test_precision"].mean(), 4),
        "recall:", round(scores["test_recall"].mean(), 4)
    )