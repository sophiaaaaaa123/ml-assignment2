# COMP30027 Project 2 - Image Classification

This project implements machine learning pipelines for two image
classification tasks from COMP30027 Machine Learning.

------------------------------------------------------------
TASKS
------------------------------------------------------------

Task 1:
Coarse-grained animal classification (10 classes)

Task 2:
Fine-grained bird species classification (10 bird species)

The project compares handcrafted features and ImageNet pretrained
ResNet50 features across multiple classifiers.

------------------------------------------------------------
FEATURES USED
------------------------------------------------------------

Provided features:
- Color histogram
- HOG PCA features
- Additional numeric features

Additional engineered features:
- ImageNet pretrained ResNet50 embeddings

Feature combinations tested:
- color
- hog
- add
- handcrafted_all
- imagenet
- imagenet+hog
- imagenet+add
- imagenet+color
- imagenet+all

------------------------------------------------------------
MODELS USED
------------------------------------------------------------

- 0-R Baseline
- Gaussian Naive Bayes
- Logistic Regression
- SVM (RBF kernel)
- k-Nearest Neighbours
- Random Forest
- Soft Voting Ensemble
- Stacking Ensemble

------------------------------------------------------------
METHODOLOGY
------------------------------------------------------------

1. Load metadata and feature CSV files
2. Extract ImageNet ResNet50 features
3. Compare feature subsets using Logistic Regression + 5-fold CV
4. Select best feature set
5. Split data into:
   - 80% training
   - 20% holdout validation
6. Tune models using GridSearchCV
7. Evaluate models using:
   - Accuracy
   - Macro F1-score
   - Weighted F1-score
8. Perform error analysis:
   - Confusion matrix
   - Per-class accuracy
   - Misclassified examples
9. Train final model on full data
10. Generate Kaggle submission CSV

------------------------------------------------------------
FEATURE SELECTION
------------------------------------------------------------

Task 1:
No feature selection used because dataset is large.

Task 2:
Mutual Information feature selection (SelectKBest) used because:
- small dataset size
- high feature dimensionality

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------

Place these folders in the project directory:

- task1_data/
- task2_data/

Run:

python test.py

------------------------------------------------------------
OUTPUT FILES
------------------------------------------------------------

The program generates:

- task1_submission.csv
- task2_submission.csv
- task1_cv_summary.csv
- task2_cv_summary.csv
- confusion matrix CSV files
- confusion matrix PNG files
- per-class accuracy CSV files
- misclassified example CSV files

------------------------------------------------------------
LIBRARIES
------------------------------------------------------------

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- torch
- torchvision
- PIL