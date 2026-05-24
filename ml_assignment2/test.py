"""
COMP30027 Project 2 - Task 1 & Task 2 
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torchvision import models, transforms
from PIL import Image

from sklearn.base import clone
from sklearn.model_selection import (train_test_split, StratifiedKFold,
    cross_val_score, cross_val_predict, GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)


from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, StackingClassifier,
)

RANDOM_STATE = 1

'''
Extract ResNt50 ImageNet features from raw image input
'''
def extract_imagenet_features(meta_df, base_dir, cache_path):
    if os.path.exists(cache_path):
        print('Loading cached ImageNet features:', cache_path)
        return pd.read_csv(cache_path)

    print('Extracting ImageNet ResNet50 features ...')
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = models.ResNet50_Weights.IMAGENET1K_V2   
    resnet  = models.resnet50(weights=weights)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    all_features, image_ids = [], []
    with torch.no_grad():
        for i, row in meta_df.iterrows():
            img = Image.open(os.path.join(base_dir, row['image_path']))\
                       .convert('RGB')
            x    = transform(img).unsqueeze(0).to(device)
            feat = resnet(x).cpu().numpy().flatten()
            all_features.append(feat)
            image_ids.append(row['image_id'])
            if (i + 1) % 200 == 0:
                print('processed', i + 1, '/', len(meta_df))

    arr = np.vstack(all_features)
    feat_df = pd.DataFrame(arr,
        columns=[f'imgnet_{i}' for i in range(arr.shape[1])])
    feat_df.insert(0, 'image_id', image_ids)
    feat_df.to_csv(cache_path, index=False)
    print('Saved ImageNet features:', cache_path)
    return feat_df


'''Load task data and create feature combinations'''
def load_task(base_dir):
    train_meta = pd.read_csv(os.path.join(base_dir, 'train_metadata.csv'))
    test_meta  = pd.read_csv(os.path.join(base_dir, 'test_metadata.csv'))
    color      = pd.read_csv(os.path.join(base_dir, 'color_histogram.csv'))
    hog        = pd.read_csv(os.path.join(base_dir, 'hog_pca.csv'))
    add        = pd.read_csv(os.path.join(base_dir, 'additional_features.csv'))
    class_map_path = os.path.join(base_dir, 'class_mapping.csv')
    class_map = pd.read_csv(class_map_path) if os.path.exists(class_map_path) else None

    # ImageNet features - use ResNet50 
    all_meta = pd.concat([
        train_meta[['image_id', 'image_path']],
        test_meta[['image_id', 'image_path']],
    ], ignore_index=True)
    imagenet = extract_imagenet_features(
        all_meta, base_dir,
        os.path.join(base_dir, 'imagenet_resnet50_features.csv'),
    )

    handcrafted = color.merge(hog, on='image_id').merge(add, on='image_id')
    feature_sets = {
        'color':            color,
        'hog':              hog,
        'add':              add,
        'handcrafted_all':  handcrafted,
        'imagenet':         imagenet,
        'imagenet+hog':     imagenet.merge(hog, on='image_id'),
        'imagenet+add':     imagenet.merge(add, on='image_id'),
        'imagenet+color':   imagenet.merge(color, on='image_id'),
        'imagenet+all':     imagenet.merge(handcrafted, on='image_id'),
    }
    return train_meta, test_meta, feature_sets, class_map



'''Create training labels, training features, and test features'''
def make_xy(train_meta, test_meta, features):
    train_df = train_meta.merge(features, on='image_id')
    test_df  = test_meta.merge(features,  on='image_id')

    drop_cols = ['image_id', 'image_path', 'class_id', 'class_name']
    X         = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y         = train_df['class_id']
    train_ids = train_df['image_id']

    X_test   = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    test_ids = test_df['image_id']
    return X, y, train_ids, X_test, test_ids



'''
Print class distribution to check imbalance 
(by count how many images are in each class)
'''
def report_class_balance(y, class_names):
    counts = y.value_counts().sort_index()
    total = counts.sum()
    print(f'\nClass distribution (n={total}):')
    for cid, n in counts.items():
        name = class_names[cid] if cid < len(class_names) else str(cid)
        print(f'  {name:<18}: {n:>4} ({n/total:.1%})')
    cv = counts.std() / counts.mean()
    print(f'Coefficient of variation: {cv:.3f}  '
          f'(>0.2 suggests using class_weight)')
    return cv


'''
Tests each feature set using Logistic Regression and 5-fold CV, 
then ranks feature sets by highest accuracy first, then by lowest standard deviation
'''
def compare_feature_subsets(train_meta, test_meta, feature_sets,
                             holdout_ids, cv5):
    base = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=5000,
                                    random_state=RANDOM_STATE)),
    ])
    rows = []
    for name, features in feature_sets.items():
        X, y, train_ids, _, _ = make_xy(train_meta, test_meta, features)
        train_mask = ~train_ids.isin(holdout_ids)
        X_tr, y_tr = X[train_mask], y[train_mask]
        scores = cross_val_score(base, X_tr, y_tr, cv=cv5,
                                 scoring='accuracy', n_jobs=-1)
        rows.append({
            'feature_set': name, 'n_features': X_tr.shape[1],
            'cv_acc_mean': scores.mean(), 'cv_acc_std': scores.std(),
        })
        print(f'  {name:<16} ({X_tr.shape[1]:>4} feats): '
              f'{scores.mean():.4f} +/- {scores.std():.4f}')
    return (pd.DataFrame(rows)
              .sort_values('cv_acc_mean', ascending=False)
              .reset_index(drop=True))


''' 
Creates a learning : 
scaling → optional feature selection → classifier
'''
def make_pipeline(clf, use_fs):
    steps = [('scaler', StandardScaler())]
    if use_fs:
        # k is tuned via grid search; placeholder k=50 will be replaced
        steps.append(('fs', SelectKBest(mutual_info_classif, k=50)))
    steps.append(('clf', clf))
    return Pipeline(steps)


'''
Tries different hyperparameters for models:
Logistic Regression, SVM, kNN, Random Forest, Voting, and Stacking
'''
def tune_models(X_train, y_train, cv5, use_fs=False):
    tuned       = {}    # best version of each model
    cv_scores   = {}    # validation accuracy from tuning
    needs_score = []    # models that still need CV accuracy calculated later

    # Clip to the available number of features so we never request k > d.
    n_feat = X_train.shape[1]
    k_options = [k for k in [50, 100, 200, 500] if k < n_feat]
    if not k_options:                       # very low-dim data
        k_options = [max(1, n_feat // 2)]
    fs_grid = {'fs__k': k_options} if use_fs else {}

    # ---- 0-R baseline ----
    # always predict the most common class
    tuned['0-R baseline'] = DummyClassifier(strategy='most_frequent',
                                             random_state=RANDOM_STATE)
    needs_score.append('0-R baseline')

    # ---- Gaussian NB ----
    # Naive Bayes classifier - only continous data
    gnb_pipe = make_pipeline(GaussianNB(), use_fs)
    if use_fs:
        gnb_gs = GridSearchCV(gnb_pipe, fs_grid, cv=cv5,
                              scoring='accuracy', n_jobs=-1)
        gnb_gs.fit(X_train, y_train)
        tuned['Gaussian NB']      = gnb_gs.best_estimator_
        cv_scores['Gaussian NB']  = (gnb_gs.best_score_,
                                     gnb_gs.cv_results_['std_test_score'][gnb_gs.best_index_])
        print(f'  Gaussian NB         best={gnb_gs.best_params_}  '
              f'CV acc={gnb_gs.best_score_:.4f}')
    else:
        tuned['Gaussian NB'] = gnb_pipe
        needs_score.append('Gaussian NB')

    # ---- Models with grids ----
    grids = [
        ('Logistic Regression',
         LogisticRegression(max_iter=5000, solver='lbfgs',
                            random_state=RANDOM_STATE),
         {'clf__C': [0.5, 2.0, 10.0]}),

        ('SVM RBF',
         SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
         {'clf__C':     [1.0, 3.0, 10.0],
          'clf__gamma': ['scale', 0.01, 0.001]}),    # [Fix 1] gamma now tuned

        ('kNN',
         KNeighborsClassifier(),
         {'clf__n_neighbors': [5, 9, 15]}),
    ]

    for name, clf, model_grid in grids:
        pipe = make_pipeline(clf, use_fs)
        grid = {**model_grid, **fs_grid}
        print(f'  Tuning {name}...')
        gs = GridSearchCV(pipe, grid, cv=cv5,
                          scoring='accuracy', n_jobs=-1)
        gs.fit(X_train, y_train)
        tuned[name]     = gs.best_estimator_
        cv_scores[name] = (gs.best_score_,
                           gs.cv_results_['std_test_score'][gs.best_index_])
        print(f'    best={gs.best_params_}  CV acc={gs.best_score_:.4f}')

    # ---- Random Forest (trees don't need scaling/FS) ----
    print('  Tuning Random Forest...')
    rf_gs = GridSearchCV(
        RandomForestClassifier(max_features='sqrt',
                                random_state=RANDOM_STATE, n_jobs=-1),
        {'n_estimators': [300, 500]},
        cv=cv5, scoring='accuracy', n_jobs=-1,
    )
    rf_gs.fit(X_train, y_train)
    tuned['Random Forest']     = rf_gs.best_estimator_
    cv_scores['Random Forest'] = (rf_gs.best_score_,
                                  rf_gs.cv_results_['std_test_score'][rf_gs.best_index_])
    print(f'    best={rf_gs.best_params_}  CV acc={rf_gs.best_score_:.4f}')

    # ---- Soft Voting ensemble (base models pre-tuned) ----
    tuned['Soft Voting'] = VotingClassifier(
        estimators=[
            ('lr',  clone(tuned['Logistic Regression'])),
            ('svm', clone(tuned['SVM RBF'])),
            ('rf',  clone(tuned['Random Forest'])),
        ],
        voting='soft', n_jobs=-1,
    )
    needs_score.append('Soft Voting')

    # ---- Stacking ensemble - tune meta-learner C  ----
    # tuned manually with cross_val_score, not GridSearchCV.
    print('  Tuning Stacking meta-learner C...')
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    best_C, best_score, best_std = None, -1.0, 0.0
    for C in [0.5, 1.0, 5.0]:
        cand = StackingClassifier(
            estimators=[
                ('lr',  clone(tuned['Logistic Regression'])),
                ('svm', clone(tuned['SVM RBF'])),
                ('rf',  clone(tuned['Random Forest'])),
            ],
            final_estimator=LogisticRegression(C=C, max_iter=5000,
                                                random_state=RANDOM_STATE),
            cv=3, stack_method='predict_proba', n_jobs=-1,
        )
        scores = cross_val_score(cand, X_train, y_train, cv=cv3,
                                 scoring='accuracy', n_jobs=-1)
        if scores.mean() > best_score:
            best_C, best_score, best_std = C, scores.mean(), scores.std()
    tuned['Stacking'] = StackingClassifier(
        estimators=[
            ('lr',  clone(tuned['Logistic Regression'])),
            ('svm', clone(tuned['SVM RBF'])),
            ('rf',  clone(tuned['Random Forest'])),
        ],
        final_estimator=LogisticRegression(C=best_C, max_iter=5000,
                                            random_state=RANDOM_STATE),
        cv=3, stack_method='predict_proba', n_jobs=-1,
    )
    cv_scores['Stacking'] = (best_score, best_std)
    print(f'    best C={best_C}  CV acc={best_score:.4f}')

    return tuned, cv_scores, needs_score


'''
Calculate CV accuracy for models that did not go through GridSearchCV:
0-R baseline, Voting
'''
def fill_remaining_cv_scores(tuned, cv_scores, needs_score,
                              X_train, y_train, cv5):
    for name in needs_score:
        scores = cross_val_score(tuned[name], X_train, y_train, cv=cv5,
                                 scoring='accuracy', n_jobs=-1)
        cv_scores[name] = (scores.mean(), scores.std())
    return cv_scores


'''
Perform error analysis using out-of-fold predictions:
confusion matrix, confused class pairs, per-class accuracy, and misclassified examples
'''
def evaluate_holdout(tuned_models, X_train, y_train, X_holdout, y_holdout):
    """Unbiased final evaluation on the 20% holdout."""
    holdout_results = {}
    val_predictions = {}
    for name, model in tuned_models.items():
        m = clone(model)
        m.fit(X_train, y_train)
        pred = m.predict(X_holdout)
        val_predictions[name] = pred
        holdout_results[name] = {
            'accuracy':    accuracy_score(y_holdout, pred),
            'macro_f1':    f1_score(y_holdout, pred, average='macro',    zero_division=0),
            'weighted_f1': f1_score(y_holdout, pred, average='weighted', zero_division=0),
        }
    return holdout_results, val_predictions


'''
Trains each tuned model on 80% training data, 
then evaluates it on the untouched 20% holdout set
'''
def rich_error_analysis(best_model, X_train, y_train, train_ids,
                        train_meta, class_names, cv5, out_prefix):
    print('\nComputing out-of-fold predictions on 80% training data ...')
    oof_pred = cross_val_predict(best_model, X_train, y_train,
                                 cv=cv5, n_jobs=-1)

    y_true = np.asarray(y_train)
    y_pred = np.asarray(oof_pred)
    ids    = np.asarray(train_ids)

    labels     = sorted(set(y_true) | set(y_pred))
    cm         = confusion_matrix(y_true, y_pred, labels=labels)
    names_used = [class_names[c] for c in labels]

    print(f'OOF accuracy: {accuracy_score(y_true, y_pred):.4f}')
    print(f'OOF macro-F1: {f1_score(y_true, y_pred, average="macro"):.4f}')

    # Confusion matrix CSV + heatmap PNG
    pd.DataFrame(cm, index=names_used, columns=names_used).to_csv(
        f'{out_prefix}_confusion_matrix.csv')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=names_used, yticklabels=names_used)
    plt.title(f'{out_prefix} - Out-of-Fold Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_confusion_matrix.png', dpi=200,
                bbox_inches='tight')
    plt.close()

    # Top confused pairs
    rows = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                rows.append({
                    'true_class':        names_used[i],
                    'predicted_class':   names_used[j],
                    'count':             int(cm[i, j]),
                    'pct_of_true_class': cm[i, j] / max(cm[i].sum(), 1),
                })
    confused_df = (pd.DataFrame(rows)
                     .sort_values('count', ascending=False)
                     .reset_index(drop=True))
    confused_df.to_csv(f'{out_prefix}_confused_pairs.csv', index=False)
    print(f'\nTop 10 confused class pairs:')
    print(confused_df.head(10).to_string(index=False))

    # Per-class accuracy
    per_class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    pc_df = pd.DataFrame({
        'class':    names_used,
        'accuracy': per_class_acc,
        'support':  cm.sum(axis=1),
    })
    pc_df.to_csv(f'{out_prefix}_per_class_accuracy.csv', index=False)
    print(f'\nPer-class accuracy:')
    print(pc_df.to_string(index=False))

    # Misclassified examples with image_path
    mis_rows = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            mis_rows.append({
                'image_id':        ids[i],
                'true_class':      class_names[y_true[i]],
                'predicted_class': class_names[y_pred[i]],
            })
    mis_df = pd.DataFrame(mis_rows)
    if 'image_path' in train_meta.columns:
        mis_df = mis_df.merge(train_meta[['image_id', 'image_path']],
                              on='image_id', how='left')
    mis_df.to_csv(f'{out_prefix}_misclassified.csv', index=False)
    print(f'\nSaved {len(mis_df)} misclassified examples to '
          f'{out_prefix}_misclassified.csv')


'''
Trains the best model on all labelled training data,
then predict labels for unlabelled test data
'''
def save_prediction(model, X_full, y_full, X_test, test_ids, out_path):
    m = clone(model)
    m.fit(X_full, y_full)
    pred = m.predict(X_test)
    sub = pd.DataFrame({'image_id': test_ids.values,
                        'class_id': pred.astype(int)})
    sub.to_csv(out_path, index=False)
    print(f'Saved Kaggle submission: {out_path} ({len(sub)} rows)')


'''
Runs the whole workflow for one task: 
load data, check balance, split holdout, compare features, 
tune models, evaluate, analyse errors, save submission
'''
def run_task(task_name, base_dir, use_fs=False):
    print('\n' + '=' * 80)
    print(f' {task_name}: data from {base_dir}')
    print('=' * 80)

    train_meta, test_meta, feature_sets, class_map = load_task(base_dir)
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Class names
    if class_map is not None:
        class_names = (class_map.sort_values('class_id')
                                ['class_name'].tolist())
    else:
        class_names = [f'class_{i}' for i
                       in sorted(train_meta['class_id'].unique())]

    # ---- Step 0a: class balance check ----
    print('\n[Step 0a] Class balance check')
    cv_balance = report_class_balance(train_meta['class_id'], class_names)
    if cv_balance > 0.2:
        print('WARNING: data is imbalanced; consider class_weight="balanced"')
    else:
        print('Data is roughly balanced; default class weighting used.')

    # ---- Step 0b: holdout split based on image_ids ----
    train_y       = train_meta['class_id'].values
    train_ids_all = train_meta['image_id'].values
    _, holdout_ids_arr = train_test_split(
        train_ids_all, test_size=0.20, stratify=train_y,
        random_state=RANDOM_STATE,
    )
    holdout_ids = set(holdout_ids_arr)
    print(f'\nReserved {len(holdout_ids)} samples as holdout (stratified).')

    # ---- Step 1: feature subset comparison ----
    print('\n[Step 1] Feature subset comparison (reporting/methodology only)')
    fs_results = compare_feature_subsets(train_meta, test_meta,
                                          feature_sets, holdout_ids, cv5)
    fs_results.to_csv(f'{task_name}_feature_subsets.csv', index=False)
    print(f'\nBest feature subset (informational): '
          f'{fs_results.iloc[0]["feature_set"]}')
    # choose highest accuracy first, then smallest std
    fs_results = fs_results.sort_values(
        ['cv_acc_mean', 'cv_acc_std'],
        ascending=[False, True]
    ).reset_index(drop=True)

    selected_feature_name = fs_results.iloc[0]['feature_set']

    print(f'\nSelected feature set: {selected_feature_name}')
    print(fs_results.head())


    # ---- Step 2: use selected feature set ----
    print(f'\n[Step 2] Final feature set: {selected_feature_name}')
    features = feature_sets[selected_feature_name]
    X, y, train_ids, X_test, test_ids = make_xy(train_meta, test_meta, features)
    holdout_mask = train_ids.isin(holdout_ids)
    X_train, X_holdout = X[~holdout_mask], X[holdout_mask]
    y_train, y_holdout = y[~holdout_mask], y[holdout_mask]
    ids_train          = train_ids[~holdout_mask]
    print(f'Training: X={X_train.shape}, holdout: X={X_holdout.shape}')

    # ---- Step 3: GridSearchCV-based model tuning ----
    print(f'\n[Step 3] Hyperparameter tuning via GridSearchCV (inner 5-fold)')
    if use_fs:
        print(f'  MI feature selection ENABLED (k tuned in grid)')
    tuned_models, cv_scores, needs_score = tune_models(
        X_train, y_train, cv5, use_fs=use_fs)
    cv_scores = fill_remaining_cv_scores(
        tuned_models, cv_scores, needs_score, X_train, y_train, cv5)

    # ---- Step 4: print CV summary ----
    print('\n' + '=' * 70)
    print(' CV accuracy on training portion (mean +/- std)')
    print('=' * 70)
    for name, (m, s) in cv_scores.items():
        print(f'  {name:<22}: {m:.4f} +/- {s:.4f}')

    # ---- Step 5: holdout evaluation ----
    print('\n' + '=' * 70)
    print(' Holdout evaluation (untouched during selection - unbiased)')
    print('=' * 70)
    holdout_results, val_predictions = evaluate_holdout(
        tuned_models, X_train, y_train, X_holdout, y_holdout)
    for name, m in holdout_results.items():
        print(f'  {name:<22}: acc={m["accuracy"]:.4f}  '
              f'macro-F1={m["macro_f1"]:.4f}  '
              f'weighted-F1={m["weighted_f1"]:.4f}')

    # Save summary tables
    pd.DataFrame([
        {'model': k, 'cv_acc_mean': v[0], 'cv_acc_std': v[1]}
        for k, v in cv_scores.items()
    ]).to_csv(f'{task_name}_cv_summary.csv', index=False)

    pd.DataFrame([
        {'model': k, **v} for k, v in holdout_results.items()
    ]).to_csv(f'{task_name}_holdout_summary.csv', index=False)

    # ---- Step 6: rich error analysis (on 80% training) ----
    best_cv = max(cv_scores, key=lambda k: cv_scores[k][0])
    print(f'\n[Step 6] Rich error analysis on best CV model: {best_cv}')
    rich_error_analysis(
        tuned_models[best_cv], X_train, y_train, ids_train,
        train_meta, class_names, cv5, task_name,
    )

    # ---- Step 7: submission ----
    print(f'\n[Step 7] Generating Kaggle submission with {best_cv}')
    save_prediction(
        tuned_models[best_cv],
        X, y, X_test, test_ids,
        f'{task_name}_submission.csv',
    )

    return cv_scores, holdout_results


'''main'''
if __name__ == '__main__':
    task1_dir = 'task1_data'
    task2_dir = 'task2_data'

    # Task 1: ~3,750 training images -> plenty of data, no MI filter
    run_task('task1', task1_dir, use_fs=False)

    # Task 2: ~417 training images, ~2,267 dims with ResNet50 ->
    # high feature/sample ratio -> apply MI filter feature selection
    # (k tuned within GridSearchCV)
    run_task('task2', task2_dir, use_fs=True)