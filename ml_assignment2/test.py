import os
import numpy as np
import pandas as pd

import torch
from torchvision import models, transforms

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from itertools import combinations

RANDOM_STATE = 1



def extract_imagenet_features(meta_df, base_dir, cache_path):
    """
    Extract ImageNet pretrained ResNet18 features.
    Uses cache so the slow CNN extraction only runs once.
    """
    if os.path.exists(cache_path):
        print('Loading cached ImageNet features:', cache_path)
        return pd.read_csv(cache_path)

    print('Extracting ImageNet ResNet18 features. This may take a few minutes...')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    resnet = models.resnet18(weights=weights)
    resnet.fc = torch.nn.Identity()     # remove final classification layer
    resnet = resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    all_features = []
    image_ids = []

    with torch.no_grad():
        for i, row in meta_df.iterrows():
            img_path = os.path.join(base_dir, row['image_path'])
            img = Image.open(img_path).convert('RGB')
            x = transform(img).unsqueeze(0).to(device)
            feat = resnet(x).cpu().numpy().flatten()

            all_features.append(feat)
            image_ids.append(row['image_id'])

            if (i + 1) % 200 == 0:
                print('processed', i + 1, '/', len(meta_df))

    arr = np.vstack(all_features)
    feat_df = pd.DataFrame(arr, columns=[f'imgnet_{i}' for i in range(arr.shape[1])])
    feat_df.insert(0, 'image_id', image_ids)
    feat_df.to_csv(cache_path, index=False)
    print('Saved ImageNet features:', cache_path)
    return feat_df


def load_task(base_dir):
    print("Loading from:", base_dir)
    print("Files:", os.listdir(base_dir))
    train_meta = pd.read_csv(os.path.join(base_dir, 'train_metadata.csv'))
    test_meta = pd.read_csv(os.path.join(base_dir, 'test_metadata.csv'))
    color = pd.read_csv(os.path.join(base_dir, 'color_histogram.csv'))
    hog = pd.read_csv(os.path.join(base_dir, 'hog_pca.csv'))
    add = pd.read_csv(os.path.join(base_dir, 'additional_features.csv'))

    class_map_path = os.path.join(base_dir, 'class_mapping.csv')
    if os.path.exists(class_map_path):
        class_map = pd.read_csv(class_map_path)
        print("Loaded class_mapping.csv")
    else:
        class_map = None
        print("No class_mapping.csv in this task")


    # ImageNet ResNet18 features from raw images
    all_meta = pd.concat(
        [train_meta[['image_id', 'image_path']],
         test_meta[['image_id', 'image_path']]],
        ignore_index=True
    )
    imagenet = extract_imagenet_features(
        all_meta,
        base_dir,
        os.path.join(base_dir, 'imagenet_resnet18_features.csv')
    )

    base_features = {
        'color': color,
        'hog': hog,
        'add': add,
        'imagenet': imagenet,
    }

    feature_sets = {}

    for r in range(1, len(base_features) + 1):
        for names in combinations(base_features.keys(), r):
            merged = base_features[names[0]]
            for name in names[1:]:
                merged = merged.merge(base_features[name], on='image_id')
            feature_sets['_'.join(names)] = merged
    
    return train_meta, test_meta, feature_sets, class_map


def make_xy(train_meta, test_meta, features):
    train_df = train_meta.merge(features, on='image_id')
    test_df = test_meta.merge(features, on='image_id')

    drop_cols = ['image_id', 'image_path', 'class_id', 'class_name']
    X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y = train_df['class_id']
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    return X, y, X_test, test_df['image_id']


def get_models():
    return {
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=2, max_iter=5000, class_weight='balanced', random_state=RANDOM_STATE))
        ]),
        'linear_svm': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LinearSVC(C=0.3, class_weight='balanced', max_iter=20000, random_state=RANDOM_STATE))
        ]),
        'rbf_svm': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(C=3, gamma='scale', kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE))
        ]),
        'random_forest': RandomForestClassifier(
            n_estimators=500, max_features='sqrt', class_weight='balanced_subsample',
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=500, max_features='sqrt', class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def compare_feature_sets(train_meta, test_meta, feature_sets):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fs_name, features in feature_sets.items():
        X, y, _, _ = make_xy(train_meta, test_meta, features)
        for model_name, model in get_models().items():
            scores = cross_validate(
                model, X, y, cv=cv,
                scoring=['accuracy', 'f1_macro'],
                n_jobs=-1
            )
            rows.append({
                'feature_set': fs_name,
                'model': model_name,
                'n_features': X.shape[1],
                'acc_mean': scores['test_accuracy'].mean(),
                'acc_std': scores['test_accuracy'].std(),
                'f1_mean': scores['test_f1_macro'].mean(),
                'f1_std': scores['test_f1_macro'].std(),
            })
    return pd.DataFrame(rows).sort_values(['f1_mean', 'acc_mean'], ascending=False).reset_index(drop=True)


def error_analysis(train_meta, test_meta, features, model, task_name, out_prefix, class_map=None):
    X, y, _, _ = make_xy(train_meta, test_meta, features)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average='macro')

    labels = sorted(y.unique())

    # Use real class names if class_mapping.csv exists
    if class_map is not None:
        id_col = 'class_id'
        name_col = [c for c in class_map.columns if c != 'class_id'][0]
        id_to_name = dict(zip(class_map[id_col], class_map[name_col]))
        target_names = [id_to_name[i] for i in labels]

    else:
        target_names = [str(i) for i in labels]

    print('CV accuracy:', round(acc, 4))
    print('CV macro F1:', round(f1, 4))

    print('\nClass labels:')
    for class_id, name in zip(labels, target_names):
        print(class_id, '=', name)

    report = classification_report(
        y,
        pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )

    cm = confusion_matrix(y, pred, labels=labels)

    pd.DataFrame(report).T.to_csv(
        f'{out_prefix}_classification_report.csv'
    )

    pd.DataFrame(
        cm,
        index=target_names,
        columns=target_names
    ).to_csv(f'{out_prefix}_confusion_matrix.csv')

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names
    )

    plt.title(f'{task_name} Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_confusion_matrix.png', dpi=300)
    plt.close()


def save_prediction(train_meta, test_meta, features, model, out_path):
    X, y, X_test, test_ids = make_xy(train_meta, test_meta, features)
    final_model = clone(model)
    final_model.fit(X, y)
    pred = final_model.predict(X_test)
    sub = pd.DataFrame({'image_id': test_ids, 'class_id': pred})
    sub.to_csv(out_path, index=False)
    print('Saved:', out_path)


def run_task(task_name, base_dir):
    print('\n' + '=' * 80)
    print(task_name)
    print('=' * 80)
    train_meta, test_meta, feature_sets, class_map = load_task(base_dir)

    results = compare_feature_sets(train_meta, test_meta, feature_sets)
    print(results.head(15))
    results.to_csv(f'{task_name}_cv_results.csv', index=False)

    best = results.iloc[0]
    best_features = feature_sets[best['feature_set']]
    best_model = get_models()[best['model']]

    print('\nBest:', best['feature_set'], '+', best['model'])
    error_analysis(train_meta, test_meta, best_features, best_model, task_name, task_name, class_map)
    save_prediction(train_meta, test_meta, best_features, best_model, f'{task_name}_submission.csv')

    return results


def save_confused_pairs(cm, out_path):
    rows = []
    for true_label in range(cm.shape[0]):
        for pred_label in range(cm.shape[1]):
            if true_label != pred_label and cm[true_label, pred_label] > 0:
                rows.append({
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'count': cm[true_label, pred_label]
                })

    pd.DataFrame(rows).sort_values('count', ascending=False).to_csv(out_path, index=False)


if __name__ == '__main__':
    # Change these two paths if your folders are somewhere else.
    task1_dir = 'task1_data'
    task2_dir = 'task2_data'

    run_task('task1', task1_dir)
    # run_task('task2', task2_dir)
