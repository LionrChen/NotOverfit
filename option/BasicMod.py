#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ChenSir'
__mtime__ = '2019/7/12'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏┓   ┏┓
            ┏┛┻━━━┛┻┓
            ┃       ┃
            ┃ ┳┛ ┗┳ ┃
            ┃   ┻   ┃
            ┗━┓   ┏━┛
              ┃   ┗━━━┓
              ┃神兽保佑 ┣┓
              ┃ 永无BUG！ ┏┛
              ┗┓┓┏━━┳┓┏┛
               ┃┫┫  ┃┫┫
               ┗┻┛  ┗┻┛
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, \
    RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import linear_model, model_selection
from catboost import CatBoostClassifier

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

X_train = train.drop(['id', 'target'], axis=1)
y_train = train['target']
X_test = test.drop(['id'], axis=1)

n_fold = 20
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
repeat_folds = RepeatedStratifiedKFold(n_splits=n_fold, n_repeats=20, random_state=42)

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)


def train_model(X, X_test, y, params, folds=folds, model_type="lgb", plot_feature_importance=False, averaging='usual',
                model=None):
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        if model_type is 'lgb':
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid)

            model = lgb.train(params, train_data, num_boost_round=2000, valid_sets=[train_data, valid_data],
                              verbose_eval=500, early_stopping_rounds=200)
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        if model_type is 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_train.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_train.columns)

            watch_list = [(train_data, 'train'), (valid_data, 'valid')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watch_list, early_stopping_rounds=200,
                              verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_valid.columns),
                                         ntree_limit=model.best_iteration)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_iteration)

        if model_type is 'cat':
            model = CatBoostClassifier(iterations=20000, eval_metric="AUC", **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
                      verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        if model_type is 'sklearn':
            model = model
            model.fit(X_train, y_train)
            y_pred_valid = model.predict(X_valid).reshape(-1, )
            score = roc_auc_score(y_valid, y_pred_valid)
            y_pred = model.predict_proba(X_test)[:, 1]

        if model_type is 'glm':
            model = sm.GLM(y_train, X_train, family=sm.families.Binomial())
            model_results = model.fit()
            model_results.predict(X_test)
            y_pred_valid = model_results.predict(X_valid).reshape(-1, )
            score = roc_auc_score(y_valid, y_pred_valid)
            y_pred = model_results.predict(X_test)

        oof[valid_index] = y_pred_valid.reshape(-1, )
        scores.append(roc_auc_score(y_valid, y_pred_valid))

        if averaging is 'usual':
            prediction += y_pred
        elif averaging is 'rank':
            prediction += pd.Series(y_pred).rank().values

        if model_type is 'lgb':
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance,fold_importance],axis=0)
    prediction /= n_fold

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    if model_type is 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)')

            return oof, prediction, feature_importance
        return oof, prediction, scores

    else:
        return oof, prediction, scores


LR_Mod = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')
oof_LR, prediction_LR, scores = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=LR_Mod)
