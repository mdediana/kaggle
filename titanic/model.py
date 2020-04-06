import argparse
import sys
import logging
from itertools import product
import importlib

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier

MODEL_CLASSES = {
    'AdaBoost': 'sklearn.ensemble.AdaBoostClassifier',
    'RandomForest': 'sklearn.ensemble.RandomForestClassifier',
    'GradientBoosting': 'sklearn.ensemble.GradientBoostingClassifier',
    'KNeighbors': 'sklearn.neighbors.KNeighborsClassifier',
    'XGB': 'xgboost.XGBClassifier',
}
PARAM_GRIDS = {
    AdaBoostClassifier: dict(
        n_estimators=[10, 50, 100, 500, 1000, 5000],
        learning_rate=[0.01, 0.1, 0.5, 1],
    ),
    GradientBoostingClassifier: dict(
        # n_estimators=150,
        # learning_rate=0.05,
        # max_depth=3,
        # max_features='sqrt',
        # max_leaf_nodes=4,
        # ccp_alpha=0.01,
        # TODO: Try adjusting lambda, gamma and colsample to avoid overfitting
    ),
    RandomForestClassifier: dict(
        n_estimators=[10, 50, 100, 500],
        max_depth=[2, 3, 5, 10],
        max_features=['auto', None],
        ccp_alpha=[0, 0.01, 0.1, 0.5],
    ),
    XGBClassifier: dict(
        # For xgboost_params, see https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
        # https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
        # max_depth=[2, 3, 4, 6, 8, 10],
        learning_rate=[0.07, 0.05],
        n_estimators=[200, 500, 1000, 2000],
        # gamma=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        # min_child_weight=[1, 3, 5, 7],
        # subsample=[0.5, 0.75, 1.0],
        # colsample_bytree=[0.4, 0.6, 0.8, 1.0],
        # reg_alpha=[0, 0.1, 0.5, 1.0],
        # reg_lambda=[0.01, 0.1, 1.0],
    ),
    KNeighborsClassifier: dict(
        # n_neighbors=50,
        # algorithm='brute',
    ),
}
BEST_PARAMS = {
    AdaBoostClassifier: dict(
        # - Pclass, Sex, Age: n_estimators: 1000, learning_rate: 0.5
        # - Pclass, Sex, Age, SibSp, Parch: n_estimators: 50, learning_rate: 1
        # - Pclass, Sex, Age, SibSp, Parch, Fare, Embarked: n_estimators: 1000, learning_rate: 0.5
        n_estimators=50,
        learning_rate=1,
        random_state=0,
    ),
    GradientBoostingClassifier: dict(
        random_state=0,
    ),
    RandomForestClassifier: dict(
        n_estimators=10,
        max_depth=10,
        max_features='auto',
        ccp_alpha=0,
        random_state=0,
    ),
    XGBClassifier: dict(
        random_state=0,
    ),
    KNeighborsClassifier: dict(),
}
N_JOBS = -1  # Use all processors, particularly useful when param grid searching
COLUMNS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# COLUMNS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

logger = logging.getLogger(__name__)


def _read_csv(filename, split_X_y=True, columns=None):
    """Return X, y if training data or X, None if test data"""
    df = pd.read_csv(filename)
    df.set_index('PassengerId', inplace=True)
    # Set missing boy ages - better imputation of ages hurts accuracy (?)
    # childrens_age_median = df[df.Age < 18].Age.median()
    # mask = df.Name.str.contains('Master') & df.Age.isnull()
    # df.at[df[mask].index, 'Age'] = childrens_age_median
    # df['Relatives'] = df.SibSp + df.Parch
    # df['FarePerPerson'] = df.Fare / (df.Relatives + 1)
    if split_X_y and 'Survived' in df:
        X = df.drop(columns=['Survived'])
        y = df['Survived']
    else:
        X = df
        y = None
    if columns is not None:
        X = X[columns]
    return X, y


def _instantiate_model(model_name, use_best_params=True):
    module_name, class_name = MODEL_CLASSES[model_name].rsplit('.', maxsplit=1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    params = BEST_PARAMS[class_] if use_best_params else {}
    params['n_jobs'] = N_JOBS
    logger.info('Instantiating %s %s', class_name, params if params else '')
    return class_(**params)


def _column_transformer(columns, remainder='passthrough'):
    imputers = {
        'Embarked': make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder()),
        'Age': make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler()),
        'Pclass': OneHotEncoder(),
        'Sex': OneHotEncoder(),
        'Fare': SimpleImputer(strategy='median'),
        'SibSp': StandardScaler(),
        'Parch': StandardScaler(),
    }
    logger.info('Preparing transformers for columns: %s', columns.tolist())
    transformers = [(imputers[col], [col]) for col in columns]
    return make_column_transformer(*transformers, remainder=remainder)


def _search_params(X, y, model):
    column_transformer = _column_transformer(X.columns)
    param_grid = PARAM_GRIDS[type(model)]
    combinations = list(product(*param_grid.values()))
    logger.info('Search hyperparamaters, number of combinations: %d', len(combinations))
    param_grid = {'model__' + params: param_grid[params] for params in param_grid}
    pipeline = Pipeline([('column_transformer', column_transformer), ('model', model)])
    cv = GridSearchCV(pipeline, param_grid, scoring='accuracy')
    cv.fit(X, y)
    best_params = {param.replace('model__', ''): value for param, value in cv.best_params_.items()}
    logger.info('Best score: %s', cv.best_score_)
    logger.info('Best parameters: %s', best_params)
    # logger.info('Results: %s', cv.cv_results_)
    return cv.best_params_


def _train(X, y, model):
    column_transformer = _column_transformer(X.columns)
    pipeline = make_pipeline(column_transformer, model)
    pipeline.fit(X, y)
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    y_predict = pipeline.predict(X)
    score_predict = (y_predict == y).sum() / len(y)
    return pipeline, scores, score_predict


def _predict(pipeline, X):
    y = pipeline.predict(X)
    return pd.DataFrame(index=X.index, data={'Survived': y})


def _predict_by_family(training_set_file, test_set_file):
    # Solution based on https://www.kaggle.com/c/titanic/discussion/57447
    # Read data
    df_train, _ = _read_csv(training_set_file, split_X_y=False)
    df_test, _ = _read_csv(test_set_file)
    # Extract family survival
    df_train['Family'] = df_train['Name'].str.split(',', n=1, expand=True)[0]
    # If age is NA consider record as adult
    without_men = df_train[~((df_train['Sex'] == 'male') &
                             ((df_train['Age'] >= 16) | df_train['Age'].isnull()))]
    by_family = without_men.groupby('Family')
    def all_survived(g): return all(p == 1 for p in g)
    def all_died(g): return all(p == 0 for p in g)
    family_survived = by_family['Survived'].agg([('FamilySurvived', all_survived),
                                                 ('FamilyDied', all_died)])
    # Predict survival based on family
    df_test['Family'] = df_test['Name'].str.split(',', n=1, expand=True)[0]
    df_test = df_test.join(family_survived, on='Family', how='inner')  # Only records with family info
    df_master_survived = df_test[df_test['Name'].str.contains('Master') & df_test['FamilySurvived']]
    df_female_died = df_test[(df_test['Sex'] == 'female') & df_test['FamilyDied']]
    return pd.concat([pd.DataFrame({'Survived': 1}, index=df_master_survived.index),
                      pd.DataFrame({'Survived': 0}, index=df_female_died.index)])


def _predict_by_gender(X_test):
    y = X_test.copy()
    y['Survived'] = np.where(y['Sex'] == 'male', 0, 1)
    return y[['Survived']]


def _predict_knn(X_train, y_train, X_test=None):
    X_train_knn = X_train[['Pclass', 'Sex', 'Age']]
    # Prepare pipeline
    column_trans = _column_transformer(X_train_knn.columns)
    nca = NeighborhoodComponentsAnalysis(random_state=0)
    model = _instantiate_model('KNeighbors')
    pipeline = make_pipeline(column_trans, nca, model)
    # Train
    pipeline.fit(X_train_knn, y_train)
    scores = cross_val_score(pipeline, X_train_knn, y_train, cv=5, scoring='accuracy')
    logger.info('%s - Score: %f+/-%f %s', type(model).__name__, scores.mean(), scores.std(), scores)
    preds = _predict(pipeline, X_train_knn)
    X_train = X_train.join(preds, rsuffix='KNN_')
    # Predict
    if X_test is not None:
        X_test_knn = X_test[['Pclass', 'Sex', 'Age']]
        preds = _predict(pipeline, X_test_knn)
        X_test = X_test.join(preds, rsuffix='KNN_')
    return X_train, X_test


def search_params(training_set_file, algorithm):
    X, y = _read_csv(training_set_file, columns=COLUMNS)
    model = _instantiate_model(algorithm, use_best_params=False)
    _search_params(X, y, model)


def train(training_set_file, algorithm):
    X_train, y_train = _read_csv(training_set_file, columns=COLUMNS)
    model = _instantiate_model(algorithm, use_best_params=True)
    pipeline, scores, score_training = _train(X_train, y_train, model)
    logger.info('%s - Score cross validation: %f+/-%f %s - Score training set: %f',
                type(model).__name__, scores.mean(), scores.std(), scores, score_training)


def predict(training_set_file, test_set_file, algorithm):
    X_train, y_train = _read_csv(training_set_file, columns=COLUMNS)
    X_test, _ = _read_csv(test_set_file, columns=COLUMNS)
    # y = _predict_by_gender(X_test)
    model = _instantiate_model(algorithm, use_best_params=True)
    pipeline, scores, score_training = _train(X_train, y_train, model)
    logger.info('%s - Score cross validation: %f+/-%f %s - Score training set: %f',
                type(model).__name__, scores.mean(), scores.std(), scores, score_training)
    y = _predict(pipeline, X_test)
    y_family = _predict_by_family(training_set_file, test_set_file)
    y.update(y_family)
    y['Survived'] = y['Survived'].astype(int)  # Fix type after update https://github.com/pandas-dev/pandas/issues/4094
    y.to_csv(sys.stdout)


def predict_majority(training_set_file, test_set_file):
    """Deprecated"""
    X_train, y_train = _read_csv(training_set_file)
    X_test, _ = _read_csv(test_set_file)
    # X_train, X_test = _predict_knn(X_train, y_train, X_test)
    # Gradient boost
    model = _instantiate_model('GradientBoosting')
    pipeline, scores = _train(X_train, y_train, model)
    logger.info('%s - Score: %f+/-%f %s', type(model).__name__, scores.mean(), scores.std(), scores)
    preds_1 = _predict(pipeline, X_test)
    # Random Forest
    model = _instantiate_model('RandomForest')
    pipeline, scores = _train(X_train, y_train, model)
    logger.info('%s - Score: %f+/-%f %s', type(model).__name__, scores.mean(), scores.std(), scores)
    preds_2 = _predict(pipeline, X_test)
    # KNN
    _, X_test_knn = _predict_knn(X_train, y_train, X_test)
    preds_3 = X_test_knn[['Survived']].copy()
    preds_3.rename(columns={'Survived': 'Survived_knn'}, inplace=True)
    # Results
    preds = preds_1.join(preds_2, lsuffix='_gb', rsuffix='_rf').join(preds_3)
    preds['Survived'] = preds.mode(axis=1)
    # preds['Diff'] = (preds.Survived_gb == preds.Survived_rf) & (preds.Survived_rf == preds.Survived_knn)
    preds.drop(columns=['Survived_gb', 'Survived_rf', 'Survived_knn'], inplace=True)
    preds.to_csv(sys.stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='Command to run', choices=['train', 'predict', 'search-params'],
                        default='train')
    parser.add_argument('--training-set-file', help='Training set file', default='train.csv')
    parser.add_argument('--test-set-file', help='Test set file', default='test.csv')
    parser.add_argument('--algorithm', help='Algorithm to be used', choices=MODEL_CLASSES.keys(), required=True)
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if args.command == 'train':
        train(args.training_set_file, args.algorithm)
    elif args.command == 'predict':
        predict(args.training_set_file, args.test_set_file, args.algorithm)
    elif args.command == 'search-params':
        search_params(args.training_set_file, args.algorithm)
