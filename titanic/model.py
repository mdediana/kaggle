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
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier,
                              ExtraTreesClassifier, VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier

CLF_CLASSES = {
    'AdaBoost': 'sklearn.ensemble.AdaBoostClassifier',
    'RandomForest': 'sklearn.ensemble.RandomForestClassifier',
    'ExtraTrees': 'sklearn.ensemble.ExtraTreesClassifier',
    'GradientBoosting': 'sklearn.ensemble.GradientBoostingClassifier',
    'KNeighbors': 'sklearn.neighbors.KNeighborsClassifier',
    'XGB': 'xgboost.XGBClassifier',
}
N_JOBS = -1  # Use all processors, particularly useful when param grid searching
PARAM_GRIDS = {
    AdaBoostClassifier: dict(
        n_estimators=[10, 50, 100, 500, 1000, 5000],
        learning_rate=[0.01, 0.1, 0.5, 1],
    ),
    GradientBoostingClassifier: dict(
        n_estimators=[100, 500, 1000],
        learning_rate=[0.1, 0.5, 0.7, 1],
        subsample=[0.25, 0.5, 1],
        max_depth=[2, 3],
        # Regularization param fixed after manual tests comparing cross val and training set scores
        min_samples_leaf=[0.2],
    ),
    RandomForestClassifier: dict(
        n_estimators=[10, 50, 100],
        max_depth=[None, 10, 20],
        # max_features=['auto', None],
        max_features=[None],
        # Regularization param fixed after manual tests comparing cross val and training set scores
        min_samples_leaf=[0.05],
        n_jobs=[N_JOBS],
    ),
    ExtraTreesClassifier: dict(
        n_estimators=[10, 50, 100],
        max_depth=[None, 10, 20],
        # max_features=['auto', None],
        max_features=[None],
        # Regularization param fixed after manual tests comparing cross val and training set scores
        min_samples_leaf=[0.05],
        n_jobs=[N_JOBS],
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
        n_neighbors=[5, 10, 100],
        weights=['uniform', 'distance'],
        metric=['minkowski', 'euclidean'],
        p=[1, 2],
        n_jobs=[N_JOBS],
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
        n_estimators=500,
        learning_rate=0.5,
        subsample=1,
        max_depth=2,
        # Regularization params
        # min_samples_split=0.1,
        min_samples_leaf=0.2,
        random_state=0,
    ),
    RandomForestClassifier: dict(
        n_estimators=50,
        max_depth=None,
        max_features=None,
        # Regularization params
        # min_samples_split=0.1,
        min_samples_leaf=0.05,
        random_state=0,
    ),
    ExtraTreesClassifier: dict(
        n_estimators=10,
        max_depth=None,
        max_features=None,
        # Regularization params
        # min_samples_split=0.1,
        min_samples_leaf=0.05,
        random_state=0,
    ),
    XGBClassifier: dict(
        random_state=0,
    ),
    KNeighborsClassifier: dict(
        n_neighbors=10,  # Overfits for many different values
        weights='distance',
        metric='minkowski',
        p=1,
    ),
}
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


def _instantiate_clf(algorithms, use_best_params=True, voting='soft'):
    clf_names = algorithms.split(',')
    logger.info('Number of estimators: %d', len(clf_names))
    estimators = []  # Contains (name, clf) tuples for the VotingClassifier
    for clf_name in clf_names:
        module_name, class_name = CLF_CLASSES[clf_name].rsplit('.', maxsplit=1)
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        params = BEST_PARAMS[class_] if use_best_params else {}
        logger.info('Instantiating %s %s', class_name, params if params else '')
        estimators.append((clf_name, class_(**params)))
    if len(estimators) == 1:
        clf = estimators[0][1]
    else:
        logger.info('Creating voting estimator: %s', voting)
        clf = VotingClassifier(estimators=estimators, voting=voting)
        estimators = [(estimator.__class__.__name__, estimator)
                      for estimator in estimators]
    return clf


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


def _train(X, y, clf):
    column_transformer = _column_transformer(X.columns)
    pipeline = make_pipeline(column_transformer, clf)
    pipeline.fit(X, y)
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    y_predict = pipeline.predict(X)
    score_predict = (y_predict == y).sum() / len(y)
    logger.info('%s - Score cross validation: %f+/-%f %s - Score training set: %f (diff: %f)',
                type(clf).__name__, scores.mean(), scores.std(), scores, score_predict,
                score_predict - scores.mean())
    return pipeline


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
    model = _instantiate_clf('KNeighbors')
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
    model = _instantiate_clf(algorithm, use_best_params=False)
    _search_params(X, y, model)


def train(training_set_file, algorithms):
    X_train, y_train = _read_csv(training_set_file, columns=COLUMNS)
    clf = _instantiate_clf(algorithms, use_best_params=True)
    _train(X_train, y_train, clf)


def predict(training_set_file, test_set_file, algorithms, voting='soft'):
    X_train, y_train = _read_csv(training_set_file, columns=COLUMNS)
    X_test, _ = _read_csv(test_set_file, columns=COLUMNS)
    # y = _predict_by_gender(X_test)
    clf = _instantiate_clf(algorithms, use_best_params=True)
    pipeline = _train(X_train, y_train, clf)
    y = _predict(pipeline, X_test)
    y_family = _predict_by_family(training_set_file, test_set_file)
    y.update(y_family)
    y['Survived'] = y['Survived'].astype(int)  # Fix type after update https://github.com/pandas-dev/pandas/issues/4094
    y.to_csv(sys.stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run one or more algorithms to predict Titanic survivorship. Implemented algorithms: ' +
        ', '.join(CLF_CLASSES.keys()))
    parser.add_argument('command', help='Command to run', choices=['train', 'predict', 'search-params'],
                        default='train')
    parser.add_argument('--training-set-file', help='Training set file', default='train.csv')
    parser.add_argument('--test-set-file', help='Test set file', default='test.csv')
    parser.add_argument('--algorithm', help='Algorithms or comma-separated list of algorithms to be used',
                        required=True)
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
