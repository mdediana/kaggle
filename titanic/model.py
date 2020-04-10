import argparse
import sys
import logging
from itertools import product

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier,
                              ExtraTreesClassifier, VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Simple names to be used on the command line
CLF_CLASSES = {
    'AdaBoost': AdaBoostClassifier,
    'RandomForest': RandomForestClassifier,
    'ExtraTrees': ExtraTreesClassifier,
    'GradientBoosting': GradientBoostingClassifier,
    'KNeighbors': KNeighborsClassifier,
    'XGB': XGBClassifier,
    'LogisticRegression': LogisticRegression,
    'SVC': SVC,
}
N_JOBS = -1  # Use all processors, particularly useful when param grid searching
CV_FOLDS = 5
SCORING = 'accuracy'
DEFAULT_VOTING = 'hard'
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
        n_estimators=[10, 100, 500, 1000],
        max_depth=[None, 10, 20],
        max_features=['auto', None],
        # max_features=[None],
        # Regularization param fixed after manual tests comparing cross val and training set scores
        min_samples_leaf=[0.01],
        n_jobs=[N_JOBS],
    ),
    ExtraTreesClassifier: dict(
        n_estimators=[10, 100, 500, 1000],
        max_depth=[None, 10, 20],
        max_features=['auto', None],
        # Regularization param fixed after manual tests comparing cross val and training set scores
        min_samples_leaf=[0.01],
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
    LogisticRegression: dict(
        C=[0.01, 0.05, 0.1, 0.5, 1],
        max_iter=[1000],
    ),
    SVC: dict(
        C=[0.1, 1, 2],
        kernel=['linear', 'rbf', 'sigmoid'],  # poly does not converge
        gamma=['scale', 'auto'],
    ),
}
BEST_PARAMS = {
    AdaBoostClassifier: dict(
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
        n_estimators=1000,
        max_depth=20,
        max_features='auto',
        # Regularization params
        # min_samples_split=0.1,
        min_samples_leaf=0.01,
        random_state=0,
    ),
    ExtraTreesClassifier: dict(
        n_estimators=100,
        max_depth=10,
        max_features=None,
        # Regularization params
        # min_samples_split=0.1,
        min_samples_leaf=0.01,
        random_state=0,
    ),
    XGBClassifier: dict(
        random_state=0,
    ),
    KNeighborsClassifier: dict(
        algorithm='auto',
        n_neighbors=7,
        leaf_size=26,
        weights='uniform',
        # metric='minkowski',
    ),
    LogisticRegression: dict(
        C=0.5,
        max_iter=1000,
        random_state=0,
    ),
    SVC: dict(
        C=1,
        kernel='linear',
        gamma='scale',
        random_state=0,
    ),
}
COLUMNS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySurvived', 'FamilySize']

logger = logging.getLogger(__name__)


def _read_csv(filename, split_X_y=True):
    """Return X, y if training data or X, None if test data"""
    df = pd.read_csv(filename)
    df.set_index('PassengerId', inplace=True)
    if split_X_y and 'Survived' in df:
        X = df.drop(columns=['Survived'])
        y = df['Survived']
    else:
        X = df
        y = None
    return X, y


def _new_features(X_train, y_train, X_test):
    # FamilySize
    for df in [X_train, X_test]:
        df['FamilySize'] = df['Parch'] + df['SibSp']

    # FamilySurvived feature idea from https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83/
    data_df = X_train.join(y_train).append(X_test, sort=True)
    data_df['Family'] = data_df['Name'].str.split(',', n=1, expand=True)[0]
    data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)

    # Group by family and fare
    for _, grp_df in data_df.groupby(['Family', 'Fare']):
        for ind, row in grp_df.iterrows():
            # 0.5 if passenger is alone or there is no survivorship info (test set)
            # 0 if everyone else in group is dead
            # 1 if there is at least one survivor in group
            others = grp_df.drop(ind)['Survived']
            data_df.loc[ind, 'FamilySurvived'] = 0.5 if others.isnull().all() else others.max()
    logger.info('Number of passengers with family survival information: %d',
                len(data_df[data_df['FamilySurvived'] != 0.5]))

    # Add information about group survival from ticket
    for _, grp_df in data_df.groupby('Ticket'):
        for ind, row in grp_df.iterrows():
            if row['FamilySurvived'] == 1:
                continue
            others = grp_df.drop(ind)['Survived']
            if others.max() == 1:    # Any others survived
                data_df.loc[ind, 'FamilySurvived'] = 1
            elif others.min() == 0:  # All others dead
                data_df.loc[ind, 'FamilySurvived'] = 0
    logger.info('Number of passengers with family survival information after ticket groups: %d',
                len(data_df[data_df['FamilySurvived'] != 0.5]))

    X_train['FamilySurvived'] = data_df['FamilySurvived'][:len(X_train)]
    X_test['FamilySurvived'] = data_df['FamilySurvived'][len(X_train):]
    return X_train, X_test


def _read_data(training_set_file, test_set_file):
    X_train, y_train = _read_csv(training_set_file)
    X_test, _ = _read_csv(test_set_file)
    X_train, _ = _new_features(X_train, y_train, X_test)
    X_train = X_train[COLUMNS]
    X_test = X_test[COLUMNS]
    return X_train, y_train, X_test


def _instantiate_clf(algorithms, use_best_params=True, voting=DEFAULT_VOTING):
    """Return a classifier. If 'algorithms' is a list of strings a VotingClassifier is returned,  if it is a string,
    a single classifier using that algorithm is returned.
    """
    if isinstance(algorithms, str):
        algorithms = [algorithms]
    logger.info('Number of estimators: %d', len(algorithms))
    estimators = []  # Contains (name, clf) tuples for the VotingClassifier
    for algorithm in algorithms:
        class_ = CLF_CLASSES[algorithm]
        params = BEST_PARAMS[class_] if use_best_params else {}
        logger.info('Instantiating %s %s', class_.__name__, params if params else '')
        estimators.append((algorithm, class_(**params)))
    if len(estimators) == 1:
        clf = estimators[0][1]
    else:
        logger.info('Creating voting estimator: %s', voting)
        clf = VotingClassifier(estimators=estimators, voting=voting)
        estimators = [(estimator.__class__.__name__, estimator) for estimator in estimators]
    return clf


def _column_transformer(columns, remainder='drop'):
    imputers = {
        'Embarked': make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder()),
        'Age': make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler()),
        'Pclass': OneHotEncoder(),
        'Sex': OneHotEncoder(),
        'Fare': SimpleImputer(strategy='mean'),
        'SibSp': StandardScaler(),
        'Parch': StandardScaler(),
        'FamilySurvived': StandardScaler(),
        'TicketSurvived': StandardScaler(),
        'FamilySize': StandardScaler(),
        'FarePerPerson': StandardScaler(),
    }
    logger.info('Preparing transformers for columns: %s', columns.tolist())
    logger.info('Columns without transformers to be dropped: %s', columns.tolist() - imputers.keys())
    transformers = [(imputers[col], [col]) for col in columns if col in imputers.keys()]
    return make_column_transformer(*transformers, remainder=remainder)


def _search_params(X, y, clf):
    column_transformer = _column_transformer(X.columns)
    param_grid = PARAM_GRIDS[clf.__class__]
    combinations = list(product(*param_grid.values()))
    logger.info('Search hyperparameters, number of combinations: %d', len(combinations))
    param_grid = {'clf__' + params: param_grid[params] for params in param_grid}
    pipeline = Pipeline([('column_transformer', column_transformer), ('clf', clf)])
    cv = GridSearchCV(pipeline, param_grid, scoring=SCORING)
    cv.fit(X, y)
    best_params = {param.replace('clf__', ''): value for param, value in cv.best_params_.items()}
    logger.info('Best score: %s', cv.best_score_)
    logger.info('Best parameters: %s', best_params)
    # logger.info('Results: %s', cv.cv_results_)
    return cv.best_params_


def _train(X, y, clf):
    column_transformer = _column_transformer(X.columns)
    pipeline = make_pipeline(column_transformer, clf)
    pipeline.fit(X, y)
    scores = cross_val_score(pipeline, X, y, cv=CV_FOLDS, scoring=SCORING)
    y_predict = pipeline.predict(X)
    score_predict = (y_predict == y).sum() / len(y)
    logger.info('%s - Score cross validation: %f+/-%f %s - Score training set: %f (diff: %f)',
                clf.__class__.__name__, scores.mean(), scores.std(), scores, score_predict,
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


def search_params(training_set_file, test_set_file, algorithms):
    X_train, y_train, _ = _read_data(training_set_file, test_set_file)
    for algorithm in algorithms:
        model = _instantiate_clf(algorithm, use_best_params=False)
        _search_params(X_train, y_train, model)


def train(training_set_file, test_set_file, algorithms):
    X_train, y_train, _ = _read_data(training_set_file, test_set_file)
    clf = _instantiate_clf(algorithms, use_best_params=True)
    _train(X_train, y_train, clf)


def predict(training_set_file, test_set_file, algorithms):
    X_train, y_train, X_test = _read_data(training_set_file, test_set_file)
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
    parser.add_argument('--algorithms', help='Algorithm or comma-separated list of algorithms to be used',
                        required=True)
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    algorithms = args.algorithms.split(',')
    if args.command == 'train':
        train(args.training_set_file, args.test_set_file, algorithms)
    elif args.command == 'predict':
        predict(args.training_set_file, args.test_set_file, algorithms)
    elif args.command == 'search-params':
        search_params(args.training_set_file, args.test_set_file, algorithms)
