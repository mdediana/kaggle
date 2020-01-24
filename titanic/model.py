import argparse
import sys
import logging
from itertools import product
import math

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def read_csv(filename):
    df = pd.read_csv(filename)
    df.set_index('PassengerId', inplace=True)
    # Set missing boy ages - better imputation of ages hurts accuracy (?)
    # childrens_age_median = df[df.Age < 18].Age.median()
    # mask = df.Name.str.contains('Master') & df.Age.isnull()
    # df.at[df[mask].index, 'Age'] = childrens_age_median
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = df['Survived'] if 'Survived' in df else None
    return X, y


def train(training_filename, model_class, model_params={}):
    X, y = read_csv(training_filename)

    embarked_trans = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder())
    column_trans = make_column_transformer(
        (embarked_trans, ['Embarked']),
        (OneHotEncoder(), ['Pclass', 'Sex']),
        (SimpleImputer(strategy='median'), ['Fare', 'Age']),
        remainder='passthrough',
    )

    params = {**model_params, **{'random_state': 0}}
    model = model_class(**params)
    pipeline = make_pipeline(column_trans, model)
    pipeline.fit(X, y)

    # Multiply by -1 since sklearn calculates *negative* MAE
    # scores = -1 * cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

    return pipeline, scores


def search_xgboost_hyperparams(training_filename):
    # For xgboost_params, see https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
    # https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
    xgboost_params = {
        # 'max_depth': [2, 3, 4, 6, 8, 10],
        'max_depth': [3],
        # 'learning_rate': [1, 0.5, 0.1, 0.05, 0.025, 0.001],
        'learning_rate': [0.07, 0.05],
        # 'n_estimators': [50, 100, 200, 500, 1000],
        'n_estimators': [200, 500, 1000, 2000],
        # 'gamma': [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        # 'min_child_weight': [1, 3, 5, 7],
        # 'subsample': [0.5, 0.75, 1.0],
        # 'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        # 'reg_alpha': [0, 0.1, 0.5, 1.0],
        # 'reg_lambda': [0.01, 0.1, 1.0],
    }
    combinations = list(product(*xgboost_params.values()))
    logger.info('Number of combinations: %d', len(combinations))
    best_score = 0
    best_params = []
    for i, combination in enumerate(combinations, 1):
        params = dict(zip(xgboost_params.keys(), combination))
        _, scores = train(training_filename, XGBClassifier, params)
        score = scores.mean()
        std = scores.std()
        if score > best_score:
            best_score = score
            best_params = [params]
        elif math.isclose(score, best_score, rel_tol=0.0001):
            best_params.append(params)
        logger.debug(f'{i}/{len(combinations)} {params}: {score}+/-{std} {scores} | best score: {best_score}')
    logger.info('Best score: %f', best_score)
    logger.info('Best params:')
    for params in best_params:
        logger.info(params)


def predict(pipeline, test_filename):
    X, _ = read_csv(test_filename)
    preds = pipeline.predict(X)
    preds_df = pd.DataFrame(index=X.index, data={'Survived': preds})
    preds_df.to_csv(sys.stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='Command to run', choices=['train', 'predict', 'search'], default='train')
    parser.add_argument('--training-set-file', help='Training set file', default='train.csv')
    parser.add_argument('--predict-file', help='Test set file', default='test.csv')
    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    MODEL_PARAMS = {
        XGBClassifier: {
            'max_depth': 3,
            'learning_rate': 0.05,
            'n_estimators': 500,
            # 'gamma': 0.05,
            # 'min_child_weight': 3,
            # 'subsample': 0.5,
            # 'colsample_bytree': 0.8,
            # 'reg_alpha': 1.0,
            # 'reg_lambda': 0.01
        },
        GradientBoostingClassifier: {
            # 'n_estimators': 175,
            # 'learning_rate': 0.05,
            # 'max_depth': 3,
            # 'max_features': 'sqrt',
            # # 'max_leaf_nodes': 4
        },
        AdaBoostClassifier: {
            # 'n_estimators': 5000,
            # # 'learning_rate': 0.1,
        },
        RandomForestClassifier: {
            'n_estimators': 5000,
            'max_depth': 3,
        },
    }
    # model_class = XGBClassifier
    model_class = GradientBoostingClassifier
    # model_class = AdaBoostClassifier
    # model_class = RandomForestClassifier
    if args.command == 'train':
        _, scores = train(args.training_set_file, model_class, MODEL_PARAMS.get(model_class, {}))
        logger.info('%s - Score: %f+/-%f %s', model_class.__name__, scores.mean(), scores.std(), scores)
    elif args.command == 'predict':
        logger.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
        pipeline, _ = train(args.training_set_file, model_class, MODEL_PARAMS.get(model_class, {}))
        predict(pipeline, args.predict_file)
    elif args.command == 'search':
        search_xgboost_hyperparams(args.training_set_file)
