#!/usr/bin/env python3

import os
import pandas as pd
from sklearn.ensemble import VotingClassifier

import logging
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump, load


logging.basicConfig(level=logging.INFO)


def generate_metrics(clf, X_test, y_test):
    accuracy = clf.score(X_test, y_test)
    logging.info('Accuracy: ' + str(accuracy))
    return accuracy


def create_random_forest_classifier(X_train, X_test, y_train, y_test):
    logging.info('Creating Random Forest Classifier...')
    random_forest = Pipeline([('scaler', StandardScaler()), ('random_forest', RandomForestClassifier(n_estimators=20))])
    random_forest.fit(X_train, y_train)
    generate_metrics(random_forest, X_test, y_test)
    logging.info('Random Forest Classifier created')
    return random_forest


def create_support_vector_classifier(X_train, X_test, y_train, y_test):
    logging.info('Creating Support Vector Classifier...')
    svm_clf = Pipeline(
        [('scaler', StandardScaler()), ('SVM', svm.SVC(C=10, gamma=0.01, kernel="rbf", probability=True))])
    svm_clf.fit(X_train, y_train)
    generate_metrics(svm_clf, X_test, y_test)
    logging.info('Support Vector Classifier created')
    return svm_clf


def create_logistic_regression_classifier(X_train, X_test, y_train, y_test):
    logging.info('Creating Logistic Regression Classifier...')
    logistic_regression = Pipeline([('random_forest', LogisticRegression(C=1))])
    logistic_regression.fit(X_train, y_train)
    generate_metrics(logistic_regression, X_test, y_test)
    logging.info('Logistic Regression Classifier created')
    return logistic_regression


def create_voting_classifier(svm_clf, rf_clf, lr_clf, X_train, X_test, y_train, y_test):
    logging.info('Creating Voting Classifier...')
    voting_clf = VotingClassifier(
        estimators=[("svc", svm_clf), ("rf", rf_clf), ("lr", lr_clf)],
        voting="soft"
    )
    voting_clf.fit(X_train, y_train)
    generate_metrics(voting_clf, X_test, y_test)
    logging.info('Voting Classifier created')
    return voting_clf


def save_model(clf):
    logging.info('Saving model...')
    dirname = "./../model"
    os.makedirs(dirname, exist_ok=True)
    dump(clf, dirname + '/heart_failure_clf.joblib')
    logging.info('Model saved')
    pass


def get_data_for_model_creation(cols):
    logging.info('Loading data for model creation...')
    heart_failure_data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    return heart_failure_data[cols]


def generate_model():
    logging.info('Generating models...')

    feature_cols = ["age", "ejection_fraction", "serum_sodium", "serum_creatinine", "time"]
    label_col = "DEATH_EVENT"
    all_cols = feature_cols + [label_col]

    data = get_data_for_model_creation(all_cols)

    logging.info('Performing train test split ...')
    X_train, X_test, y_train, y_test = train_test_split(
        data[feature_cols], data[label_col], test_size=0.2, random_state=0)
    logging.info('Train test split complete')

    rf_clf = create_random_forest_classifier(X_train, X_test, y_train, y_test)
    svm_clf = create_support_vector_classifier(X_train, X_test, y_train, y_test)
    lr_clf = create_logistic_regression_classifier(X_train, X_test, y_train, y_test)

    clf = create_voting_classifier(svm_clf, rf_clf, lr_clf, X_train, X_test, y_train, y_test)

    save_model(clf)


if __name__ == '__main__':
    generate_model()
