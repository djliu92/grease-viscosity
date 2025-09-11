#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import (
    train_test_split, LeaveOneOut, GridSearchCV
)
from sklearn.model_selection import RandomizedSearchCV, LeaveOneOut

def get_data(file_path: str, sheet_name: str):
    """Get training data

    Args:
        file_path (str): File path to xlsx.
        sheet_name (str): Sheet name.

    Returns:
        list: X and y data.
    """
    # read file
    file = pd.read_excel(file_path, sheet_name=sheet_name)
    data = file.values
    X, y = data[:, :3], data[:, 3]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data set into train and test.

    Args:
        X (list): X data.
        y (list): y data.
        test_size (float, optional): Test size. Defaults to 0.2.
        random_state (int, optional): Random seed. Defaults to 42.

    Returns:
        list: Train and test data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


#def grid_search(X_train, y_train, estimator, param_grid):
#    clf = GridSearchCV(
#        estimator=estimator, param_grid=param_grid, cv=LeaveOneOut(),
#        scoring='neg_mean_squared_error'
#    )
#    clf.fit(X_train, y_train)

#    return clf.best_params_, clf.best_score_
def randomized_search(X_train, y_train, estimator, param_dist, n_iter=50):

    clf = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=LeaveOneOut(), 
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,  
        #verbose=0  
        verbose=2  
    )
    
    clf.fit(X_train, y_train)

    return clf.best_params_, clf.best_score_
