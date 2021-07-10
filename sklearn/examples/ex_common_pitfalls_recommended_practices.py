# -*- coding: utf-8 -*-
"""
 Created on 2021/7/10 15:06
 Filename   : ex_common_pitfalls_recommended_practices.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def demo_inconsistent_preprocessing():
    random_state = 42
    X, y = make_regression(random_state=random_state, n_features=1, noise=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=random_state)

    """ Wrong """
    """ The train dataset is scaled, but not the test dataset, so model performance on the test dataset is worse than expected """
    scaler = StandardScaler()
    X_train_transformed = scaler.fit_transform(X_train)
    model = LinearRegression().fit(X_train_transformed, y_train)
    mean_squared_error(y_test, model.predict(X_test))

    """ Right """
    X_test_transformed = scaler.transform(X_test)
    mean_squared_error(y_test, model.predict(X_test_transformed))

    """ using a Pipeline """
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    mean_squared_error(y_test, model.predict(X_test))


def demo_data_leakage():
    n_samples, n_features, n_classes = 200, 10000, 2
    rng = np.random.RandomState(42)
    X = rng.standard_normal((n_samples, n_features))
    y = rng.choice(n_classes, n_samples)

    """ Wrong """
    # Incorrect preprocessing: the entire data is transformed
    X_selected = SelectKBest(k=25).fit_transform(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, random_state=42)
    gbc = GradientBoostingClassifier(random_state=1)
    gbc.fit(X_train, y_train)

    y_pred = gbc.predict(X_test)
    accuracy_score(y_test, y_pred)

    """ Right """
    """ To prevent data leakage, it is good practice to split your data into train and test subsets first """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    select = SelectKBest(k=25)
    X_train_selected = select.fit_transform(X_train, y_train)

    gbc = GradientBoostingClassifier(random_state=1)
    gbc.fit(X_train_selected, y_train)

    X_test_selected = select.transform(X_test)
    y_pred = gbc.predict(X_test_selected)
    accuracy_score(y_test, y_pred)

    """ using a Pipeline """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    pipeline = make_pipeline(SelectKBest(k=25), GradientBoostingClassifier(random_state=1))
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy_score(y_test, y_pred)

    scores = cross_val_score(pipeline, X, y)
    print(f"Mean accuracy: {scores.mean():.2f}+/-{scores.std():.2f}")


def demo_controlling_randomness():
    # Estimators
    rng = np.random.RandomState(0)
    X, y = make_classification(n_features=5, random_state=rng)
    sgd = SGDClassifier(random_state=rng)

    sgd.fit(X, y).coef_
    sgd.fit(X, y).coef_

    # CV splitters
    X = y = np.arange(10)
    rng = np.random.RandomState(0)
    cv = KFold(n_splits=2, shuffle=True, random_state=rng)

    for train, test in cv.split(X, y):
        print(train, test)

    for train, test in cv.split(X, y):
        print(train, test)


def demo_estimators():
    X, y = make_classification(random_state=0)

    rf_123 = RandomForestClassifier(random_state=123)
    cross_val_score(rf_123, X, y)

    rf_inst = RandomForestClassifier(random_state=np.random.RandomState(0))
    cross_val_score(rf_inst, X, y)


def demo_clone():
    rng = np.random.RandomState(0)
    a = RandomForestClassifier(random_state=rng)
    b = clone(a)


def demo_cv_splitter():
    rng = np.random.RandomState(0)
    X, y = make_classification(random_state=rng)
    cv = KFold(shuffle=True, random_state=rng)
    lda = LinearDiscriminantAnalysis()
    nb = GaussianNB()

    for est in (lda, nb):
        print(cross_val_score(est, X, y, cv=cv))


def demo_random_state():
    rng = np.random.RandomState(0)
    X, y = make_classification(random_state=rng)
    rf = RandomForestClassifier(random_state=rng)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=rng)
    rf.fit(X_train, y_train).score(X_test, y_test)


if __name__ == '__main__':
    demo_inconsistent_preprocessing()
    demo_data_leakage()
    demo_controlling_randomness()
    demo_estimators()
    demo_clone()
    demo_cv_splitter()
    demo_random_state()
