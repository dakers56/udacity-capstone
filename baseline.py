import sys
import joblib
import dateutil
import datetime
import pandas as pd
import json
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def by_symbol(X_train):
    df = pd.DataFrame(X_train)
    as_dict = {}
    for i in range(X_train.shape[0]):
        sym = df.iloc[i][0]
        row = df.iloc[i][:].reshape(1,-1)
        if sym not in as_dict:
            as_dict[sym] = row.reshape(1, -1)
        else:
            as_dict[sym] = np.concatenate((as_dict[sym], row), axis=0)
    return as_dict


def replace_nan(X_):
    bad = []
    for i in range(X_.shape[0]):
        for j in range(X_.shape[1]):
            if not np.isfinite(X_[i][j]):
                # print("Entry %s:%s was not finite." % (i, j))
                bad.append((i,j, X_[i][j]))
                X_[i][j] = np.nan_to_num(X_[i][j], copy=True)
    return bad


if __name__ == "__main__":
    print("Training baseline linear regression model.")
    data_file = sys.argv[1]
    print("Loading data from %s." % data_file)

    data = joblib.load(data_file)
    X_, y_ = data.X_train, data.all_eps
    sym_map = joblib.load(sys.argv[2])

    bad = replace_nan(X_)

    print("Bad entries:")
    for b in bad:
        print("%s:%s = %s" % (b[0], b[1], b[2]))
    joblib.dump(bad, "bad_data/bad-data-for-%s.pkl" % data_file[data_file.rfind("/")+1:].replace("/", "-"))


    X_train, X_test, y_train, y_test = train_test_split(X_, y_)

    X_by_symbol = by_symbol(X_train)

    X_scaler = preprocessing.StandardScaler().fit(X_train)
    y_scaler = preprocessing.StandardScaler().fit(y_train)

    _X_train = X_scaler.transform(X_train)
    _X_test = X_scaler.transform(X_test)
    _y_train = y_scaler.transform(y_train)
    _y_test = y_scaler.transform(y_test)

    print("Training model.")
    model_type = "lin-reg"
    reg = LinearRegression().fit(_X_train, _y_train)
    print("Done training model.")
    print("Performance on training data: %s." % reg.score(X_train, y_train))
    print("Performance on test data: %s." % reg.score(X_test, y_test))
    joblib.dump(reg, "baseline/%s-%s" % (
        model_type, str(dateutil.parser.parse(str(datetime.datetime.now()))).replace(" ", "_")))
