import pandas as pd
import time
import sys
import numpy as np
import joblib
import dateutil
import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import pyplot as plt


def strip_funds(np_arr):
    a = set(range(3, 15))
    return np.delete(np_arr, list(a), axis=1)


if __name__ == "__main__":
    f = sys.argv[1]
    print("Loading model input from %s." % f)
    model_input = joblib.load(f)
    print("Done loading.")

    print("Removing invalid examples.")
    X_train_, y_train, transcripts = strip_funds(model_input.X_train), model_input.all_eps, model_input.transcript
    print("X_train.shape before removing funds: %s" % str(X_train_.shape))

    print("Using elbow method to choose number of clusters.")
    sum_sq = []
    kmeans = []
    now = str(datetime.datetime.now()).replace(" ", "_")
    for i in range(1, 100):
        print("Training KMeans for %s clusters." % i)
        start = time.time()
        kmeans_ = KMeans(n_clusters=i, n_jobs=-1).fit(X_train_)
        sum_sq.append(kmeans_.inertia_)
        kmeans.append(kmeans_)
        stop = time.clock()
        print("Done - time taken: %s s." % str(stop - start))

    joblib.dump(sum_sq, "elbow/%s-sum-sq.pkl" % now)
    joblib.dump(kmeans, "elbow/%s-kmeans.pkl" % now)
    print("Generating and saving plot.")
    plt.plot(range(1, 100), sum_sq)
    plt.title("Elbow Curve")
    plt.savefig("elbow/%s-elbow_curve.jpeg" % now)
