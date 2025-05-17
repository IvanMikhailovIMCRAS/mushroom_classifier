import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import spectrums as sp
from fcm import fuzzy_c_means
from outputs import show_exp_var
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    keyword = "stipe"  # cap, stipe, gills
    X = np.loadtxt(f"Xder_{keyword}.txt")
    y = np.loadtxt(f"y_{keyword}.txt")

    clss_file = open(f"classes_{keyword}.json")
    classes = json.load(clss_file)
    labels = {v: int(k) for k, v in classes.items()}
    classes = {v: k for k, v in labels.items()}

    # standardization (спорный момент - нужна ли стандартизация после второй производной)
    # scaler = preprocessing.StandardScaler().fit(X)
    # Xs = scaler.transform(X)
    # PCA processing

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train = X
    X_test = X
    y_train = y
    y_test = y

    n_pc = 100
    pca = PCA(n_components=n_pc)
    pca.fit(X_train)
    X_pca_train = pca.transform(X_train)
    X_pca_test = pca.transform(X_test)
    # show explaned variance curves
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    print("Number of pca components: ", n_pc)
    print("Max cumulative explained variance: ", cum_sum_eigenvalues[-1])
    show_exp_var(exp_var_pca, output_file=f"explained_variance_{keyword}.jpg")

    assert len(set(y_test)) == len(set(y_train)) == len(set(y))
    n_clusters = len(classes)

    U0 = np.zeros(shape=(len(y_train), n_clusters), dtype=float)
    for i in range(len(y_train)):
        U0[i, int(y[i])] = 1.0

    methods = [
        "euclidean",
        "minkowski",
        "cityblock",
        # "seuclidean",
        "cosine",
        "correlation",
        # "mahalanobis",
        "chebyshev",
        # "jensenshannon",
        "braycurtis",
        # "canberra"
    ]

    methods = [
        "cityblock",
    ]

    max_prec = 0.0
    max_m = 0.0
    max_iter = 1000
    for method in methods:
        xx = np.linspace(1.1, 2.0, 19)
        xx = np.array([1.15])
        yy = []
        for m in xx:
            U, centroids, it = fuzzy_c_means(
                X_pca_train, U0, m=m, max_iter=max_iter, tdist=method
            )

            # Updating the membership matrix
            distances = cdist(X_pca_test, centroids, method)
            # Avoid division by 0
            distances = np.fmax(distances, np.finfo(np.float64).eps)
            U = 1 / (distances ** (2 / (m - 1)))
            U /= np.sum(U, axis=1)[:, np.newaxis]  # Normalization

            predict = dict()
            for n, u in zip(y_test, U):
                n = int(n)
                max_value = np.max(u)
                indices = np.where(u == max_value)[0]
                if len(indices == 1):
                    if n in predict:
                        predict[n].append(indices[0])
                    else:
                        predict[n] = [indices[0]]
            true_answers = dict()
            for mushroom_index in predict:
                counter = Counter(predict[mushroom_index])
                most_common_element, count = counter.most_common(1)[0]
                print(
                    f"{classes[mushroom_index]}:\t\t{most_common_element},\t{round(count / len(predict[mushroom_index])*100,1)} %"
                )
                if most_common_element in true_answers:
                    true_answers[most_common_element].append(count)
                    print(mushroom_index, "OOOOOOOOOOOOOO")
                else:
                    true_answers[most_common_element] = [count]
            print(f"m = {m}")
            print(f"всего образцов: {len(y_test)}")
            count_predicted = 0
            for t in true_answers:
                mean_count = np.mean(np.array(true_answers[t]))
                # print(t, mean_count)
                count_predicted += mean_count
            print(f"правильно предсказано: {count_predicted}")
            prec = count_predicted / len(y_test) * 100
            if it >= max_iter:
                print("not ok", it)
                prec = 0.0
            else:
                print("ok", it)
            print(f"{round(prec, 2)} %")

            yy.append(prec)
            if prec > max_prec:
                max_prec = prec
                max_m = m
        # np.savetxt(method + f"_{keyword}.txt", np.vstack([xx, yy]))
    #     plt.plot(xx, yy, "-o", label=method)
    # plt.xlabel("$m$")
    # plt.ylabel("classification accuracy, %")
    # plt.legend()
    # plt.title(f"Max accuracy {round(max_prec,2)} %,  m = {round(max_m,2)}")
    # # plt.xlim(1,4.5)
    # plt.ylim(0, 100)
    # # plt.semilogx()
    # plt.savefig(f"{keyword}_dist.jpg")
