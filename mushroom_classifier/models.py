import json

import numpy as np
from alive_progress import alive_bar
from outputs import show_exp_var
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

if __name__ == "__main__":
    keyword = "gills"  # cap, stipe, gills
    X = np.loadtxt(f"Xder_{keyword}.txt")
    y = np.loadtxt(f"y_{keyword}.txt")

    # X = np.vstack([np.loadtxt(f"Xder_cap.txt"),np.loadtxt(f"Xder_stipe.txt"),np.loadtxt(f"Xder_gills.txt")])
    # y = np.hstack([np.loadtxt(f"y_cap.txt"),np.loadtxt(f"y_stipe.txt"),np.loadtxt(f"y_gills.txt")])

    print(len(X))
    print(len(y))

    clss_file = open(f"classes_{keyword}.json")
    classes = json.load(clss_file)
    labels = {v: int(k) for k, v in classes.items()}
    classes = {v: k for k, v in labels.items()}

    scaler = preprocessing.StandardScaler().fit(X)
    Xs = scaler.transform(X)

    loo = LeaveOneOut()

    # Инициализация списка для хранения предсказаний
    y_true, y_pred = [], []
    C = 1.0
    # Цикл LOOCV
    with alive_bar(len(Xs)) as bar:
        for train_index, test_index in loo.split(Xs):
            bar()
            X_train, X_test = Xs[train_index], Xs[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Обучение SVM
            model = SVC(kernel="linear", gamma="scale", C=C, degree=3)
            model.fit(X_train, y_train)
            # print(model.get_params())

            # Предсказание
            y_pred.append(model.predict(X_test)[0])
            y_true.append(y_test[0])

    # Вычисление точности
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Точность предсказания (LOOCV): {accuracy:.4f}, C={C}")

    # print(confusion_matrix(y_true, y_pred))
