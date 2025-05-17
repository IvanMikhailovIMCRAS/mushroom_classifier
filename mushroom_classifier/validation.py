import numpy as np
from typing import Callable, Type, Tuple, Iterable
from alive_progress import alive_bar
from sklearn.metrics import accuracy_score, confusion_matrix


def model_validation(X: np.ndarray, y: np.ndarray, model: Type, folding: Iterable, n_bar: int) -> Tuple[float, np.ndarray]:
    y_true, y_pred = [], []
    with alive_bar(n_bar) as bar:
        for train_index, test_index in folding:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            for y_p, y_t in zip(model.predict(X_test), y_test):
                y_pred.append(y_p)
                y_true.append(y_t)
            bar()
    return accuracy_score(y_true, y_pred), confusion_matrix(y_true, y_pred)
