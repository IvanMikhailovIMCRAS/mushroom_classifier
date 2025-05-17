import numpy as np
from alive_progress import alive_bar
from outputs import show_exp_var
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split, check_cv
from sklearn.svm import SVC
from tqdm import tqdm
import pandas as pd
from validation import model_validation
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


if __name__ == "__main__":
    df = pd.read_csv('spectra_metadata.csv').sort_values(by='species')
    # mask = df['part'] == 'gills' # 'cap', 'gills', 'stipe'
    # spectrum_paths = df.loc[mask, 'spectrum_path']
    spectrum_paths = df['spectrum_path']
    spectra = [np.load(path) for path in spectrum_paths]
    X = np.vstack(spectra)
    # get derivatives
    # X_der = savgol_filter(X, 17, polyorder=5, deriv=2)
    
    scaler = preprocessing.StandardScaler().fit(X)
    Xss = scaler.transform(X)
    
    n_pc = 100
    pca = PCA(n_components=n_pc)
    pca.fit(Xss)
    X_pca = pca.transform(Xss)
    
    Xs = X_pca
    
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    print('Number of pca components: ', n_pc)
    print('Max cumulative explained variance: ', cum_sum_eigenvalues[-1])
    # show_exp_var(exp_var_pca, output_file='explained_variance.jpg')        
    
    num_class = dict()
    for i, name in enumerate(df['species'].unique()):
        num_class[name] = i
    y = np.array([num_class[s] for s in df['species']])
    
    # scaler = preprocessing.StandardScaler().fit(X_pca)
    # Xs = scaler.transform(X_pca)

    
    cv = 5
    print(check_cv(cv=cv, y=y, classifier=True))
    # folding = LeaveOneOut().split(y)
    # folding = StratifiedKFold(n_splits=cv, shuffle=False, random_state=None).split(Xs,y)
    # for id_train, id_test in folding:
    #     print(set(y[id_train]) ^ set(y[id_test]))
    
    # model = DummyClassifier(strategy='stratified') # 2.90 %
    model = OneVsRestClassifier(LogisticRegression(solver='liblinear')) # 99.50 % 100.00 (5:34.2) 98.92 (2.15)
    # model = SVC(kernel="linear", gamma="scale", C=1.0) # 95.93 % 99.57 (3.3) 98.92 (0.4)
    # model = DecisionTreeClassifier() # 55.85 % 64.23 (24.5) 50.62 (1.0)
    # model = LinearDiscriminantAnalysis(solver='svd') # 99.41 % 99.17 (5.8)  97.93 (1.4)
    # model = RidgeClassifier(alpha=0.1) # 99.92 % 99.5 (1.5) 97.34 (0.5)
    # model = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000) # 92.86 % 99.57 (13.4) 98.76 (0.6)
    # model = KNeighborsClassifier(n_neighbors=1) # 90.04 % 95.85 (0.2) 97.34 (0.1)   
    # model = GaussianNB() # 66.88 % 92.11 (0.5) 93.44 (0.1)
    
    folding = StratifiedKFold(n_splits=cv, shuffle=False, random_state=None).split(Xs,y)

    accuracy_score, confusion_matrix = model_validation(Xs, y, model, folding, n_bar=cv)
    print(accuracy_score)



