import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Load data from numpy file

def svm(feature, label):
    X = np.load(feature)
    # X = X.reshape(-1, 1)
    y = np.load(label).ravel()
    # print(X.shape)
    # Split data into training and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # print(X_train)
    # print(y_train)
    print('SVM **************************************')
    clf_svm = SVC(C = 10, kernel = 'linear')
    clf_svm.fit(X_train, y_train)
    y_pred = clf_svm.predict(X_test)
    # tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    # acc_svm = clf_svm.score(X_test, y_test)
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1 = 2*tp / (2*tp + fp + fn)
    # print("svm acc=%0.3f" % acc_svm)
    # print('Precision: %f' % precision)
    # print('Recall: %f' % recall)

    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    print('Recall1: %.3f' % recall_score(y_test, y_pred))
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('F1 Score: %.3f' % f1_score(y_test, y_pred))

def gnb(feature, label):
    X = np.load(feature)
    y = np.load(label).ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print('GNB **************************************')
    clf_gnb = GaussianNB()
    clf_gnb.fit(X_train, y_train)
    y_pred = clf_gnb.predict(X_test)
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    print('Recall1: %.3f' % recall_score(y_test, y_pred))
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('F1 Score: %.3f' % f1_score(y_test, y_pred))

def knn(feature, label):
    X = np.load(feature)
    y = np.load(label).ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print('KNN **************************************')
    clf_knn = KNeighborsClassifier(n_neighbors=6)
    clf_knn.fit(X_train, y_train)
    y_pred = clf_knn.predict(X_test)
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    print('Recall1: %.3f' % recall_score(y_test, y_pred))
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('F1 Score: %.3f' % f1_score(y_test, y_pred))

if __name__ == '__main__':
    svm('../feature_results/loaded_vs_unloaded/feat_combination_I.npy',
        '../feature_results/loaded_vs_unloaded/label_combination_I.npy')
    gnb('../feature_results/loaded_vs_unloaded/feat_combination_I.npy',
        '../feature_results/loaded_vs_unloaded/label_combination_I.npy')
    knn('../feature_results/loaded_vs_unloaded/feat_combination_I.npy',
        '../feature_results/loaded_vs_unloaded/label_combination_I.npy')
