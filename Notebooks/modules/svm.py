"""
Original code:

svclassifier = SVC(kernel='linear',probability=True)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print('SVM accuracy:', svclassifier.score(X_test, y_test))
print('SVM classification report:\n', classification_report(y_test, y_pred))
y_scores = svclassifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
# Calculate AUC
svm_auc = auc(fpr, tpr)
print('SVM AUC value:', svm_auc)
newsvm=SVC(kernel='linear', probability=True)
newsvm.fit(X_train_pca, y_train)
y_pred = newsvm.predict(X_test_pca)
print('\nSVM accuracy for PCA:', newsvm.score(X_test_pca, y_test))
print('SVM classification report for PCA:\n', classification_report(y_test, y_pred))
y_scores = newsvm.predict_proba(X_test_pca)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
pcasvm_auc = auc(fpr, tpr)
print('SVM AUC value for PCA:', pcasvm_auc)"""

"""New code:"""

from sklearn.svm import SVC
#from sklearn import svm (you don't need this, you're importing sklearn.svm above. unless you need another thing from svm you can exclude this)

def evaluate_svm(X_train, y_train, X_test, y_test):
    # non-PCA
    svclassifier = SVC(kernel='linear',probability=True)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    svm_accuracy = svclassifier.score(X_test, y_test) # i think this is wrong
    y_scores = svclassifier.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    svm_auc = auc(fpr, tpr) # calculate AUC

    # PCA
    newsvm=SVC(kernel='linear', probability=True)
    newsvm.fit(X_train_pca, y_train)
    y_pred = newsvm.predict(X_test_pca)
    svm_accuracy_pca = newsvm.score(X_test_pca, y_test)
    y_scores = newsvm.predict_proba(X_test_pca)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    pcasvm_auc = auc(fpr, tpr)

    # return results
    return svm_accuracy, svm_auc, svm_accuracy_pca, pcasvm_auc