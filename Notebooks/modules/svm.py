from sklearn.svm import SVC

def svm_eval(X_train, y_train, X_test, y_test):
    # non-PCA
    svclassifier = SVC(kernel='linear',probability=True)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    svm_accuracy = svclassifier.score(X_test, y_test)
    y_scores = svclassifier.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    svm_auc = auc(fpr, tpr)
    print('SVM AUC value:', svm_auc)

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
