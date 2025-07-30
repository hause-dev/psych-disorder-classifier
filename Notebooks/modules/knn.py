from sklearn.neighbors import KNeighborsClassifier

def knn_eval(X_train, y_train, X_test, y_test):
    # non-PCA
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)
    knn_accuracy = knn.score(X_test, y_test)
    y_pred = knn.predict(X_test)
    y_scores = knn.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    knn_auc = auc(fpr, tpr)

    # PCA
    pknn = KNeighborsClassifier(n_neighbors=10)
    pknn.fit(X_train_pca, y_train)
    knn_accuracy_pca = pknn.score(X_test_pca, y_test)
    y_pred = pknn.predict(X_test_pca)
    y_scores = pknn.predict_proba(X_test_pca)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    pcaknn_auc = auc(fpr, tpr)

    # return results
    return knn_accuracy, knn_auc, knn_accuracy_pca, pcaknn_auc
