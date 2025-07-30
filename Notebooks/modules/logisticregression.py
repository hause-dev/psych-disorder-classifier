from sklearn.linear_model import LogisticRegression

def lg_eval(X_train, y_train, X_test, y_test):
    # non-PCA
    lg = LogisticRegression()
    lg.fit(X_train, y_train)
    lg_accuracy = lg.score(X_test, y_test)
    y_pred = lg.predict(X_test)
    y_scores = lg.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    lg_auc = auc(fpr, tpr)

    # PCA
    plg = LogisticRegression()
    plg.fit(X_train_pca, y_train)
    lg_accuracy_pca = plg.score(X_test_pca, y_test)
    y_pred = plg.predict(X_test_pca)
    y_scores = plg.predict_proba(X_test_pca)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    pcalg_auc = auc(fpr, tpr)

    # return results
    return lg_accuracy, lg_auc, lg_accuracy_pca, pcalg_auc
