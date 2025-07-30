from sklearn.ensemble import RandomForestClassifier

def rf_eval(X_train, y_train, X_test, y_test):
    # non-PCA
    model = RandomForestClassifier(n_estimators=40)
    model.fit(X_train, y_train)
    rf_accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    rf_auc = auc(fpr, tpr)

    # PCA
    pmodel = RandomForestClassifier(n_estimators=40)
    pmodel.fit(X_train_pca, y_train)
    rf_accuracy_pca = pmodel.score(X_test_pca, y_test)
    y_pred = pmodel.predict(X_test_pca)
    y_scores = pmodel.predict_proba(X_test_pca)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    pcarf_auc = auc(fpr, tpr)

    # return results
    return rf_accuracy, rf_auc, rf_accuracy_pca, pcarf_auc
