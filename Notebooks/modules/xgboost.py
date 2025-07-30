import xgboost as xgb

def xgb_eval(X_train, y_train, x_test, y_test):
    # non-PCA
    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    y_pred = clf.predict(X_test)
    xgb_accuracy = clf.score(X_test, y_test)
    y_scores = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    xgb_auc = auc(fpr, tpr)

    # PCA
    pxgb = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)
    pxgb.fit(X_train_pca, y_train, eval_set=[(X_test_pca, y_test)])
    xgb_accuracy_pca = pxgb.score(X_test_pca, y_test)
    y_pred = pxgb.predict(X_test_pca)
    y_scores = pxgb.predict_proba(X_test_pca)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    pxgb_auc = auc(fpr, tpr)

    # return results
    return xgb_accuracy, xgb_auc, xgb_accuracy_pca, pxgb_auc
