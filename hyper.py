import numpy as np

## predict f
l_pred_clr = []
l_pred_clnb = []
for s in range(0, Ns):
    X_train = matX[cca_mat_inf[:, 0] != s]
    y_train = cca_mat_inf[cca_mat_inf[:, 0] != s, 2]
    X_test = matX[cca_mat_inf[:, 0] == s]
    y_test = cca_mat_inf[cca_mat_inf[:, 0] == s, 2]

    clnb = MultinomialNB()
    clr = LogisticRegression(C=100, class_weight='balanced', max_iter=1000, penalty='l2', multi_class='multinomial', solver='newton-cg', tol=1e-4)
    clr.fit(X_train, y_train)
    clnb.fit(X_train, y_train)
    l_pred_clr.append(clr.predict(X_test))
    l_pred_clnb.append(clnb.predict(X_test))
    print("Accuracy LR - %.2f, NB - %.2f for length %.2f in subject %i." % (clr.score(X_test,y_test)*100,clnb.score(X_test,y_test)*100,l,s))


