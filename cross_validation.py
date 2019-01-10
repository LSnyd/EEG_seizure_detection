from sklearn.model_selection import KFold

def kfold(data, label, bag):

    pred = []
    true = []
    prob = []

    kf = KFold(n_splits=4)
    kf.get_n_splits(data)
    KFold(n_splits=4, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(data):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        bag.fit(X_train, y_train)
        prediction = bag.predict(X_test)
        prob.extend(bag.predict_proba(X_test))
        pred.extend(prediction)
        true.extend(y_test)

        print("Real Class:", y_test)
        print("Predicted class:", prediction)
        print("Probabilities", bag.predict_proba(X_test))
        print(pred)
        print(true)
        print(prob)

    return true, pred