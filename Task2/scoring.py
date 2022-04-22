import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import tree, __all__, neighbors, svm
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold


# read in data from csv files
from sklearn.neural_network import MLPClassifier

# removed timestamp and pid
cols = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36)
test_features_data = np.genfromtxt('test_features.csv', delimiter=',', skip_header=1, usecols=cols)
train_features_data = np.genfromtxt('train_features.csv', delimiter=',', skip_header=1, usecols=cols)
train_labels_data = np.genfromtxt('train_labels.csv', delimiter=',', skip_header=1)

# impute missing values into test_features_data, with mean method
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

# fit imp with train features and test features
imp.fit(np.concatenate((train_features_data, test_features_data), axis=0))
SimpleImputer()
test_features_data = imp.transform(test_features_data)

# impute missing values into train_features_data, with mean method
train_features_data = imp.transform(train_features_data)

# reshape test features into 3D array
pat_nr = len(test_features_data)/12
test_features_data = np.ravel(test_features_data, order="C")
test_features_data = np.reshape(test_features_data, (int(pat_nr), 12*35))

# reshape train features into 3D array
pat_nr2 = len(train_features_data)/12
train_features_data = np.ravel(train_features_data, order="C")
train_features_data = np.reshape(train_features_data, (int(pat_nr2), 12*35))

# create 10-fold for validation
kf = KFold(n_splits=10, shuffle=True)

# 1. classifier for label base excess
# clf = tree.DecisionTreeClassifier(), best
# clf = neighbors.KNeighborsClassifier(30, weights='distance'), second best
# clf = svm.SVC(), runs long
# clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000), rly inconsistent some good results
# clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(35, 15), random_state=1, max_iter=100000),third 0,73
# estimators 200 :  0.88,0.86,0.87,0.86, rly slow
# entropy, 100 est : 0.87, 0.88, 0.86 , 0.86, 0.87
# estimators 100: 0.88,0.88, 0.87


# best n_estimators = 170, score = 0.875440148554641
for i in range(1,20):
    clf = RandomForestClassifier(n_estimators=i*10)
    sumScore = 0
    for train_index, test_index in kf.split(train_features_data):
        clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 1])

        score = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 1])
        sumScore += score

    print("label 1, " + str(i) + " estimators, score: " + str(sumScore/10))

print("-----------------------------------")

# label fibrinogen best n_estimators = 170, score = 0.9344563066433856
for i in range(1,20):
    clf = RandomForestClassifier(n_estimators=i*10)
    sumScore2 = 0
    for train_index, test_index in kf.split(train_features_data):
        clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 2])

        score2 = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 2])
        sumScore2 += score2

    print("label 2, " + str(i) +" estimators, score: " + str(sumScore2/10))

print("-----------------------------------")

# label AST, best n = 100, score = 0.7812040409079571
for i in range(1,20):
    clf = RandomForestClassifier(n_estimators=i*10)
    sumScore = 0
    for train_index, test_index in kf.split(train_features_data):
        clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 3])

        score = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 3])
        sumScore += score

    print("label 3, " + str(i) +" estimators, score: " + str(sumScore/10))

print("-----------------------------------")

# label Alkalinephos, best n = 150, score = 0.7855222693384329
for i in range(1,20):
    clf = RandomForestClassifier(n_estimators=i*10)
    sumScore = 0
    for train_index, test_index in kf.split(train_features_data):
        clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 4])

        score = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 4])
        sumScore += score

    print("label 4, " + str(i) +" estimators, score: " + str(sumScore/10))

print("-----------------------------------")

# label Bilirubin_total, best n = 100, score = 0.7810482525428896
for i in range(1,20):
    clf = RandomForestClassifier(n_estimators=i*10)
    sumScore = 0
    for train_index, test_index in kf.split(train_features_data):
        clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 5])

        score = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 5])
        sumScore += score

    print("label 5, " + str(i) +" estimators, score: " + str(sumScore/10))

print("-----------------------------------")

# label lactate, best n = 150, score = 0.8389575122640724
for i in range(1,20):
    clf = RandomForestClassifier(n_estimators=i*10)
    sumScore = 0
    for train_index, test_index in kf.split(train_features_data):
        clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 6])

        score = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 6])
        sumScore += score

    print("label 6, " + str(i) +" estimators, score: " + str(sumScore/10))

print("-----------------------------------")

# label TroponinI, best n = 180, score = 0.9057121476677474
for i in range(1,20):
    clf = RandomForestClassifier(n_estimators=i*10)
    sumScore = 0
    for train_index, test_index in kf.split(train_features_data):
        clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 7])

        score = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 7])
        sumScore += score

    print("label 7, " + str(i) +" estimators, score: " + str(sumScore/10))

print("-----------------------------------")

# label SaO2, best n = 170, score = 0.8322741886311356
for i in range(1,20):
    clf = RandomForestClassifier(n_estimators=i*10)
    sumScore = 0
    for train_index, test_index in kf.split(train_features_data):
        clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 8])

        score = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 8])
        sumScore += score

    print("label 8, " + str(i) +" estimators, score: " + str(sumScore/10))

print("-----------------------------------")

# label Bilirubin_direct, n = 120, score = 0.9676755910312907
for i in range(1,20):
    clf = RandomForestClassifier(n_estimators=i*10)
    sumScore = 0
    for train_index, test_index in kf.split(train_features_data):
        clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 9])

        score = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 9])
        sumScore += score

    print("label 9, " + str(i) +" estimators, score: " + str(sumScore/10))

print("-----------------------------------")



# label ETC02, n = 140, score = 0.9712029600066516
for i in range(1,20):
    clf = RandomForestClassifier(n_estimators=i*10)
    sumScore = 0
    for train_index, test_index in kf.split(train_features_data):
        clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 10])

        score = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 10])
        sumScore += score

    print("label 10, " + str(i) +" estimators, score: " + str(sumScore/10))

print("-----------------------------------")



# label Sepsis
for i in range(1,20):
    clf = RandomForestClassifier(n_estimators=i*10)
    sumScore = 0
    for train_index, test_index in kf.split(train_features_data):
        clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 11])

        score = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 11])
        sumScore += score

    print("label 11, " + str(i) +" estimators, score: " + str(sumScore/10))
