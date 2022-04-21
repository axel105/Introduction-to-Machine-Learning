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
clf = RandomForestClassifier(n_estimators=100)
clf2 = RandomForestClassifier(n_estimators=100)
clf3 = RandomForestClassifier(n_estimators=100)
clf4 = RandomForestClassifier(n_estimators=100)
clf5 = RandomForestClassifier(n_estimators=100)
clf6 = RandomForestClassifier(n_estimators=100)
clf7 = RandomForestClassifier(n_estimators=100)
clf8 = RandomForestClassifier(n_estimators=100)
clf9 = RandomForestClassifier(n_estimators=100)
clf10 = RandomForestClassifier(n_estimators=100)
clf11 = RandomForestClassifier(n_estimators=100)
clf12 = RandomForestClassifier(n_estimators=100)

sumScore = 0
for train_index, test_index in kf.split(train_features_data):
    clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 1])

    score = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 1])
    sumScore += score
    print(score)
print(sumScore/10)

# label fibrinogen, is pretty slow but very good 0.93
sumScore2 = 0
for train_index, test_index in kf.split(train_features_data):
    clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 2])

    score2 = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 2])
    sumScore2 += score2
    print(score2)
print(sumScore2/10)

# label AST
sumScore3 = 0
for train_index, test_index in kf.split(train_features_data):
    clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 3])

    score3 = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 3])
    sumScore3 += score3
    print(score3)
print(sumScore3/10)

# label Alkalinephos
sumScore4 = 0
for train_index, test_index in kf.split(train_features_data):
    clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 4])

    score4 = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 4])
    sumScore4 += score4
    print(score4)
print(sumScore4/10)


# label Bilirubin_total
sumScore5 = 0
for train_index, test_index in kf.split(train_features_data):
    clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 5])

    score5 = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 5])
    sumScore5 += score5
    print(score5)
print(sumScore5/10)

# label lactate
sumScore6 = 0
for train_index, test_index in kf.split(train_features_data):
    clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 6])

    score6 = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 6])
    sumScore6 += score6
    print(score6)
print(sumScore6/10)

# label TroponinI
sumScore7 = 0
for train_index, test_index in kf.split(train_features_data):
    clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 7])

    score7 = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 7])
    sumScore7 += score7
    print(score7)
print(sumScore7/10)

# label SaO2
sumScore8 = 0
for train_index, test_index in kf.split(train_features_data):
    clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 8])

    score8 = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 8])
    sumScore8 += score8
    print(score8)
print(sumScore8/10)

# label Bilirubin_direct
sumScore9 = 0
for train_index, test_index in kf.split(train_features_data):
    clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 9])

    score9 = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 9])
    sumScore9 += score9
    print(score9)
print(sumScore9/10)

# label ETC02
sumScore10 = 0
for train_index, test_index in kf.split(train_features_data):
    clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 10])

    score10 = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 10])
    sumScore10 += score10
    print(score10)
print(sumScore10/10)

# label Sepsis
sumScore11 = 0
for train_index, test_index in kf.split(train_features_data):
    clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 11])

    score11 = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 11])
    sumScore11 += score11
    print(score11)
print(sumScore11/10)