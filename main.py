import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import tree
from sklearn.model_selection import KFold


# read in data from csv files
cols = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
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
test_features_data = np.reshape(test_features_data, (int(pat_nr), 12, 36))

# reshape train features into 3D array
pat_nr2 = len(train_features_data)/12
train_features_data = np.ravel(train_features_data, order="C")
train_features_data = np.reshape(train_features_data, (int(pat_nr2), 12, 36))

# create 10-fold for validation
kf = KFold(n_splits=10, shuffle=True)

clf = tree.DecisionTreeClassifier()

for train_index, test_index in kf.split(train_features_data):
    clf.fit(train_features_data[train_index], train_labels_data[train_index][:, 1])

    score = clf.score(train_features_data[test_index], train_labels_data[test_index][:, 1])
    print(score)




















