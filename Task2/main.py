import csv

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import linear_model
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
pid_array = np.genfromtxt('test_features.csv', delimiter=',', skip_header=1, usecols=(0))

# impute missing values into test_features_data, with mean method
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

# fit imp with train features and test features
imp.fit(np.concatenate((train_features_data, test_features_data), axis=0))
SimpleImputer()
test_features_data = imp.transform(test_features_data)

# impute missing values into train_features_data, with mean method
train_features_data = imp.transform(train_features_data)

# reshape test features into 3D array
pat_nr = len(test_features_data) / 12
test_features_data = np.ravel(test_features_data, order="C")
test_features_data = np.reshape(test_features_data, (int(pat_nr), 12 * 35))

# reshape train features into 3D array
pat_nr2 = len(train_features_data) / 12
train_features_data = np.ravel(train_features_data, order="C")
train_features_data = np.reshape(train_features_data, (int(pat_nr2), 12 * 35))

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

# create all classifiers
clf = RandomForestClassifier(n_estimators=170, random_state=0) # score = 0.875440148554641
clf2 = RandomForestClassifier(n_estimators=170, random_state=0) # score = 0.9344563066433856
clf3 = RandomForestClassifier(n_estimators=100, random_state=0) # score = 0.7812040409079571
clf4 = RandomForestClassifier(n_estimators=150, random_state=0) # score = 0.7855222693384329
clf5 = RandomForestClassifier(n_estimators=100, random_state=0) # score = 0.7810482525428896
clf6 = RandomForestClassifier(n_estimators=150, random_state=0) # score = 0.8389575122640724
clf7 = RandomForestClassifier(n_estimators=180, random_state=0) # score = 0.9057121476677474
clf8 = RandomForestClassifier(n_estimators=170, random_state=0) # score = 0.8322741886311356
clf9 = RandomForestClassifier(n_estimators=120, random_state=0) # score = 0.9676755910312907
clf10 = RandomForestClassifier(n_estimators=140, random_state=0) # score = 0.9712029600066516
clf11 = RandomForestClassifier(n_estimators=100, random_state=0)

# fit all classifiers
print("Let's get started!")
clf.fit(train_features_data, train_labels_data[:, 1])
print("first done, 14 more to go")
clf2.fit(train_features_data, train_labels_data[:, 2])
clf3.fit(train_features_data, train_labels_data[:, 3])
print("3 done, 12 more to go")
clf4.fit(train_features_data, train_labels_data[:, 4])
clf5.fit(train_features_data, train_labels_data[:, 5])
clf6.fit(train_features_data, train_labels_data[:, 6])
print("6 done, 9 more to go")
clf7.fit(train_features_data, train_labels_data[:, 7])
clf8.fit(train_features_data, train_labels_data[:, 8])
clf9.fit(train_features_data, train_labels_data[:, 9])
clf10.fit(train_features_data, train_labels_data[:, 10])
print("10 done, almost there")
# sepsis prediction
clf11.fit(train_features_data, train_labels_data[:, 11])
print("gz you made it")

# ridge regression, sub task 3
ridgeReg = linear_model.Ridge(alpha=0.1, tol=np.finfo(float).eps)
ridgeReg2 = linear_model.Ridge(alpha=0.1, tol=np.finfo(float).eps)
ridgeReg3 = linear_model.Ridge(alpha=0.1, tol=np.finfo(float).eps)
ridgeReg4 = linear_model.Ridge(alpha=0.1, tol=np.finfo(float).eps)

# fit regression
ridgeReg.fit(train_features_data, train_labels_data[:, 12])
ridgeReg2.fit(train_features_data, train_labels_data[:, 13])
ridgeReg3.fit(train_features_data, train_labels_data[:, 14])
ridgeReg4.fit(train_features_data, train_labels_data[:, 15])
print("WOW! all models fitted")

results = np.zeros((int(pat_nr), 16))

i = 0
j = 0
for patient in test_features_data:
    patient = np.reshape(patient, (1, 12*35))
    print(str(pid_array[i]))
    results[j][0] = int(pid_array[i])
    results[j][1] = clf.predict_proba(patient)[0][1]
    results[j][2] = clf2.predict_proba(patient)[0][1]
    results[j][3] = clf3.predict_proba(patient)[0][1]
    results[j][4] = clf4.predict_proba(patient)[0][1]
    results[j][5] = clf5.predict_proba(patient)[0][1]
    results[j][6] = clf6.predict_proba(patient)[0][1]
    results[j][7] = clf7.predict_proba(patient)[0][1]
    results[j][8] = clf8.predict_proba(patient)[0][1]
    results[j][9] = clf9.predict_proba(patient)[0][1]
    results[j][10] = clf10.predict_proba(patient)[0][1]
    results[j][11] = clf11.predict_proba(patient)[0][1]
    results[j][12] = ridgeReg.predict(patient)
    results[j][13] = ridgeReg2.predict(patient)
    results[j][14] = ridgeReg3.predict(patient)
    results[j][15] = ridgeReg4.predict(patient)

    i += 12
    j += 1

TESTS = ['pid', 'LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2','LABEL_Sepsis', 'LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2',
         'LABEL_Heartrate']

pd.DataFrame(results).to_csv('prediction.csv', index=False, header=TESTS, float_format='%.3f')




