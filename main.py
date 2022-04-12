import csv
import numpy as np

test_features_data = np.genfromtxt('test_features.csv', delimiter=',', skip_header=1)
train_features_data = np.genfromtxt('train_features.csv', delimiter=',', skip_header=1)
train_labels_data = np.genfromtxt('train_labels.csv', delimiter=',', skip_header=1)

# each patient has 12 * 36 entries in features
test_feature = np.array([])
train_features = np.array([])
train_labels = np.array([])
print(test_features_data)
for i in range(len(test_features_data)):
    temp = test_features_data[i]
    test_feature = np.append(test_feature, temp[1:])
    print(i)






