import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

#load data into 2d numpy array, leaving out the header line (containing labels)
my_data = np.genfromtxt('train.csv', delimiter=',', skip_header=1)

# initliaze arrays containing data points and values
x_array = np.array([])
y_array = np.array([])

for i in range(len(my_data)):
    temp = my_data[i]
    x_array = np.append(x_array, temp[1:])
    y_array = np.append(y_array, temp[0])

x_array = np.reshape(x_array, (len(my_data), 13))

#try 2
# get index arrays for the 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=0)

# create ridge regression model with lamda = 1.0 and tolerance = machine precision
ridgeReg = linear_model.Ridge(alpha=1.0, tol=np.finfo(float).eps)
errorSum = 0.0
for train_index, test_index in kf.split(x_array):
    # per iteration, the arrays contain the indices for the 9 train folds and 1 test fault respectively
    # print(train_index)
    # print(test_index)

    # train model with train folds
    ridgeReg.fit(x_array[train_index], y_array[train_index])

    # compute the RMSE (with squared false flag) of the test fold
    error = mean_squared_error(y_array[test_index], ridgeReg.predict(x_array[test_index]), squared=False)

    errorSum += error

print(errorSum)






#try 1
clf = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 200.0], scoring='neg_mean_squared_error', store_cv_values=True)
clf.fit(x_array, y_array)
#print(clf.cv_values_)


