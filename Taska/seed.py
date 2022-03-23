import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import csv

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
optval1 = 10
optseed1 = 0
optval2 = 10
optseed2 = 0
optval3 = 10
optseed3 = 0
optval4 = 10
optseed4 = 0
optval5 = 10
optseed5 = 0


for i in range(0,1000):

    kf = KFold(n_splits=10, shuffle=True, random_state=i)

    # create ridge regression model with lamda = 1.0 and tolerance = machine precision
    ridgeReg1 = linear_model.Ridge(alpha=0.1, tol=np.finfo(float).eps)
    ridgeReg2 = linear_model.Ridge(alpha=1.0, tol=np.finfo(float).eps)
    ridgeReg3 = linear_model.Ridge(alpha=10.0, tol=np.finfo(float).eps)
    ridgeReg4 = linear_model.Ridge(alpha=100.0, tol=np.finfo(float).eps)
    ridgeReg5 = linear_model.Ridge(alpha=200.0, tol=np.finfo(float).eps)

    errorSum1 = 0.0
    errorSum2 = 0.0
    errorSum3 = 0.0
    errorSum4 = 0.0
    errorSum5 = 0.0

    for train_index, test_index in kf.split(x_array):
        # per iteration, the arrays contain the indices for the 9 train folds and 1 test fault respectively
        # print(train_index)
        # print(test_index)

        # train model with train folds
        ridgeReg1.fit(x_array[train_index], y_array[train_index])
        ridgeReg2.fit(x_array[train_index], y_array[train_index])
        ridgeReg3.fit(x_array[train_index], y_array[train_index])
        ridgeReg4.fit(x_array[train_index], y_array[train_index])
        ridgeReg5.fit(x_array[train_index], y_array[train_index])

        # compute the RMSE (with squared false flag) of the test fold
        error1 = mean_squared_error(y_array[test_index], ridgeReg1.predict(x_array[test_index]), squared=False)
        error2 = mean_squared_error(y_array[test_index], ridgeReg2.predict(x_array[test_index]), squared=False)
        error3 = mean_squared_error(y_array[test_index], ridgeReg3.predict(x_array[test_index]), squared=False)
        error4 = mean_squared_error(y_array[test_index], ridgeReg4.predict(x_array[test_index]), squared=False)
        error5 = mean_squared_error(y_array[test_index], ridgeReg5.predict(x_array[test_index]), squared=False)

        errorSum1 += error1
        errorSum2 += error2
        errorSum3 += error3
        errorSum4 += error4
        errorSum5 += error5

    average1 = errorSum1 / 10
    average2 = errorSum2 / 10
    average3 = errorSum3 / 10
    average4 = errorSum4 / 10
    average5 = errorSum5 / 10

    if average1 < optval1:
        optseed1 = i
        optval1 = average1
        print(optseed1)
    if average2 < optval2:
        optseed2 = i
        optval2 = average2
        print(optseed2)
    if average3 < optval3:
        optseed3 = i
        optval3 = average3
        print(optseed3)
    if average4 < optval4:
        optseed4 = i
        optval4 = average4
        print(optseed4)
    if average5 < optval5:
        optseed5 = i
        optval5 = average5
        print(optseed5)

print(optseed1)
print(optseed2)
print(optseed3)
print(optseed4)
print(optseed5)

# open the file in the write mode
with open("submission.csv", 'w') as f:

    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    writer.writerow([average1])
    writer.writerow([average2])
    writer.writerow([average3])
    writer.writerow([average4])
    writer.writerow([average5])

    # close the file
    f.close()
