import numpy as np
import math as m
import csv

#load data into 2d numpy array, leaving out the header line (containing labels)
my_data = np.genfromtxt('train.csv', delimiter=',', skip_header=1)

# initliaze arrays containing data points and values
x_features_prelim = np.array([])
y_array = np.array([])

for i in range(len(my_data)):
    temp = my_data[i]
    x_features_prelim = np.append(x_features_prelim, temp[2:])
    y_array = np.append(y_array, temp[1])

x_features_prelim = np.reshape(x_features_prelim, (len(my_data), 5))


# create feature array from given data points according to formulas
x_array = np.array([])
for x in x_features_prelim:
    x_array = np.append(x_array, x[0])
    x_array = np.append(x_array, x[1])
    x_array = np.append(x_array, x[2])
    x_array = np.append(x_array, x[3])
    x_array = np.append(x_array, x[4])
    x_array = np.append(x_array, x[0] ** 2)
    x_array = np.append(x_array, x[1] ** 2)
    x_array = np.append(x_array, x[2] ** 2)
    x_array = np.append(x_array, x[3] ** 2)
    x_array = np.append(x_array, x[4] ** 2)
    x_array = np.append(x_array, m.exp(x[0]))
    x_array = np.append(x_array, m.exp(x[1]))
    x_array = np.append(x_array, m.exp(x[2]))
    x_array = np.append(x_array, m.exp(x[3]))
    x_array = np.append(x_array, m.exp(x[4]))
    x_array = np.append(x_array, m.cos(x[0]))
    x_array = np.append(x_array, m.cos(x[1]))
    x_array = np.append(x_array, m.cos(x[2]))
    x_array = np.append(x_array, m.cos(x[3]))
    x_array = np.append(x_array, m.cos(x[4]))
    x_array = np.append(x_array, 1)

x_array = np.reshape(x_array, (len(my_data), 21))

print(x_array)