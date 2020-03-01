import csv
import numpy as np
from plotDecBoundaries import plotDecBoundaries

data_train = np.zeros((89, 13))
label_train = np.zeros(89, dtype=np.int)
mean_train = np.zeros((78, 6))
data_test = np.zeros((89, 13))
label_test = np.zeros(89, dtype=np.int)
index_train = np.zeros((78, 2))
index_test = np.zeros((78, 2))

trip_cnt_train = 0
trip_cnt_test = 0
error_cnt_train = 0
error_all_train = []
error_rate_train = []
error_cnt_test = 0
error_all_test = []
error_rate_test = []
i_train = 0
i_test = 0
j_train = 0
j_test = 0

with open('python3/wine_train.csv') as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        for j_train in range(0, 13):
            data_train[i_train, j_train] = float(row[j_train])

        label_train[i_train] = int(row[13])
        i_train += 1

# main 2-dimension loop to calculate error rate
for dim_x_train in range(0, 12):
    for dim_y_train in range(dim_x_train + 1, 13):

        index_train[trip_cnt_train, 0] = dim_x_train
        index_train[trip_cnt_train, 1] = dim_y_train

        sum1_x = 0
        sum1_y = 0
        sum2_x = 0
        sum2_y = 0
        sum3_x = 0
        sum3_y = 0
        class1_cnt = 0
        class2_cnt = 0
        class3_cnt = 0

        train_select = np.hstack((data_train[:, dim_x_train].reshape(89, 1), data_train[:, dim_y_train].reshape(89, 1)))

        # after a new 89 * 2 array is obtained:
        # calculate sum based on class
        for row_cnt_train in range(0, 89):
            if label_train[row_cnt_train] == 1:
                sum1_x += train_select[row_cnt_train, 0]
                sum1_y += train_select[row_cnt_train, 1]
                class1_cnt += 1
            elif label_train[row_cnt_train] == 2:
                sum2_x += train_select[row_cnt_train, 0]
                sum2_y += train_select[row_cnt_train, 1]
                class2_cnt += 1
            else:
                sum3_x += train_select[row_cnt_train, 0]
                sum3_y += train_select[row_cnt_train, 1]
                class3_cnt += 1

        # calculate the sample mean
        mean_train[trip_cnt_train, 0] = sum1_x / class1_cnt
        mean_train[trip_cnt_train, 1] = sum1_y / class1_cnt
        mean_train[trip_cnt_train, 2] = sum2_x / class2_cnt
        mean_train[trip_cnt_train, 3] = sum2_y / class2_cnt
        mean_train[trip_cnt_train, 4] = sum3_x / class3_cnt
        mean_train[trip_cnt_train, 5] = sum3_y / class3_cnt

        mean_train_reshaped = mean_train[trip_cnt_train, :].reshape(3, 2)
        trip_cnt_train += 1

        # calculate distance and compare
        for m in range(0, 89):

            d1_train = np.sqrt(np.sum(np.square(train_select[m] - mean_train_reshaped[0])))
            d2_train = np.sqrt(np.sum(np.square(train_select[m] - mean_train_reshaped[1])))
            d3_train = np.sqrt(np.sum(np.square(train_select[m] - mean_train_reshaped[2])))

            # count error data points
            # not taking points on boundary into consideration
            if d1_train < d2_train and d1_train < d3_train and label_train[m] != 1:
                error_cnt_train += 1
            elif d2_train < d1_train and d2_train < d3_train and label_train[m] != 2:
                error_cnt_train += 1
            elif d3_train < d2_train and d3_train < d1_train and label_train[m] != 3:
                error_cnt_train += 1

        # write current number of error points into the array
        error_all_train.append(error_cnt_train)
        # write current error rate into the array
        error_rate_train.append(error_cnt_train / i_train)
        # reset error counter
        error_cnt_train = 0

# print the final result
# looking for the data with min error rate
n_test = 0
for n in range(0, 78):
    # if find the min error rate data
    if error_all_train[n] == min(error_all_train):
        n_test = n
        first_row = int(index_train[n, 0])
        second_row = int(index_train[n, 1])
        # looking for x and y in index_recorder
        result_reshaped = np.hstack((data_train[:, int(index_train[n, 0])].reshape(89, 1), data_train[:, int(index_train[n, 1])].reshape(89, 1)))
        # looking for sample mean in mean_recorder
        # transform the 1 * 6 array into 3 * 2 array
        mean_reshaped = mean_train[n, :].reshape(3, 2)

        print("train first feature:", int(index_train[n, 0] + 1))
        print("train second feature:", int(index_train[n, 1] + 1))
        print("train min error rate:", min(error_rate_train))
        print("train error rate standard deviation:", (np.var(error_rate_train)) ** 0.5)

        plotDecBoundaries(result_reshaped, label_train, mean_reshaped)

with open('python3/wine_test.csv') as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        for j_test in range(0, 13):
            data_test[i_test, j_test] = float(row[j_test])

        label_test[i_test] = int(row[13])
        i_test += 1


for dim_x_test in range(0, 12):
    for dim_y_test in range(dim_x_test + 1, 13):
        index_test[trip_cnt_test, 0] = dim_x_test
        index_test[trip_cnt_test, 1] = dim_y_test

        test_select = np.hstack((data_test[:, dim_x_test].reshape(89, 1), data_test[:, dim_y_test].reshape(89, 1)))
        mean_test_reshaped = mean_train[trip_cnt_test, :].reshape(3, 2)

        trip_cnt_test += 1

        for m in range(0, 89):
            d1_test = np.sqrt(np.sum(np.square(test_select[m] - mean_test_reshaped[0])))
            d2_test = np.sqrt(np.sum(np.square(test_select[m] - mean_test_reshaped[1])))
            d3_test = np.sqrt(np.sum(np.square(test_select[m] - mean_test_reshaped[2])))

            if d1_test < d2_test and d1_test < d3_test and label_test[m] != 1:
                error_cnt_test += 1
            elif d2_test < d1_test and d2_test < d3_test and label_test[m] != 2:
                error_cnt_test += 1
            elif d3_test < d2_test and d3_test < d1_test and label_test[m] != 3:
                error_cnt_test += 1

        error_all_test.append(error_cnt_test)
        error_rate_test.append(error_cnt_test / 89)
        error_cnt_test = 0

print("test error rate standard deviation:", (np.var(error_rate_test)) ** 0.5)
print("test min error rate:", error_rate_test[n_test])


