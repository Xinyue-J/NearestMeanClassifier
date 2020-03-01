from plotDecBoundaries import plotDecBoundaries
import csv
import numpy as np


def disteclud(vec1, vec2):  # calculate Euclidean Distance
    return np.sqrt(np.sum(np.power(vec1 - vec2, 2)))


training = np.zeros((100, 2))
label_train = np.zeros((100), dtype=np.int)
sample_mean = np.zeros((2, 2))
test = np.zeros((100, 2))
label_test = np.zeros(100, dtype=np.int)

sumx1 = 0
sumx2 = 0
sumy1 = 0
sumy2 = 0
i = 0
j = 0
k = 0
q = 0
error_train = 0
error_test = 0
class1_cnt = 0
class2_cnt = 0

with open('python3/synthetic2_train.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        training[i, 0] = float(row[0])
        training[i, 1] = float(row[1])
        label_train[i] = float(row[2])
        i += 1

        if int(row[2]) == 1:
            sumx1 = sumx1 + float(row[0])
            sumy1 = sumy1 + float(row[1])
            class1_cnt += 1
        else:
            sumx2 = sumx2 + float(row[0])
            sumy2 = sumy2 + float(row[1])
            class2_cnt += 1

sample_mean[0, 0] = sumx1 / class1_cnt
sample_mean[0, 1] = sumy1 / class1_cnt
sample_mean[1, 0] = sumx2 / class2_cnt
sample_mean[1, 1] = sumy2 / class2_cnt

plotDecBoundaries(training, label_train, sample_mean)

# error rate for train
for axis_train in training:
    d1 = disteclud(axis_train, sample_mean[0])
    d2 = disteclud(axis_train, sample_mean[1])
    if d1 > d2 and label_train[j] == 1:
        error_train += 1
    if d1 < d2 and label_train[j] == 2:
        error_train += 1
    j += 1

print("error rate for train:", error_train / i)

# error rate for test
with open('python3/synthetic2_test.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        test[k, 0] = float(row[0])
        test[k, 1] = float(row[1])
        label_test[k] = float(row[2])
        k += 1

for axis_test in test:
    d1 = disteclud(axis_test, sample_mean[0])
    d2 = disteclud(axis_test, sample_mean[1])
    if d1 > d2 and label_test[q] == 1:
        error_test += 1
    if d1 < d2 and label_test[q] == 2:
        error_test += 1
    q += 1
print("error rate for test:", error_test / i)
print("sample mean:", sample_mean)
