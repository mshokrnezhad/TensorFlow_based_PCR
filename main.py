import numpy as np
from sklearn import preprocessing

# loading raw dataset
raw_csv_data = np.loadtxt('Audiobooks_data.csv', delimiter=',')
unscaled_inputs_all = raw_csv_data[:,1:-1]
targets_all = raw_csv_data[:,-1]

# balancing dataset
number_of_one_targets = int(np.sum(targets_all))
number_of_zero_targets = 0
indexes_to_remove = []
for i in range(len(targets_all)):
    if targets_all[i] == 0:
        number_of_one_targets += 1
        if number_of_zero_targets > number_of_one_targets:
            indexes_to_remove.append(i)
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indexes_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indexes_to_remove, axis=0)

