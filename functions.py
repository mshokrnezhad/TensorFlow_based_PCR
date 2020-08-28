import numpy as np
from sklearn import preprocessing

def preprocessing_and_storing_data():
    # loading raw dataset
    raw_csv_data = np.loadtxt('Audiobooks_data.csv', delimiter=',')
    unscaled_inputs_all = raw_csv_data[:,1:-1]
    targets_all = raw_csv_data[:,-1]

    # balancing dataset
    number_of_one_targets = int(np.sum(targets_all))
    number_of_zero_targets = 0
    indexes_to_remove = []
    for i in range(targets_all.shape[0]):
        if targets_all[i] == 0:
            number_of_zero_targets += 1
            if number_of_zero_targets > number_of_one_targets:
                indexes_to_remove.append(i)
    unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indexes_to_remove, axis=0)
    targets_equal_priors = np.delete(targets_all, indexes_to_remove, axis=0)

    # standardize the inputs
    scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

    # shuffle the data
    shuffled_indices = np.arange(scaled_inputs.shape[0])
    np.random.shuffle(shuffled_indices)
    shuffled_inputs = scaled_inputs[shuffled_indices]
    shuffled_targets = targets_equal_priors[shuffled_indices]

    # split the dataset into train, validation, and test
    samples_count = shuffled_inputs.shape[0]
    train_samples_count = int(0.8 * samples_count)
    validation_samples_count = int(0.1 * samples_count)
    test_samples_count = samples_count - train_samples_count - validation_samples_count
    train_inputs = shuffled_inputs[:train_samples_count]
    train_targets = shuffled_targets[:train_samples_count]
    validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
    validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]
    test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
    test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

    # check if the data is balanced
    print(np.sum(train_targets), train_samples_count, np.sum(train_targets)/train_samples_count)
    print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets)/validation_samples_count)
    print(np.sum(test_targets), test_samples_count, np.sum(test_targets)/test_samples_count)

    # save data into .npz files
    np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
    np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
    np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)