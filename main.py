import numpy as np
import tensorflow as tf
from functions import *

# preprocessing_and_storing_data()

# load data
npz = np.load('Audiobooks_data_train.npz')
train_inputs, train_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)
npz = np.load('Audiobooks_data_validation.npz')
validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)
npz = np.load('Audiobooks_data_test.npz')
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

# defining model
input_size = train_inputs.shape[1]
output_size = 2
hidden_layer_size = 50
model = tf.keras.Sequential([
                            tf.keras.layers.Dense(hidden_layer_size, activation="relu"),
                            tf.keras.layers.Dense(hidden_layer_size, activation="relu"),
                            tf.keras.layers.Dense(output_size, activation="softmax")
                            ])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# learning
batch_size = 100
epochs = 100
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)
model.fit(train_inputs, train_targets,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=early_stopping,
          validation_data=(validation_inputs, validation_targets),
          verbose=2
          )

# testing
model.evaluate(test_inputs, test_targets)