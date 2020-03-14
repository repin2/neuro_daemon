from data_engineering import get_data
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import random
import pickle
import json
import numpy as np
from PIL import Image


def get_idx(pred_list, n):
    helper = [(i, x) for i, x in enumerate(pred_list)]
    helper = sorted(helper, key= lambda x: x[1])
    return [x[0] for x in helper[-n:]]

def get_nn(i1, data, values):
    #data, values, correct_values = get_data()

    data = data / 255.

    data = np.reshape(data, data.shape + (1, ))

    data_train, val_train = [], []
    data_test, val_test = [], []

    for i in range(data.shape[0]):
        if random.randint(0, 8) not in range(0, 6):
            data_train.append(data[i])
            val_train.append(values[i])
        else:
            data_test.append(data[i])
            val_test.append(values[i])

    data_train = np.array(data_train)
    data_test = np.array(data_test)
    val_train = np.array(val_train)
    val_test = np.array(val_test)


    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(32, 32)),
    #     keras.layers.Dense(256, activation=tf.nn.sigmoid),
    #     keras.layers.Dense(10, activation=tf.nn.softmax)
    # ])

    model = keras.Sequential()
    model.add(keras.layers.Convolution2D(20, (2, 2),  input_shape=(32, 32, 1), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.8))
    model.add(keras.layers.Convolution2D(12, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dropout(0.8))
    model.add(keras.layers.Dense(9, activation='sigmoid'))


    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(data_train, val_train, epochs=1, batch_size=10)
    test_loss, test_acc = model.evaluate(data_test, val_test)
    prediction = model.predict(data_test)
    prediction_list = [get_idx(x, 1) for x in prediction]

    good_num, bad_num = 0, 0
    for i, num in enumerate(prediction_list):
        if any([val_test[i][x] for x in num]):
            good_num += 1
        else:
            bad_num += 1
    print("RESULT IS " + str(good_num / (good_num + bad_num)))

    print("TRAINING " + str(i1) + "IS OVER" + str(test_acc))
    model.save(os.path.join('models', 'model_' + str(i1) + '.hdf5'))


data, values, correct_values = get_data()

data_train, val_train, correct_values_train = [], np.array([]), np.array([])
data_test, val_test, correct_values_test = [], np.array([]), np.array([])


for i in range(data.shape[0]):
    if random.randint(0, 15) not in range(0, 1):
        data_train.append(data[i])
        val_train = np.append(val_train, np.array([values[i]]))
        correct_values_train = np.append(correct_values_train, correct_values[i])
    else:
        data_test.append(data[i])
        val_test = np.append(val_test, np.array([values[i]]))
        correct_values_test = np.append(correct_values_test, correct_values[i])

data_train = np.array(data_train)
data_test = np.array(data_test)

val_train = keras.utils.to_categorical(val_train, 9)
val_test = keras.utils.to_categorical(val_test, 9)

with open (os.path.join('data', 'test_data.sav'), 'wb') as tdada:
    pickle.dump(data_test, tdada)

with open (os.path.join('data', 'test_vals.sav'), 'wb') as tdada:
    pickle.dump(val_test, tdada)

with open (os.path.join('data', 'test_corr_vals.sav'), 'wb') as tdada:
    pickle.dump(correct_values_test, tdada)


for i in range(30):

    get_nn(i, data_train, val_train)
