'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function
import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adagrad
from keras.utils import np_utils
from keras.callbacks import History
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.noise import GaussianDropout
from keras.models import model_from_json
import keras
# from keras.models import load_model
import pdb
import csv
import numpy as np


    
filenames_model = ['Original.h5', 'Part1.h5', 'Part2.h5', 'Part3.h5', 'Part4_1.h5', 'Part4_2.h5']
filenames_data = ['Original.csv', 'Part1.csv', 'Part2.csv', 'Part3.csv', 'Part4_1.csv', 'Part4_2.csv']
filenames_json = [fn.replace('csv', 'json') for fn in filenames_data]
i_iter = 0


batch_size = 32
nb_classes = 10
nb_epoch = 80
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# X_train = X_train[0:100]
# y_train = y_train[0:100]
# X_test = X_test[0:100]
# Y_test = Y_test[0:100]
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
# for i_iter in [range(len(filenames_data))]:
for i_iter in [1]:


    model = Sequential()
    model.add(AveragePooling2D(pool_size=(2,2), input_shape=(img_channels, img_rows, img_cols)))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    
    # model.add(Dropout(0.5)) 
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # if i_iter == 1: #only remove this in step 1
    #     model.layers.pop(-3)

    if i_iter == 2:
        # load json and create model from previous iteration
        json_file = open('Part1.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        # load model
        model = model_from_json(loaded_model_json)
        # load weights from previous run into new model
        model.load_weights('Part1.h5')
        # model = model.load_weights('Part1.h5')
        model2 = Sequential()
        model2.layers = model.layers[0:-2]
        model2.layers.append(Convolution2D(512, 3, 3, border_mode='same'))
        # model.layers.add(Convolution2D(512, 3, 3, border_mode='same'))
        # model.add(Dense(nb_classes))
        # model.add(Activation('softmax'))
        model2.layers.append(model.layers[-2])
        model2.layers.append(model.layers[-1])
        model = model2

    if i_iter == 3:
        adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)

    if i_iter == 4:
        ReLU_layers = [num for num,l in enumerate(model.layers) if l.name.split('_')[0] == Activation('relu').name.split('_')[0]]
        for i in ReLU_layers:
            model.layers[i] = LeakyReLU(alpha = 0.1)

    if i_iter == 5:
        Dropout_layers = [num for num, l in enumerate(model.layers) if l.name.split('_')[0] == Dropout(0.25).name.split('_')[0]]
        Dropout_amounts = np.array([0.5, 0.5, 0.75])
        i_amount = 0
        for i in Dropout_layers:
            model.layers[i] = GaussianDropout(Dropout_amounts[i_amount])
            i_amount += 1

    # pdb.set_trace()
    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    if i_iter == 3:
        model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=["accuracy"])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

    

    # X_train = X_train.astype('float32')
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    history = History
    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(X_train, Y_train, batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, Y_test), shuffle=True)
    else:
        print('Using real-time data augmentation.')

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        # fit the model on the batches generated by datagen.flow()
        history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch, show_accuracy=True,
                            validation_data=(X_test, Y_test),
                            nb_worker=1)
      
    model.save_weights(filenames_model[i_iter], overwrite=True)
    model_json = model.to_json()
    with open(filenames_json[i_iter], "w") as json_file:
        json_file.write(model_json)

    print(history.history.keys())
    with open(filenames_data[i_iter], 'wb') as csv_file:
        writer = csv.writer(csv_file, history.history.keys())
        writer.writerows(history.history.items())
        # writer.writeheader()
        # for i in range(len(history.history['acc'])):
        #     writer.writerow(history.history)





