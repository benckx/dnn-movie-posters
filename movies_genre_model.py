import os
import time

import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

import movies_dataset as movies


def get_kernel_dimensions(version, shape, divisor):
    image_width = shape[1]

    # original
    if version == 1:
        return 3, 3

    # square 10% width
    if version == 2:
        return int(0.1 * image_width / divisor), int(0.1 * image_width / divisor)

    # square 20% width
    if version == 3:
        return int(0.2 * image_width / divisor), int(0.2 * image_width / divisor)


def build(version, min_year, max_year, genres, ratio, epochs,
          x_train=None, y_train=None, x_validation=None, y_validation=None):
    # log
    print()
    print('version:', version)
    print('min_year:', min_year)
    print('max_year:', max_year)
    print('genres:', genres)
    print('ratio:', ratio)
    print()

    # load data if not provided
    if x_train is None or y_train is None or x_validation is None or y_validation is None:
        begin = time.time()
        x_train, y_train = movies.load_genre_data(min_year, max_year, genres, ratio, 'train')
        x_validation, y_validation = movies.load_genre_data(min_year, max_year, genres, ratio, 'validation')
        print('loaded in', (time.time() - begin) / 60, 'min.')
    else:
        print('data provided in arguments')

    print()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_validation.shape[0], 'validation samples')

    # build model
    num_classes = len(y_train[0])
    kernel_dimensions1 = get_kernel_dimensions(version, x_train.shape, 1)
    kernel_dimensions2 = get_kernel_dimensions(version, x_train.shape, 2)
    print('kernel_dimensions1:', kernel_dimensions1)
    print('kernel_dimensions2:', kernel_dimensions2)

    model = Sequential([
        Conv2D(32, kernel_dimensions1, padding='same', input_shape=x_train.shape[1:], activation='relu'),
        Conv2D(32, kernel_dimensions1, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, kernel_dimensions2, padding='same', activation='relu'),
        Conv2D(64, kernel_dimensions2, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')
    ])

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_validation, y_validation))

    # create dir if none
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # save model
    model_file_name = 'genres' \
                      + '_' + str(min_year) + '_' + str(max_year) \
                      + '_g' + str(len(genres)) \
                      + '_r' + str(ratio) \
                      + '_e' + str(epochs) \
                      + '_v' + str(version) + '.h5'

    model_path = os.path.join(save_dir, model_file_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
